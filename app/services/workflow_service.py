"""Operational Workflow Integration Layer (orchestration only).

This service stitches the existing engines into one unified deal view.
It does NOT:
  - re-implement any scoring logic,
  - introduce new thresholds for investment / acquisition / renovation,
  - mutate the underlying services.

Flow per request:
  1. Run ``analyze_investments``    -> InvestmentOutput per area.
  2. Run ``analyze_acquisitions``   -> acquisition decision per area.
  3. Run ``analyze_renovations``    -> renovation evaluation per area.
  4. Join the three on ``area`` and project a single operational record
     per area with a workflow priority, next action, confidence, pipeline
     stage, and a short business summary.

Priority / next_action / stage are derived from the already-computed
decisions of the sub-engines. No new scoring is introduced.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from app.models.investment_input import InvestmentInput
from app.services.acquisition_service import analyze_acquisitions
from app.services.investment_service import (
    analyze_investments,
    get_data_source_status,
)
from app.services.renovation_service import analyze_renovations


Priority = Literal["HIGH", "MEDIUM", "LOW", "DROP"]
PipelineStage = Literal["LEAD", "UNDER_REVIEW", "APPROVED"]


_RENOVATE_ROI_HIGH_THRESHOLD = 0.20


def _derive_priority(
    acquisition_decision: Optional[str],
    renovation_decision: Optional[str],
    renovation_roi: Optional[float],
) -> Priority:
    """Priority rules (from spec):

        BUY + RENOVATE + ROI > 20% -> HIGH
        BUY only                   -> MEDIUM
        HOLD                       -> LOW
        PASS                       -> DROP
    """
    if acquisition_decision == "PASS":
        return "DROP"
    if acquisition_decision == "HOLD":
        return "LOW"
    if acquisition_decision == "BUY":
        if (
            renovation_decision == "RENOVATE"
            and renovation_roi is not None
            and renovation_roi > _RENOVATE_ROI_HIGH_THRESHOLD
        ):
            return "HIGH"
        return "MEDIUM"
    return "LOW"


def _derive_next_action(
    acquisition_decision: Optional[str],
    renovation_decision: Optional[str],
) -> str:
    if acquisition_decision == "BUY" and renovation_decision == "RENOVATE":
        return "Send to acquisitions + project team"
    if acquisition_decision == "BUY":
        return "Send to acquisitions"
    if acquisition_decision == "HOLD":
        return "Monitor"
    if acquisition_decision == "PASS":
        return "Discard"
    return "Monitor"


def _derive_stage(priority: Priority) -> PipelineStage:
    """Map workflow priority onto an operational pipeline stage."""
    if priority == "HIGH":
        return "APPROVED"
    if priority == "MEDIUM":
        return "UNDER_REVIEW"
    return "LEAD"


def _combine_confidence(
    investment_confidence: Optional[float],
    renovation_roi: Optional[float],
    risk_score: Optional[float],
) -> float:
    """Confidence = investment engine's confidence, shaded by risk.

    This is not a new scoring model; it is a deterministic shading that
    penalises borderline / risky deals without introducing new weights.
    """
    base = investment_confidence if investment_confidence is not None else 0.5
    base = max(0.0, min(1.0, base))
    if risk_score is not None:
        risk = max(0.0, min(1.0, risk_score))
        base *= 1.0 - 0.25 * risk
    if renovation_roi is not None and renovation_roi < 0.0:
        base *= 0.85
    return round(max(0.0, min(1.0, base)), 4)


def _summary(
    area: str,
    acquisition_decision: Optional[str],
    renovation_decision: Optional[str],
    roi_estimate: Optional[float],
    risk_score: Optional[float],
    renovation_roi: Optional[float],
    renovation_potential: Optional[str],
    strategy: Optional[str],
) -> str:
    roi_pct = (roi_estimate or 0.0) * 100.0
    risk_pct = (risk_score or 0.0) * 100.0
    reno_roi_pct = (renovation_roi or 0.0) * 100.0
    pot = (renovation_potential or "unknown").lower()

    if acquisition_decision == "BUY" and renovation_decision == "RENOVATE":
        return (
            f"{area} clears the acquisition hurdle (~{roi_pct:.1f}% ROI, "
            f"risk {risk_pct:.0f}/100) and shows {pot} value-add headroom "
            f"via {strategy or 'renovation'} (~{reno_roi_pct:.1f}% uplift "
            f"ROI): progress as a full buy-and-improve play."
        )
    if acquisition_decision == "BUY":
        return (
            f"{area} is acquisition-grade at ~{roi_pct:.1f}% ROI with "
            f"risk {risk_pct:.0f}/100; renovation decision is "
            f"'{renovation_decision or 'N/A'}' so treat as a hold-yield buy."
        )
    if acquisition_decision == "HOLD":
        return (
            f"{area} sits on the edge (~{roi_pct:.1f}% ROI, risk "
            f"{risk_pct:.0f}/100); monitor for a better entry price or "
            f"stronger demand signals before committing capital."
        )
    if acquisition_decision == "PASS":
        return (
            f"{area} fails the return-for-risk test (~{roi_pct:.1f}% ROI, "
            f"risk {risk_pct:.0f}/100); discard from the active pipeline."
        )
    return f"{area}: insufficient signal to place in the operational pipeline."


def _index_by_area(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        area = rec.get("area")
        if isinstance(area, str) and area:
            out[area] = rec
    return out


_PRIORITY_RANK: Dict[Priority, int] = {
    "HIGH": 0,
    "MEDIUM": 1,
    "LOW": 2,
    "DROP": 3,
}


def run_workflow(request: InvestmentInput) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Execute the full operational workflow across all scored areas.

    Returns ``(records, counts_by_priority)``. ``records`` are sorted
    by priority (HIGH first), then by investment score descending, then
    by area name for stability.
    """
    investment_results = analyze_investments(request, include_excluded=True)
    acquisitions = analyze_acquisitions(request)
    renovations = analyze_renovations(request)

    acq_by_area = _index_by_area(acquisitions)
    reno_by_area = _index_by_area(renovations)

    records: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "DROP": 0}

    for inv in investment_results:
        acq = acq_by_area.get(inv.area, {})
        reno = reno_by_area.get(inv.area, {})

        acquisition_decision = acq.get("decision")
        renovation_decision = reno.get("decision")
        renovation_roi = reno.get("roi")

        # If the investment gate excluded the area (over budget, etc.),
        # force DROP so operationally it never reaches acquisitions.
        if not inv.viable:
            priority: Priority = "DROP"
        else:
            priority = _derive_priority(
                acquisition_decision=acquisition_decision,
                renovation_decision=renovation_decision,
                renovation_roi=renovation_roi,
            )

        next_action = (
            "Discard" if priority == "DROP"
            else _derive_next_action(acquisition_decision, renovation_decision)
        )
        stage = _derive_stage(priority)
        confidence = _combine_confidence(
            investment_confidence=float(inv.confidence_score),
            renovation_roi=renovation_roi,
            risk_score=float(inv.risk_score),
        )
        summary = _summary(
            area=inv.area,
            acquisition_decision=acquisition_decision,
            renovation_decision=renovation_decision,
            roi_estimate=float(inv.roi_estimate),
            risk_score=float(inv.risk_score),
            renovation_roi=renovation_roi,
            renovation_potential=reno.get("renovation_potential"),
            strategy=reno.get("strategy"),
        )

        record: Dict[str, Any] = {
            "area": inv.area,
            "acquisition_decision": acquisition_decision,
            "recommended_buy_price": acq.get("recommended_buy_price"),
            "renovation_decision": renovation_decision,
            "renovation_potential": reno.get("renovation_potential"),
            "strategy": reno.get("strategy"),
            "estimated_cost": reno.get("estimated_cost"),
            "renovation_roi": renovation_roi,
            "payback_period": reno.get("payback_period"),
            "total_investment": reno.get("total_investment"),
            "post_renovation_roi": reno.get("post_renovation_roi"),
            "value_drivers": reno.get("value_drivers", []),
            "investment_score": float(inv.investment_score),
            "roi_estimate": float(inv.roi_estimate),
            "risk_score": float(inv.risk_score),
            "viable": bool(inv.viable),
            "priority": priority,
            "next_action": next_action,
            "stage": stage,
            "confidence": confidence,
            "summary": summary,
        }
        counts[priority] = counts.get(priority, 0) + 1
        records.append(record)

    records.sort(
        key=lambda r: (
            _PRIORITY_RANK.get(r["priority"], 9),
            -float(r["investment_score"]),
            r["area"],
        )
    )
    return records, counts


def get_workflow_data_source_status() -> Dict[str, Any]:
    """Reuse the investment service's data-source status verbatim."""
    return get_data_source_status()
