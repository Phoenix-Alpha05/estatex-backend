"""Dual Intelligence service layer.

Runs the tenant engine and the investment engine in parallel for the
same set of Dubai areas, then fuses their outputs into a cross-stakeholder
view. All logic here is deterministic orchestration - no new scoring.

Design principles:
  * Each engine stays the single source of truth for its own score.
  * We only reconcile results by ``area`` name and compute a deterministic
    combined score and a plain-language tradeoff insight.
  * Empty or partial coverage (area only in one engine) is handled: the
    missing side is reported as ``None`` and the persona reflects it.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from app.models.analysis_output import (
    CompareDataSourceStatus,
    CompareRequest,
    CompareResponse,
    DualAreaScore,
)
from app.models.investment_input import (
    InvestmentHorizon,
    InvestmentInput,
    PillarWeights,
    RiskLevel,
)
from app.models.investment_output import InvestmentOutput
from app.models.tenant_input import (
    CommutePriority,
    CommuteSensitivity,
    HouseholdType,
    TenantInput,
    TenantProfile,
)
from app.models.tenant_output import TenantOutput
from app.services.investment_service import (
    analyze_investments,
    get_data_source_status as investment_status,
)
from app.services.tenant_service import (
    get_data_source_status as tenant_status,
    recommend_areas,
)

logger = logging.getLogger("app.services.analysis_service")


def _safe_enum(enum_cls, value: str, fallback):
    try:
        return enum_cls(value)
    except (ValueError, KeyError):
        return fallback


def _build_tenant_request(req: CompareRequest) -> TenantInput:
    return TenantInput(
        budget_aed=req.tenant_budget_aed,
        commute_priority=_safe_enum(
            CommutePriority, req.commute_priority, CommutePriority.MED
        ),
        commute_sensitivity=_safe_enum(
            CommuteSensitivity, req.commute_sensitivity, CommuteSensitivity.MED
        ),
        owns_car=req.owns_car,
        household_type=_safe_enum(
            HouseholdType, req.household_type, HouseholdType.SINGLE
        ),
        tenant_profile=_safe_enum(
            TenantProfile, req.tenant_profile, TenantProfile.YOUNG_PROFESSIONAL
        ),
    )


def _build_investment_request(req: CompareRequest) -> InvestmentInput:
    return InvestmentInput(
        budget_aed=req.investment_budget_aed,
        risk_level=_safe_enum(RiskLevel, req.risk_level, RiskLevel.MEDIUM),
        investment_horizon=_safe_enum(
            InvestmentHorizon, req.investment_horizon, InvestmentHorizon.MEDIUM
        ),
        pillar_weights=PillarWeights(),
    )


_PERSONA_TO_CLASSIFICATION: Dict[str, str] = {
    "balanced": "LIVE + HOLD",
    "tenant_leaning": "LIVE (INVEST LATER)",
    "investor_leaning": "HOLD (RENT ELSEWHERE)",
    "tenant_only": "LIVE ONLY",
    "investor_only": "HOLD ONLY",
    "weak": "AVOID",
}


def _classification_for(persona: str) -> str:
    return _PERSONA_TO_CLASSIFICATION.get(persona, "AVOID")


def _classify_persona(
    tenant_score: Optional[float],
    investment_score: Optional[float],
) -> Tuple[str, float]:
    """Return (persona_label, delta) in a deterministic way."""
    if tenant_score is None and investment_score is None:
        return "weak", 0.0
    if tenant_score is None:
        return "investor_only", float(investment_score or 0.0)
    if investment_score is None:
        return "tenant_only", float(-(tenant_score or 0.0))

    delta = investment_score - tenant_score
    if tenant_score < 0.35 and investment_score < 0.35:
        return "weak", delta
    if abs(delta) <= 0.08:
        return "balanced", delta
    if delta > 0.08:
        return "investor_leaning", delta
    return "tenant_leaning", delta


def _liveability_phrase(tenant_score: Optional[float]) -> str:
    if tenant_score is None:
        return "no tenant-side coverage"
    if tenant_score >= 0.75:
        return "strong liveability with favourable commute and lifestyle fit"
    if tenant_score >= 0.55:
        return "solid liveability with acceptable commute and amenities"
    if tenant_score >= 0.35:
        return "moderate liveability with noticeable tradeoffs"
    return "weak liveability fit for this profile"


def _yield_phrase(yield_pct: Optional[float]) -> str:
    if yield_pct is None:
        return "yield data unavailable"
    if yield_pct >= 7.5:
        return f"high gross rental yield (~{yield_pct:.1f}%)"
    if yield_pct >= 5.5:
        return f"stable rental yield (~{yield_pct:.1f}%)"
    if yield_pct >= 3.5:
        return f"modest rental yield (~{yield_pct:.1f}%)"
    return f"thin rental yield (~{yield_pct:.1f}%)"


def _roi_phrase(roi: Optional[float]) -> str:
    if roi is None:
        return "ROI not estimable"
    pct = roi * 100.0
    if pct >= 8.0:
        return f"strong projected ROI ({pct:.1f}%/yr)"
    if pct >= 5.0:
        return f"healthy projected ROI ({pct:.1f}%/yr)"
    if pct >= 2.5:
        return f"moderate appreciation potential ({pct:.1f}%/yr)"
    return f"limited appreciation potential ({pct:.1f}%/yr)"


def _risk_phrase(risk_score: Optional[float]) -> str:
    if risk_score is None:
        return ""
    if risk_score <= 0.33:
        return "low market risk"
    if risk_score <= 0.66:
        return "balanced market risk"
    return "elevated market risk"


def _price_phrase(price: Optional[float]) -> str:
    if price is None or price <= 0:
        return ""
    if price >= 3_000_000:
        return "premium entry ticket"
    if price >= 1_500_000:
        return "mid-market entry ticket"
    return "accessible entry ticket"


def _join_clauses(parts: List[str]) -> str:
    cleaned = [p for p in parts if p]
    return ", ".join(cleaned)


def _build_insight(
    area: str,
    tenant_score: Optional[float],
    investment_score: Optional[float],
    persona: str,
    roi: Optional[float],
    yield_pct: Optional[float],
    risk_score: Optional[float],
) -> str:
    live = _liveability_phrase(tenant_score)
    yld = _yield_phrase(yield_pct)
    roi_ph = _roi_phrase(roi)
    risk = _risk_phrase(risk_score)

    if persona == "balanced":
        return (
            f"{area}: {live}, combined with {yld} and {roi_ph}"
            + (f", against {risk}." if risk else ".")
        )
    if persona == "investor_leaning":
        return (
            f"{area}: capital-efficient with {yld} and {roi_ph}, "
            f"but {live}"
            + (f" under {risk}." if risk else ".")
        )
    if persona == "tenant_leaning":
        return (
            f"{area}: {live}, yet returns look softer with {yld} and {roi_ph}."
        )
    if persona == "tenant_only":
        return (
            f"{area}: {live}; investor lens has no coverage for this area."
        )
    if persona == "investor_only":
        return (
            f"{area}: investor-only signal with {yld} and {roi_ph}; "
            f"no tenant-side evidence for this profile."
        )
    return (
        f"{area}: weak signal from both engines - "
        f"{live}; {yld}; {roi_ph}."
    )


def _build_classification_reason(
    classification: str,
    tenant_score: Optional[float],
    investment_score: Optional[float],
    roi: Optional[float],
    yield_pct: Optional[float],
    risk_score: Optional[float],
    price: Optional[float],
) -> str:
    live = _liveability_phrase(tenant_score)
    yld = _yield_phrase(yield_pct)
    roi_ph = _roi_phrase(roi)
    risk = _risk_phrase(risk_score)
    price_ph = _price_phrase(price)

    if classification == "LIVE + HOLD":
        body = _join_clauses(
            [live, yld, roi_ph, risk, price_ph]
        )
        return (
            "Suitable both to live in and to hold as an investment: "
            f"{body}."
        )
    if classification == "LIVE (INVEST LATER)":
        body = _join_clauses([live, yld, roi_ph])
        return (
            "Primarily a place to live right now; capital case is weaker - "
            f"{body}."
        )
    if classification == "HOLD (RENT ELSEWHERE)":
        body = _join_clauses([yld, roi_ph, risk, price_ph, live])
        return (
            "Attractive as a capital play but not the best fit to live in - "
            f"{body}."
        )
    if classification == "LIVE ONLY":
        return (
            "Tenant-viable for this profile, but no investor-side evidence "
            f"({live}; investor engine has no coverage)."
        )
    if classification == "HOLD ONLY":
        body = _join_clauses([yld, roi_ph, risk, price_ph])
        return (
            "Investor-viable without tenant-side evidence for this profile - "
            f"{body}."
        )
    return (
        "Weak signal across both engines: "
        f"{_join_clauses([live, yld, roi_ph, risk])}."
    )


def _combined_score(
    tenant_score: Optional[float],
    investment_score: Optional[float],
) -> float:
    ts = tenant_score if tenant_score is not None else 0.0
    invs = investment_score if investment_score is not None else 0.0
    return max(0.0, min(1.0, 0.5 * ts + 0.5 * invs))


def _status_to_compare(status: Dict[str, object]) -> CompareDataSourceStatus:
    return CompareDataSourceStatus(
        degraded=bool(status.get("degraded", False)),
        source=str(status.get("source", "unknown")),
        reason=str(status.get("reason", "") or ""),
    )


def _build_summary(results: List[DualAreaScore]) -> str:
    if not results:
        return "No areas could be scored by either engine for this request."
    total = len(results)
    balanced = sum(1 for r in results if r.persona == "balanced")
    tenant_lean = sum(1 for r in results if r.persona == "tenant_leaning")
    inv_lean = sum(1 for r in results if r.persona == "investor_leaning")
    top = results[0]
    return (
        f"Scanned {total} areas. {balanced} balanced, "
        f"{tenant_lean} tenant-leaning, {inv_lean} investor-leaning. "
        f"Top combined pick: {top.area} (tenant "
        f"{(top.tenant_score or 0) * 100:.0f}% / investor "
        f"{(top.investment_score or 0) * 100:.0f}%)."
    )


def compare_areas(request: CompareRequest) -> CompareResponse:
    """Run both engines and fuse results into a cross-stakeholder view."""
    tenant_req = _build_tenant_request(request)
    investment_req = _build_investment_request(request)

    tenant_results: List[TenantOutput] = recommend_areas(tenant_req) or []
    investment_results: List[InvestmentOutput] = (
        analyze_investments(investment_req) or []
    )

    tenant_by_area: Dict[str, TenantOutput] = {t.area: t for t in tenant_results}
    invest_by_area: Dict[str, InvestmentOutput] = {
        i.area: i for i in investment_results
    }
    all_areas = sorted(set(tenant_by_area.keys()) | set(invest_by_area.keys()))

    dual: List[DualAreaScore] = []
    for area in all_areas:
        t = tenant_by_area.get(area)
        i = invest_by_area.get(area)

        tenant_score = t.final_score if t is not None else None
        investment_score = i.investment_score if i is not None else None

        persona, delta = _classify_persona(tenant_score, investment_score)
        combined = _combined_score(tenant_score, investment_score)

        roi = i.roi_estimate if i is not None else None
        yield_pct = i.gross_rental_yield_pct if i is not None else None
        risk_score = i.risk_score if i is not None else None
        avg_price = i.avg_unit_price_aed if i is not None else None
        classification = _classification_for(persona)
        insight = _build_insight(
            area,
            tenant_score,
            investment_score,
            persona,
            roi,
            yield_pct,
            risk_score,
        )
        classification_reason = _build_classification_reason(
            classification,
            tenant_score,
            investment_score,
            roi,
            yield_pct,
            risk_score,
            avg_price,
        )

        dual.append(
            DualAreaScore(
                area=area,
                tenant_score=tenant_score,
                tenant_viable=bool(t.viable) if t is not None else False,
                tenant_headline=(t.explanation.headline if t is not None else None),
                tenant_low_confidence=bool(t.low_confidence) if t is not None else False,
                investment_score=investment_score,
                investment_viable=bool(i.viable) if i is not None else False,
                investment_headline=(
                    i.explanation.headline if i is not None else None
                ),
                investment_low_confidence=(
                    bool(i.low_confidence) if i is not None else False
                ),
                roi_estimate=roi,
                risk_score=(i.risk_score if i is not None else None),
                avg_unit_price_aed=(
                    i.avg_unit_price_aed if i is not None else None
                ),
                gross_rental_yield_pct=(
                    i.gross_rental_yield_pct if i is not None else None
                ),
                combined_score=combined,
                tradeoff_delta=max(-1.0, min(1.0, float(delta))),
                persona=persona,
                classification=classification,
                classification_reason=classification_reason,
                insight=insight,
            )
        )

    dual.sort(
        key=lambda d: (
            -d.combined_score,
            -(d.tenant_score or 0.0),
            -(d.investment_score or 0.0),
            d.area,
        )
    )

    return CompareResponse(
        results=dual,
        tenant_data_source=_status_to_compare(tenant_status()),
        investment_data_source=_status_to_compare(investment_status()),
        summary=_build_summary(dual),
    )
