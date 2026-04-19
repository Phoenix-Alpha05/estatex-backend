"""Investment Recommendation Service Layer (v1-DXB.1).

Orchestration-only. This module does NOT:
  - expose HTTP endpoints (no FastAPI here),
  - re-implement any scoring logic (core functions are called as-is),
  - mutate global state.

It glues together the pure functions in ``app.core.investment_scoring``
with the Pydantic models in ``app.models.investment_input`` /
``investment_output``, producing a sorted, explainable list of investment
recommendations.

The dataset is loaded from the Supabase ``dubai_investments`` table via
``_load_investment_dataset``. An in-code mock dataset is retained strictly
as a resilience fallback if the DB is unreachable or returns no rows, so
the scoring pipeline never sees an empty input.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as _urlreq
from urllib.error import HTTPError, URLError

logger = logging.getLogger("app.services.investment_service")

from app.core.investment_scoring import (
    AppreciationResult,
    BudgetGateResult,
    ConfidenceResult,
    DemandResult,
    FinalInvestmentResult,
    LiquidityResult,
    PillarWeightsResolved,
    RiskResult,
    ROIResult,
    YieldResult,
    compute_appreciation_score,
    compute_budget_gate,
    compute_confidence,
    compute_demand_score,
    compute_final_investment_score,
    compute_liquidity_score,
    compute_pillar_weights,
    compute_rental_yield_score,
    compute_risk_score,
    compute_roi_estimate,
    RISK_TOLERANCE_CEIL,
)
from app.models.investment_input import InvestmentInput
from app.models.investment_output import (
    InvestmentDriver,
    InvestmentExplanation,
    InvestmentOutput,
    InvestmentPillarScores,
)


# ---------------------------------------------------------------------------
# Investment dataset record.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvestmentAreaRecord:
    """Raw investment feature bundle for a single Dubai area."""

    area_name: str
    avg_price_per_sqft_aed: float
    avg_unit_price_aed: float
    gross_rental_yield_pct: float
    price_growth_3y_pct: float
    price_growth_1y_pct: float
    transaction_volume_score: float
    occupancy_rate: float
    days_on_market: float
    supply_pipeline_score: float
    off_plan_ratio: float
    price_volatility: float
    listings_count: int
    last_updated_days: int


_MOCK_INVESTMENTS: Tuple[InvestmentAreaRecord, ...] = (
    InvestmentAreaRecord("Downtown Dubai",         2100, 2_650_000, 5.8, 28.0, 11.0, 0.92, 0.92, 38, 0.35, 0.28, 0.22, 520, 14),
    InvestmentAreaRecord("Dubai Marina",           1650, 1_850_000, 6.8, 22.0,  9.0, 0.90, 0.90, 42, 0.30, 0.22, 0.20, 610, 21),
    InvestmentAreaRecord("Jumeirah Village Circle",1050,   950_000, 8.2, 18.0, 12.0, 0.78, 0.86, 55, 0.62, 0.45, 0.28, 820, 45),
    InvestmentAreaRecord("Business Bay",           1550, 1_400_000, 6.5, 24.0, 10.0, 0.88, 0.88, 45, 0.48, 0.35, 0.22, 680, 18),
    InvestmentAreaRecord("Dubai Silicon Oasis",     900,   780_000, 7.4, 14.0,  7.0, 0.62, 0.84, 62, 0.30, 0.18, 0.18, 360, 60),
    InvestmentAreaRecord("Mirdif",                 1150, 1_250_000, 6.2, 10.0,  5.0, 0.55, 0.88, 70, 0.18, 0.10, 0.14, 310, 75),
    InvestmentAreaRecord("Deira",                   950,   780_000, 7.8,  8.0,  4.0, 0.58, 0.82, 65, 0.12, 0.05, 0.16, 680, 35),
    InvestmentAreaRecord("Al Barsha",              1250, 1_150_000, 6.4, 12.0,  6.0, 0.66, 0.86, 58, 0.22, 0.14, 0.16, 470, 28),
    InvestmentAreaRecord("Palm Jumeirah",          3100, 5_400_000, 5.2, 34.0, 14.0, 0.70, 0.94, 48, 0.28, 0.20, 0.26, 220, 22),
    InvestmentAreaRecord("Dubai Sports City",       850,   720_000, 7.9, 12.0,  8.0, 0.48, 0.80, 78, 0.55, 0.40, 0.24, 180,110),
)


# ---------------------------------------------------------------------------
# Supabase data source.
# ---------------------------------------------------------------------------


_DUBAI_INVESTMENTS_TABLE = "dubai_investments"


def _load_env_file_into_os_environ() -> None:
    here = Path(__file__).resolve()
    for candidate in (
        here.parents[3] / ".env",
        here.parents[2] / ".env",
        Path.cwd() / ".env",
    ):
        try:
            if not candidate.is_file():
                continue
            for raw in candidate.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            return
        except OSError:
            continue


def _resolve_supabase_credentials() -> Tuple[Optional[str], Optional[str]]:
    _load_env_file_into_os_environ()
    url = os.environ.get("SUPABASE_URL") or os.environ.get("VITE_SUPABASE_URL")
    anon_key = (
        os.environ.get("SUPABASE_ANON_KEY")
        or os.environ.get("VITE_SUPABASE_ANON_KEY")
        or os.environ.get("VITE_SUPABASE_SUPABASE_ANON_KEY")
    )
    if url:
        url = url.rstrip("/")
    return url, anon_key


DATA_SOURCE_STATUS: Dict[str, Any] = {
    "source": "unknown",
    "degraded": False,
    "error": None,
    "row_count": 0,
}


def _safe_float(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_int(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return fallback


def _clamp_unit(x: float) -> float:
    return max(0.0, min(1.0, x))


def _row_to_investment_record(row: Dict[str, Any]) -> Optional[InvestmentAreaRecord]:
    area_name = row.get("area_name")
    if not area_name or not isinstance(area_name, str):
        return None
    return InvestmentAreaRecord(
        area_name=area_name,
        avg_price_per_sqft_aed=_safe_float(row.get("avg_price_per_sqft_aed"), 0.0),
        avg_unit_price_aed=_safe_float(row.get("avg_unit_price_aed"), 0.0),
        gross_rental_yield_pct=_safe_float(row.get("gross_rental_yield_pct"), 0.0),
        price_growth_3y_pct=_safe_float(row.get("price_growth_3y_pct"), 0.0),
        price_growth_1y_pct=_safe_float(row.get("price_growth_1y_pct"), 0.0),
        transaction_volume_score=_clamp_unit(_safe_float(row.get("transaction_volume_score"), 0.5)),
        occupancy_rate=_clamp_unit(_safe_float(row.get("occupancy_rate"), 0.85)),
        days_on_market=max(0.0, _safe_float(row.get("days_on_market"), 60.0)),
        supply_pipeline_score=_clamp_unit(_safe_float(row.get("supply_pipeline_score"), 0.3)),
        off_plan_ratio=_clamp_unit(_safe_float(row.get("off_plan_ratio"), 0.2)),
        price_volatility=_clamp_unit(_safe_float(row.get("price_volatility"), 0.2)),
        listings_count=max(0, _safe_int(row.get("listings_count"), 0)),
        last_updated_days=max(0, _safe_int(row.get("last_updated_days"), 0)),
    )


def _fetch_investment_rows(
    timeout_seconds: Optional[float] = None,
    max_rows: Optional[int] = None,
    max_attempts: int = 2,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if timeout_seconds is None:
        timeout_seconds = _safe_float(
            os.environ.get("DUBAI_INVESTMENTS_TIMEOUT_SECONDS"), 8.0
        )
    if max_rows is None:
        max_rows = _safe_int(os.environ.get("DUBAI_INVESTMENTS_ROW_LIMIT"), 200)
    if max_rows <= 0:
        max_rows = 200
    if max_attempts <= 0:
        max_attempts = 1

    url, anon_key = _resolve_supabase_credentials()
    if not url or not anon_key:
        err = "credentials_missing"
        logger.error("supabase:%s table=%s", err, _DUBAI_INVESTMENTS_TABLE)
        return [], err

    endpoint = (
        f"{url}/rest/v1/{_DUBAI_INVESTMENTS_TABLE}"
        "?select=area_name,avg_price_per_sqft_aed,avg_unit_price_aed,"
        "gross_rental_yield_pct,price_growth_3y_pct,price_growth_1y_pct,"
        "transaction_volume_score,occupancy_rate,days_on_market,"
        "supply_pipeline_score,off_plan_ratio,price_volatility,"
        "listings_count,last_updated_days"
        "&order=area_name.asc"
        f"&limit={int(max_rows)}"
    )
    headers = {
        "apikey": anon_key,
        "Authorization": f"Bearer {anon_key}",
        "Accept": "application/json",
    }

    last_error: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        try:
            req = _urlreq.Request(endpoint, headers=headers, method="GET")
            with _urlreq.urlopen(req, timeout=timeout_seconds) as resp:
                payload = resp.read()
            data = json.loads(payload.decode("utf-8"))
            if not isinstance(data, list):
                last_error = "unexpected_payload_shape"
                continue
            rows = [r for r in data if isinstance(r, dict)]
            return rows, None
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            last_error = f"{type(exc).__name__}:{exc}"
            logger.warning(
                "supabase:fetch_attempt_failed table=%s attempt=%d/%d err=%s",
                _DUBAI_INVESTMENTS_TABLE, attempt, max_attempts, last_error,
            )

    logger.error(
        "supabase:fetch_failed_all_attempts table=%s attempts=%d err=%s",
        _DUBAI_INVESTMENTS_TABLE, max_attempts, last_error,
    )
    return [], last_error or "unknown_error"


def _load_investment_dataset() -> Tuple[InvestmentAreaRecord, ...]:
    rows, fetch_error = _fetch_investment_rows()
    records: List[InvestmentAreaRecord] = []
    for row in rows:
        rec = _row_to_investment_record(row)
        if rec is None:
            continue
        records.append(rec)

    if records:
        DATA_SOURCE_STATUS.update(
            source="supabase",
            degraded=False,
            error=None,
            row_count=len(records),
        )
        logger.info(
            "dubai_investments:loaded source=supabase count=%d",
            len(records),
        )
        return tuple(records)

    reason = fetch_error or "empty_result"
    logger.error(
        "dubai_investments:degraded reason=%s using_mock_fallback count=%d",
        reason, len(_MOCK_INVESTMENTS),
    )
    DATA_SOURCE_STATUS.update(
        source="mock_fallback",
        degraded=True,
        error=reason,
        row_count=len(_MOCK_INVESTMENTS),
    )
    return _MOCK_INVESTMENTS


# ---------------------------------------------------------------------------
# Drivers.
# ---------------------------------------------------------------------------


_PILLAR_LABEL: Dict[str, str] = {
    "rental_yield": "rental_yield",
    "appreciation": "appreciation",
    "demand":       "demand",
    "liquidity":    "liquidity",
}


def _build_drivers(
    pillars: Dict[str, float],
    pillar_weights: Dict[str, float],
) -> Tuple[List[InvestmentDriver], Optional[InvestmentDriver]]:
    if not pillars:
        return [], None
    mean = sum(pillars.values()) / len(pillars)
    contributions: List[Tuple[str, float]] = []
    for name, score in pillars.items():
        weight = pillar_weights.get(name, 0.0)
        contributions.append((name, (score - mean) * weight))
    contributions.sort(key=lambda kv: kv[1], reverse=True)

    positives: List[InvestmentDriver] = []
    for name, contrib in contributions:
        if contrib <= 0.0:
            continue
        positives.append(
            InvestmentDriver(
                feature=name,
                pillar=_PILLAR_LABEL[name],
                contribution=contrib,
                note=f"{name} pillar lifts this area above its own average.",
            )
        )
        if len(positives) >= 3:
            break

    worst_name, worst_contrib = contributions[-1]
    negative: Optional[InvestmentDriver] = None
    if worst_contrib < 0.0:
        negative = InvestmentDriver(
            feature=worst_name,
            pillar=_PILLAR_LABEL[worst_name],
            contribution=worst_contrib,
            note=f"{worst_name} is the weakest pillar for this area.",
        )
    return positives, negative


# ---------------------------------------------------------------------------
# Warnings + explanation.
# ---------------------------------------------------------------------------


def _build_warnings(
    area: InvestmentAreaRecord,
    budget_gate: BudgetGateResult,
    risk: RiskResult,
    risk_level: str,
    confidence: ConfidenceResult,
    freshness_threshold_days: int,
) -> List[str]:
    warnings: List[str] = []
    if budget_gate.excluded:
        warnings.append("excluded:unit_price_above_budget_headroom")
    elif budget_gate.price_to_budget > 1.0:
        warnings.append("budget:unit_price_above_budget_within_headroom")

    ceil = RISK_TOLERANCE_CEIL.get(risk_level, 0.65)
    if risk.score > ceil:
        warnings.append(f"risk:above_tolerance_for_{risk_level}_investor")

    if confidence.low_confidence:
        warnings.append("confidence:low_confidence_result")

    if area.last_updated_days > freshness_threshold_days:
        warnings.append(
            f"freshness:data_older_than_threshold_{freshness_threshold_days}d"
        )
    if area.supply_pipeline_score > 0.55:
        warnings.append("supply:high_incoming_pipeline")
    if area.off_plan_ratio > 0.4:
        warnings.append("risk:high_off_plan_share")
    return warnings


def _build_explanation(
    area_name: str,
    final: FinalInvestmentResult,
    viable: bool,
    roi: ROIResult,
    risk: RiskResult,
    top_positive: List[InvestmentDriver],
    top_negative: Optional[InvestmentDriver],
) -> InvestmentExplanation:
    if not viable:
        headline = (
            f"{area_name} is not viable at this budget or has insufficient data."
        )
    elif top_positive:
        lead = top_positive[0]
        headline = (
            f"{area_name} scores {final.investment_score:.2f}, lifted by "
            f"{lead.feature} (+{lead.contribution:.2f}); projected ROI "
            f"{roi.roi_estimate * 100:.1f}%/yr."
        )
    else:
        top_pillar = max(final.pillars.items(), key=lambda kv: kv[1])[0]
        headline = (
            f"{area_name} scores {final.investment_score:.2f} overall, led "
            f"by {top_pillar}; projected ROI {roi.roi_estimate * 100:.1f}%/yr."
        )

    positive_frag = (
        ", ".join(f"{d.feature} (+{d.contribution:.2f})" for d in top_positive)
        if top_positive else "no pillar stands out above the area mean"
    )
    negative_frag = (
        f"{top_negative.feature} ({top_negative.contribution:+.2f})"
        if top_negative is not None else "no material drag"
    )
    details = (
        f"Strengths: {positive_frag}. Weakness: {negative_frag}. "
        f"Raw blended={final.raw_blended:.3f}, penalty={final.penalty_applied:.3f}, "
        f"budget_factor={final.budget_factor:.3f}, risk={risk.score:.3f} "
        f"(vol={risk.volatility_component:.2f}, supply={risk.supply_component:.2f}, "
        f"off_plan={risk.offplan_component:.2f}). "
        f"ROI split: net_yield={roi.net_yield_component * 100:.2f}%, "
        f"appreciation={roi.appreciation_component * 100:.2f}%."
    )
    return InvestmentExplanation(headline=headline, details=details)


# ---------------------------------------------------------------------------
# Per-area processing.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ScoredInvestment:
    output: InvestmentOutput
    investment_score: float
    excluded: bool


def _score_one_area(area: InvestmentAreaRecord, req: InvestmentInput) -> _ScoredInvestment:
    # Step 1: rental yield
    yield_result: YieldResult = compute_rental_yield_score(area.gross_rental_yield_pct)

    # Step 2: appreciation
    apprec: AppreciationResult = compute_appreciation_score(
        price_growth_3y_pct=area.price_growth_3y_pct,
        price_growth_1y_pct=area.price_growth_1y_pct,
    )

    # Step 3: demand
    demand: DemandResult = compute_demand_score(
        transaction_volume_score=area.transaction_volume_score,
        occupancy_rate=area.occupancy_rate,
    )

    # Step 4: liquidity
    liquidity: LiquidityResult = compute_liquidity_score(
        days_on_market=area.days_on_market,
        listings_count=area.listings_count,
    )

    # Step 5: risk
    risk: RiskResult = compute_risk_score(
        price_volatility=area.price_volatility,
        supply_pipeline_score=area.supply_pipeline_score,
        off_plan_ratio=area.off_plan_ratio,
        data_age_days=float(area.last_updated_days),
    )

    # Step 6: pillar weights
    pillar_weights: PillarWeightsResolved = compute_pillar_weights(
        risk_level=req.risk_level,
        investment_horizon=req.investment_horizon,
        rental_yield_override=req.pillar_weights.rental_yield,
        appreciation_override=req.pillar_weights.appreciation,
        demand_override=req.pillar_weights.demand,
        liquidity_override=req.pillar_weights.liquidity,
    )

    # Step 7: budget gate
    budget_gate: BudgetGateResult = compute_budget_gate(
        avg_unit_price_aed=area.avg_unit_price_aed,
        budget_aed=req.budget_aed,
        headroom=req.budget_headroom,
    )

    # Step 8: confidence
    confidence: ConfidenceResult = compute_confidence(
        data_age_days=float(area.last_updated_days),
        listings_count=area.listings_count,
    )

    # Step 9: final score
    final: FinalInvestmentResult = compute_final_investment_score(
        rental_yield_score=yield_result.score,
        appreciation_score=apprec.score,
        demand_score=demand.score,
        liquidity_score=liquidity.score,
        pillar_weights=pillar_weights,
        data_age_days=float(area.last_updated_days),
        budget_gate=budget_gate,
        low_confidence=confidence.low_confidence,
    )

    # Step 10: ROI
    roi: ROIResult = compute_roi_estimate(
        gross_rental_yield_pct=area.gross_rental_yield_pct,
        blended_annualised_appreciation_pct=apprec.blended_annualised_pct,
        horizon=req.investment_horizon,
        risk_score=risk.score,
    )

    pillars_model = InvestmentPillarScores(
        rental_yield_score=final.pillars["rental_yield"],
        appreciation_score=final.pillars["appreciation"],
        demand_score=final.pillars["demand"],
        liquidity_score=final.pillars["liquidity"],
    )
    top_positive, top_negative = _build_drivers(
        pillars=final.pillars,
        pillar_weights=final.pillar_weights,
    )
    viable = not budget_gate.excluded
    explanation = _build_explanation(
        area.area_name, final, viable, roi, risk, top_positive, top_negative,
    )
    warnings = _build_warnings(
        area=area,
        budget_gate=budget_gate,
        risk=risk,
        risk_level=req.risk_level,
        confidence=confidence,
        freshness_threshold_days=req.freshness_threshold_days,
    )

    output = InvestmentOutput(
        area=area.area_name,
        roi_estimate=roi.roi_estimate,
        risk_score=risk.score,
        investment_score=final.investment_score,
        confidence_score=confidence.score,
        low_confidence=confidence.low_confidence,
        viable=viable,
        avg_unit_price_aed=area.avg_unit_price_aed,
        gross_rental_yield_pct=area.gross_rental_yield_pct,
        penalty_applied=final.penalty_applied,
        pillars=pillars_model,
        top_positive_drivers=top_positive,
        top_negative_factor=top_negative,
        explanation=explanation,
        warnings=warnings,
    )
    return _ScoredInvestment(
        output=output,
        investment_score=final.investment_score,
        excluded=budget_gate.excluded,
    )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def analyze_investments(
    request: InvestmentInput,
    include_excluded: bool = False,
) -> List[InvestmentOutput]:
    """Run the full investment recommendation pipeline.

    Order of operations:
      1. Load dataset.
      2. For each area, run the 10-step scoring pipeline.
      3. Sort viable areas first, then by investment_score DESC,
         then by area name ASC for deterministic tie-breaking.
      4. Filter out or retain excluded areas depending on ``include_excluded``.
    """
    dataset = _load_investment_dataset()
    scored: List[_ScoredInvestment] = [
        _score_one_area(area, request) for area in dataset
    ]
    scored.sort(
        key=lambda s: (
            0 if not s.excluded else 1,
            -s.investment_score,
            s.output.area,
        )
    )
    if include_excluded:
        return [s.output for s in scored]
    return [s.output for s in scored if not s.excluded]


def get_data_source_status() -> Dict[str, Any]:
    raw_source = DATA_SOURCE_STATUS.get("source", "unknown")
    if raw_source == "supabase":
        source = "db"
    elif raw_source == "mock_fallback":
        source = "mock_fallback"
    else:
        source = "unknown"
    error = DATA_SOURCE_STATUS.get("error")
    return {
        "degraded": bool(DATA_SOURCE_STATUS.get("degraded", False)),
        "source": source,
        "reason": "" if error is None else str(error),
    }
