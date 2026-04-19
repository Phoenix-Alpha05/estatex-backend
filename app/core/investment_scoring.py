"""Core scoring engine for the Dubai Investment Recommendation Engine (v1-DXB.1).

This module is intentionally IO-free and framework-free. Every function is
pure and deterministic: same inputs always produce the same outputs, no
randomness, no database access, no HTTP calls, no filesystem access, no
global mutable state.

Design:
  - 4 pillars: rental_yield, appreciation, demand, liquidity.
  - Structured, explainable, deterministic scoring (NO machine learning).
  - Pillar weights resolved from ``risk_level`` and ``investment_horizon``
    unless the caller overrides them. Provided overrides are honored.
  - Final score = raw_blended * (1 - penalty) * budget_factor.
  - Separate risk_score aggregates volatility, supply, off-plan ratio,
    and data freshness (NOT subtracted from the investment score; it is
    surfaced as an independent dimension).
  - ROI estimate blends gross yield (net of an ops haircut) and a
    horizon-adjusted annualised appreciation expectation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional


# ---------------------------------------------------------------------------
# Constants (v1-DXB.1).
# ---------------------------------------------------------------------------

# Pillar weight presets keyed by (risk_level, horizon). All rows sum to 1.0.
PILLAR_WEIGHT_PRESETS: Dict[str, Dict[str, Dict[str, float]]] = {
    "low": {
        "short":  {"rental_yield": 0.50, "appreciation": 0.10, "demand": 0.20, "liquidity": 0.20},
        "medium": {"rental_yield": 0.45, "appreciation": 0.20, "demand": 0.20, "liquidity": 0.15},
        "long":   {"rental_yield": 0.40, "appreciation": 0.30, "demand": 0.20, "liquidity": 0.10},
    },
    "medium": {
        "short":  {"rental_yield": 0.40, "appreciation": 0.20, "demand": 0.25, "liquidity": 0.15},
        "medium": {"rental_yield": 0.30, "appreciation": 0.30, "demand": 0.25, "liquidity": 0.15},
        "long":   {"rental_yield": 0.25, "appreciation": 0.40, "demand": 0.25, "liquidity": 0.10},
    },
    "high": {
        "short":  {"rental_yield": 0.25, "appreciation": 0.35, "demand": 0.25, "liquidity": 0.15},
        "medium": {"rental_yield": 0.20, "appreciation": 0.45, "demand": 0.25, "liquidity": 0.10},
        "long":   {"rental_yield": 0.15, "appreciation": 0.55, "demand": 0.20, "liquidity": 0.10},
    },
}

# Yield normalization band: linear 0..1 across [YIELD_MIN .. YIELD_MAX]%.
YIELD_FLOOR_PCT: float = 3.0
YIELD_CEIL_PCT: float = 10.0

# Appreciation normalization uses trailing 3Y and 1Y, annualized. A ~10%/yr
# trailing appreciation maps near the top of the band.
APPREC_FLOOR_ANNUAL_PCT: float = -2.0
APPREC_CEIL_ANNUAL_PCT: float = 12.0
APPREC_3Y_WEIGHT: float = 0.65
APPREC_1Y_WEIGHT: float = 0.35

# Demand pillar weights.
DEMAND_TXN_WEIGHT: float = 0.55
DEMAND_OCC_WEIGHT: float = 0.45

# Liquidity: shorter days-on-market + deeper listings => more liquid.
DOM_FLOOR_DAYS: float = 20.0  # capped at max liquidity
DOM_CEIL_DAYS: float = 180.0  # capped at min liquidity
LISTINGS_DEEP: float = 500.0   # counts above this saturate to 1.0
LIQ_DOM_WEIGHT: float = 0.65
LIQ_LISTINGS_WEIGHT: float = 0.35

# Risk composition (weights sum to 1.0).
RISK_VOLATILITY_W: float = 0.35
RISK_SUPPLY_W: float = 0.25
RISK_OFFPLAN_W: float = 0.20
RISK_FRESHNESS_W: float = 0.20

# Horizon multiplier on appreciation contribution to ROI.
HORIZON_APPREC_MULT: Dict[str, float] = {
    "short":  0.25,
    "medium": 0.60,
    "long":   1.00,
}

# Operating-cost + ownership-drag haircut on gross yield to reach a
# deterministic "net" yield for ROI (service charges, management,
# vacancy). Conservative, fixed.
GROSS_YIELD_OPS_HAIRCUT: float = 0.25  # 25% haircut

# ROI cap (sanity clamp).
ROI_MIN: float = -0.50
ROI_MAX: float = 0.50

# Risk tolerance -> maximum acceptable risk_score before a viability
# warning is attached (soft; the area remains viable unless the budget
# gate fails).
RISK_TOLERANCE_CEIL: Dict[str, float] = {
    "low":    0.40,
    "medium": 0.65,
    "high":   0.90,
}

# Freshness tiers (days -> penalty on final score).
FRESHNESS_TIERS = (
    (30,  0.00),
    (90,  0.05),
    (180, 0.10),
    (365, 0.15),
)
FRESHNESS_MAX_PENALTY: float = 0.20

# Budget gate. If avg_unit_price_aed > budget * headroom => hard-exclude.
# Otherwise a soft factor rewards lower price-to-budget ratios.
BUDGET_SOFT_FACTOR_FLOOR: float = 0.80

# Confidence.
LOW_CONFIDENCE_THRESHOLD: float = 0.55
LOW_CONFIDENCE_PENALTY: float = 0.05

_EPS: float = 1e-9


# ---------------------------------------------------------------------------
# Result dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class YieldResult:
    score: float
    gross_yield_pct: float


@dataclass(frozen=True)
class AppreciationResult:
    score: float
    annualised_3y_pct: float
    pct_1y: float
    blended_annualised_pct: float


@dataclass(frozen=True)
class DemandResult:
    score: float
    txn_component: float
    occupancy_component: float


@dataclass(frozen=True)
class LiquidityResult:
    score: float
    dom_component: float
    listings_component: float


@dataclass(frozen=True)
class RiskResult:
    score: float
    volatility_component: float
    supply_component: float
    offplan_component: float
    freshness_component: float


@dataclass(frozen=True)
class PillarWeightsResolved:
    rental_yield: float
    appreciation: float
    demand: float
    liquidity: float
    source: str  # "preset" | "override" | "mixed"


@dataclass(frozen=True)
class BudgetGateResult:
    excluded: bool
    factor: float
    price_to_budget: float


@dataclass(frozen=True)
class ROIResult:
    roi_estimate: float
    net_yield_component: float
    appreciation_component: float


@dataclass(frozen=True)
class ConfidenceResult:
    score: float
    low_confidence: bool
    freshness_unit: float
    data_depth_unit: float


@dataclass(frozen=True)
class FinalInvestmentResult:
    investment_score: float
    raw_blended: float
    penalty_applied: float
    budget_factor: float
    pillars: Dict[str, float]
    pillar_weights: Dict[str, float]
    freshness_penalty: float
    low_confidence_penalty: float


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _normalize(weights: Mapping[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in weights.values())
    if total <= _EPS:
        n = len(weights)
        if n == 0:
            return {}
        return {k: 1.0 / n for k in weights}
    return {k: max(0.0, v) / total for k, v in weights.items()}


def _linear_band(x: float, lo: float, hi: float) -> float:
    """Map x linearly into [0, 1] across [lo, hi]; clamp outside."""
    if hi <= lo:
        return 0.0
    return _clamp((x - lo) / (hi - lo))


def _inverse_linear_band(x: float, lo: float, hi: float) -> float:
    """Map x linearly into [1, 0] across [lo, hi]; clamp outside.

    Used for "smaller is better" features (e.g. days-on-market).
    """
    if hi <= lo:
        return 0.0
    return _clamp(1.0 - (x - lo) / (hi - lo))


# ---------------------------------------------------------------------------
# 1. Rental yield pillar.
# ---------------------------------------------------------------------------


def compute_rental_yield_score(gross_rental_yield_pct: float) -> YieldResult:
    if gross_rental_yield_pct < 0.0:
        raise ValueError("gross_rental_yield_pct must be non-negative.")
    score = _linear_band(gross_rental_yield_pct, YIELD_FLOOR_PCT, YIELD_CEIL_PCT)
    return YieldResult(score=score, gross_yield_pct=gross_rental_yield_pct)


# ---------------------------------------------------------------------------
# 2. Appreciation pillar.
# ---------------------------------------------------------------------------


def compute_appreciation_score(
    price_growth_3y_pct: float,
    price_growth_1y_pct: float,
) -> AppreciationResult:
    # Annualise 3y growth (geometric). Clamp to sane band to avoid
    # degenerate inputs.
    g3 = max(-99.0, price_growth_3y_pct) / 100.0
    annualised_3y = ((1.0 + g3) ** (1.0 / 3.0) - 1.0) * 100.0
    blended = APPREC_3Y_WEIGHT * annualised_3y + APPREC_1Y_WEIGHT * price_growth_1y_pct
    score = _linear_band(blended, APPREC_FLOOR_ANNUAL_PCT, APPREC_CEIL_ANNUAL_PCT)
    return AppreciationResult(
        score=score,
        annualised_3y_pct=annualised_3y,
        pct_1y=price_growth_1y_pct,
        blended_annualised_pct=blended,
    )


# ---------------------------------------------------------------------------
# 3. Demand pillar.
# ---------------------------------------------------------------------------


def compute_demand_score(
    transaction_volume_score: float,
    occupancy_rate: float,
) -> DemandResult:
    for name, v in (
        ("transaction_volume_score", transaction_volume_score),
        ("occupancy_rate", occupancy_rate),
    ):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1] (got {v}).")
    txn = transaction_volume_score
    occ = occupancy_rate
    score = _clamp(DEMAND_TXN_WEIGHT * txn + DEMAND_OCC_WEIGHT * occ)
    return DemandResult(score=score, txn_component=txn, occupancy_component=occ)


# ---------------------------------------------------------------------------
# 4. Liquidity pillar.
# ---------------------------------------------------------------------------


def compute_liquidity_score(
    days_on_market: float,
    listings_count: int,
) -> LiquidityResult:
    if days_on_market < 0.0:
        raise ValueError("days_on_market must be non-negative.")
    if listings_count < 0:
        raise ValueError("listings_count must be non-negative.")
    dom_score = _inverse_linear_band(days_on_market, DOM_FLOOR_DAYS, DOM_CEIL_DAYS)
    listings_score = _linear_band(float(listings_count), 0.0, LISTINGS_DEEP)
    score = _clamp(LIQ_DOM_WEIGHT * dom_score + LIQ_LISTINGS_WEIGHT * listings_score)
    return LiquidityResult(
        score=score,
        dom_component=dom_score,
        listings_component=listings_score,
    )


# ---------------------------------------------------------------------------
# 5. Risk composition.
# ---------------------------------------------------------------------------


def _freshness_unit(data_age_days: float) -> float:
    """Map data age to a [0, 1] unit where 1 = fresh, 0 = very stale."""
    if data_age_days < 0.0:
        raise ValueError("data_age_days must be non-negative.")
    penalty = FRESHNESS_MAX_PENALTY
    for max_days, p in FRESHNESS_TIERS:
        if data_age_days <= max_days:
            penalty = p
            break
    if FRESHNESS_MAX_PENALTY <= 0.0:
        return 1.0
    return _clamp(1.0 - (penalty / FRESHNESS_MAX_PENALTY))


def compute_risk_score(
    price_volatility: float,
    supply_pipeline_score: float,
    off_plan_ratio: float,
    data_age_days: float,
) -> RiskResult:
    for name, v in (
        ("price_volatility", price_volatility),
        ("supply_pipeline_score", supply_pipeline_score),
        ("off_plan_ratio", off_plan_ratio),
    ):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1] (got {v}).")
    fresh_unit = _freshness_unit(data_age_days)
    fresh_risk = 1.0 - fresh_unit
    score = _clamp(
        RISK_VOLATILITY_W * price_volatility
        + RISK_SUPPLY_W * supply_pipeline_score
        + RISK_OFFPLAN_W * off_plan_ratio
        + RISK_FRESHNESS_W * fresh_risk
    )
    return RiskResult(
        score=score,
        volatility_component=price_volatility,
        supply_component=supply_pipeline_score,
        offplan_component=off_plan_ratio,
        freshness_component=fresh_risk,
    )


# ---------------------------------------------------------------------------
# 6. Pillar weight resolution.
# ---------------------------------------------------------------------------


def compute_pillar_weights(
    risk_level: str,
    investment_horizon: str,
    rental_yield_override: Optional[float] = None,
    appreciation_override: Optional[float] = None,
    demand_override: Optional[float] = None,
    liquidity_override: Optional[float] = None,
) -> PillarWeightsResolved:
    if risk_level not in PILLAR_WEIGHT_PRESETS:
        raise ValueError(f"risk_level must be in {list(PILLAR_WEIGHT_PRESETS)}.")
    if investment_horizon not in PILLAR_WEIGHT_PRESETS[risk_level]:
        raise ValueError(
            f"investment_horizon must be in "
            f"{list(PILLAR_WEIGHT_PRESETS[risk_level])}."
        )
    preset = PILLAR_WEIGHT_PRESETS[risk_level][investment_horizon]
    overrides = {
        "rental_yield": rental_yield_override,
        "appreciation": appreciation_override,
        "demand": demand_override,
        "liquidity": liquidity_override,
    }
    n_provided = sum(1 for v in overrides.values() if v is not None)
    if n_provided == 4:
        source = "override"
    elif n_provided == 0:
        source = "preset"
    else:
        source = "mixed"
    resolved = {
        k: (overrides[k] if overrides[k] is not None else preset[k])
        for k in preset
    }
    norm = _normalize(resolved)
    return PillarWeightsResolved(
        rental_yield=norm["rental_yield"],
        appreciation=norm["appreciation"],
        demand=norm["demand"],
        liquidity=norm["liquidity"],
        source=source,
    )


# ---------------------------------------------------------------------------
# 7. Budget gate.
# ---------------------------------------------------------------------------


def compute_budget_gate(
    avg_unit_price_aed: float,
    budget_aed: float,
    headroom: float = 1.15,
) -> BudgetGateResult:
    if budget_aed <= 0.0:
        raise ValueError("budget_aed must be positive.")
    if avg_unit_price_aed < 0.0:
        raise ValueError("avg_unit_price_aed must be non-negative.")
    if headroom < 1.0:
        raise ValueError("headroom must be >= 1.0.")
    if avg_unit_price_aed <= 0.0:
        return BudgetGateResult(excluded=True, factor=0.0, price_to_budget=0.0)
    ratio = avg_unit_price_aed / budget_aed
    if ratio > headroom:
        return BudgetGateResult(excluded=True, factor=0.0, price_to_budget=ratio)
    # Soft factor: ratio <= 1.0 -> factor = 1.0.
    # 1.0 < ratio <= headroom -> linear decay to BUDGET_SOFT_FACTOR_FLOOR.
    if ratio <= 1.0:
        factor = 1.0
    else:
        span = headroom - 1.0
        t = (ratio - 1.0) / span if span > _EPS else 1.0
        factor = 1.0 - t * (1.0 - BUDGET_SOFT_FACTOR_FLOOR)
    return BudgetGateResult(
        excluded=False,
        factor=_clamp(factor, BUDGET_SOFT_FACTOR_FLOOR, 1.0),
        price_to_budget=ratio,
    )


# ---------------------------------------------------------------------------
# 8. Final score.
# ---------------------------------------------------------------------------


def compute_final_investment_score(
    rental_yield_score: float,
    appreciation_score: float,
    demand_score: float,
    liquidity_score: float,
    pillar_weights: PillarWeightsResolved,
    data_age_days: float,
    budget_gate: BudgetGateResult,
    low_confidence: bool = False,
) -> FinalInvestmentResult:
    for name, v in (
        ("rental_yield_score", rental_yield_score),
        ("appreciation_score", appreciation_score),
        ("demand_score", demand_score),
        ("liquidity_score", liquidity_score),
    ):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1] (got {v}).")
    pillars = {
        "rental_yield": rental_yield_score,
        "appreciation": appreciation_score,
        "demand":       demand_score,
        "liquidity":    liquidity_score,
    }
    w = {
        "rental_yield": pillar_weights.rental_yield,
        "appreciation": pillar_weights.appreciation,
        "demand":       pillar_weights.demand,
        "liquidity":    pillar_weights.liquidity,
    }
    raw_blended = sum(w[k] * pillars[k] for k in pillars)

    # Freshness penalty (tiered).
    freshness_pen = FRESHNESS_MAX_PENALTY
    for max_days, p in FRESHNESS_TIERS:
        if data_age_days <= max_days:
            freshness_pen = p
            break

    low_conf_pen = LOW_CONFIDENCE_PENALTY if low_confidence else 0.0
    penalty = _clamp(freshness_pen + low_conf_pen, 0.0, 1.0)

    budget_factor = 0.0 if budget_gate.excluded else budget_gate.factor
    final = _clamp(raw_blended * (1.0 - penalty) * budget_factor)

    return FinalInvestmentResult(
        investment_score=final,
        raw_blended=raw_blended,
        penalty_applied=penalty,
        budget_factor=budget_factor,
        pillars=pillars,
        pillar_weights=w,
        freshness_penalty=freshness_pen,
        low_confidence_penalty=low_conf_pen,
    )


# ---------------------------------------------------------------------------
# 9. ROI estimate.
# ---------------------------------------------------------------------------


def compute_roi_estimate(
    gross_rental_yield_pct: float,
    blended_annualised_appreciation_pct: float,
    horizon: str,
    risk_score: float,
) -> ROIResult:
    if horizon not in HORIZON_APPREC_MULT:
        raise ValueError(f"horizon must be in {list(HORIZON_APPREC_MULT)}.")
    if not (0.0 <= risk_score <= 1.0):
        raise ValueError("risk_score must be in [0, 1].")

    net_yield = (gross_rental_yield_pct / 100.0) * (1.0 - GROSS_YIELD_OPS_HAIRCUT)
    apprec = (blended_annualised_appreciation_pct / 100.0) * HORIZON_APPREC_MULT[horizon]
    # Risk-adjusted haircut on the appreciation leg only (yield is realised
    # income and therefore less sensitive to volatility in this model).
    apprec_adj = apprec * (1.0 - 0.5 * risk_score)
    roi = _clamp(net_yield + apprec_adj, ROI_MIN, ROI_MAX)
    return ROIResult(
        roi_estimate=roi,
        net_yield_component=net_yield,
        appreciation_component=apprec_adj,
    )


# ---------------------------------------------------------------------------
# 10. Confidence.
# ---------------------------------------------------------------------------


def compute_confidence(
    data_age_days: float,
    listings_count: int,
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> ConfidenceResult:
    if listings_count < 0:
        raise ValueError("listings_count must be non-negative.")
    if not (0.0 <= low_confidence_threshold <= 1.0):
        raise ValueError("low_confidence_threshold must be in [0, 1].")

    fresh_unit = _freshness_unit(data_age_days)
    depth_unit = _linear_band(float(listings_count), 0.0, LISTINGS_DEEP)

    score = _clamp(0.6 * fresh_unit + 0.4 * depth_unit)
    low = score < low_confidence_threshold
    return ConfidenceResult(
        score=score,
        low_confidence=low,
        freshness_unit=fresh_unit,
        data_depth_unit=depth_unit,
    )
