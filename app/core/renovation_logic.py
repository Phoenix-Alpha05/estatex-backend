"""Renovation / Value-Add Decision Logic (deterministic, rules-only).

This module is a *pure* decision layer stacked on top of the existing
tenant and investment engines. It does NOT:
  - re-score areas,
  - implement any ML or statistical prediction,
  - call external APIs,
  - mutate global state,
  - duplicate any scoring performed elsewhere.

Given already-scored signals for an area:
  - ``livability_score`` (reused from the tenant engine's final_score),
  - ``demand_score``    (reused from the investment engine's demand pillar),
  - ``value_before``    (proxy market value from the acquisition layer),

it produces:
  - a ``renovation_potential`` bucket (LOW / MEDIUM / HIGH),
  - a concrete ``strategy`` (Cosmetic Upgrade / Layout Optimization /
    Full Refurbishment),
  - deterministic ``estimated_cost`` and ``value_after`` anchored inside
    stated cost-% and uplift-% bands,
  - ``roi`` = (value_after - value_before - cost) / cost,
  - ``payback_period`` in years (uplift recouped via extra rent),
  - ``value_drivers``: explainable reasons why the ROI exists,
  - a plain-English ``decision`` (RENOVATE / HOLD / SKIP) refined by
    ROI AND payback, not ROI alone,
  - combined deal economics (``total_investment`` +
    ``post_renovation_roi``) when a buy price is supplied.

All thresholds are deterministic constants. Running the same inputs
twice produces identical outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


RenovationPotential = Literal["LOW", "MEDIUM", "HIGH"]
RenovationDecision = Literal["RENOVATE", "HOLD", "SKIP"]
RenovationStrategy = Literal[
    "Cosmetic Upgrade",
    "Layout Optimization",
    "Full Refurbishment",
]


# Livability bucket thresholds (tenant engine final_score is in [0, 1]).
_LIVABILITY_LOW_MAX = 0.55
_LIVABILITY_HIGH_MIN = 0.75

# Demand bucket thresholds (investment engine demand pillar in [0, 1]).
_DEMAND_HIGH_MIN = 0.70
_DEMAND_MED_MIN = 0.50

# Cost bands as fractions of value_before.
_COST_BAND_HIGH = (0.12, 0.18)
_COST_BAND_MED = (0.07, 0.12)
_COST_BAND_LOW = (0.03, 0.07)

# Value uplift bands as fractions of value_before.
_UPLIFT_BAND_HIGH = (0.15, 0.30)
_UPLIFT_BAND_MED = (0.08, 0.15)
_UPLIFT_BAND_LOW = (0.03, 0.08)

# Decision thresholds.
_ROI_STRONG_MIN = 0.20
_ROI_HOLD_MIN = 0.10
_PAYBACK_FAST_MAX = 5.0   # years
_PAYBACK_SLOW_MIN = 7.0   # years


@dataclass(frozen=True)
class RenovationEvaluation:
    area: str
    renovation_potential: RenovationPotential
    strategy: RenovationStrategy
    estimated_cost: float
    value_before: float
    value_after: float
    roi: float
    payback_period: float  # years; clamped to 99.0 for JSON safety
    decision: RenovationDecision
    reason: str
    value_drivers: List[str] = field(default_factory=list)
    total_investment: Optional[float] = None
    post_renovation_roi: Optional[float] = None


def _clamp_unit(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _bucket_livability(livability_score: float) -> str:
    s = _clamp_unit(livability_score)
    if s < _LIVABILITY_LOW_MAX:
        return "LOW"
    if s >= _LIVABILITY_HIGH_MIN:
        return "HIGH"
    return "MEDIUM"


def _bucket_demand(demand_score: float) -> str:
    s = _clamp_unit(demand_score)
    if s >= _DEMAND_HIGH_MIN:
        return "HIGH"
    if s >= _DEMAND_MED_MIN:
        return "MEDIUM"
    return "LOW"


def classify_renovation_potential(
    livability_score: float,
    demand_score: float,
) -> RenovationPotential:
    """Rules-only classifier matching the spec."""
    liv = _bucket_livability(livability_score)
    dem = _bucket_demand(demand_score)

    if liv == "HIGH":
        return "LOW"
    if liv == "LOW" and dem == "HIGH":
        return "HIGH"
    if liv == "LOW" and dem == "MEDIUM":
        return "MEDIUM"
    if liv == "LOW" and dem == "LOW":
        return "MEDIUM"
    if dem == "HIGH":
        return "HIGH"
    return "MEDIUM"


def _strategy_for(potential: RenovationPotential) -> RenovationStrategy:
    if potential == "LOW":
        return "Cosmetic Upgrade"
    if potential == "MEDIUM":
        return "Layout Optimization"
    return "Full Refurbishment"


def _interp(band: tuple[float, float], t: float) -> float:
    lo, hi = band
    t = _clamp_unit(t)
    return lo + (hi - lo) * t


def _cost_fraction(potential: RenovationPotential, demand_score: float) -> float:
    if potential == "HIGH":
        band = _COST_BAND_HIGH
    elif potential == "MEDIUM":
        band = _COST_BAND_MED
    else:
        band = _COST_BAND_LOW
    return _interp(band, _clamp_unit(demand_score))


def _uplift_fraction(
    potential: RenovationPotential,
    livability_score: float,
    demand_score: float,
) -> float:
    if potential == "HIGH":
        band = _UPLIFT_BAND_HIGH
    elif potential == "MEDIUM":
        band = _UPLIFT_BAND_MED
    else:
        band = _UPLIFT_BAND_LOW
    livability_gap = 1.0 - _clamp_unit(livability_score)
    t = 0.5 * _clamp_unit(demand_score) + 0.5 * livability_gap
    return _interp(band, t)


def _decide(roi: float, payback: float) -> RenovationDecision:
    """ROI + payback refined verdict.

        ROI > 20% AND payback < 5y  -> RENOVATE
        ROI > 20% AND payback > 7y  -> HOLD
        ROI < 10%                   -> SKIP
        otherwise                   -> HOLD
    """
    if roi < _ROI_HOLD_MIN:
        return "SKIP"
    if roi >= _ROI_STRONG_MIN and payback < _PAYBACK_FAST_MAX:
        return "RENOVATE"
    if roi >= _ROI_STRONG_MIN and payback > _PAYBACK_SLOW_MIN:
        return "HOLD"
    return "HOLD"


def _value_drivers(
    livability_score: float,
    demand_score: float,
    gross_rental_yield_pct: float,
    potential: RenovationPotential,
    roi: float,
) -> List[str]:
    """Explainable reasons the ROI exists. Purely derived from signals."""
    drivers: List[str] = []
    liv = _bucket_livability(livability_score)
    dem = _bucket_demand(demand_score)

    if liv == "LOW":
        drivers.append("Below-market livability score")
    elif liv == "MEDIUM":
        drivers.append("Room to lift livability above area peers")

    if dem == "HIGH":
        drivers.append("High tenant demand in area")
    elif dem == "MEDIUM":
        drivers.append("Steady tenant demand supports uplift absorption")

    if gross_rental_yield_pct >= 7.0:
        drivers.append("Strong rental yield accelerates payback")
    elif gross_rental_yield_pct >= 5.5:
        drivers.append("Healthy rental yield supports payback")

    if potential == "HIGH" and roi >= _ROI_STRONG_MIN:
        drivers.append("Price inefficiency relative to peers")

    if not drivers:
        drivers.append("Limited value-creation levers at current signals")
    return drivers


def _reason(
    decision: RenovationDecision,
    potential: RenovationPotential,
    livability_score: float,
    demand_score: float,
    roi: float,
    payback: float,
    strategy: RenovationStrategy,
) -> str:
    roi_pct = roi * 100.0
    liv_bucket = _bucket_livability(livability_score).lower()
    dem_bucket = _bucket_demand(demand_score).lower()

    if decision == "RENOVATE":
        if liv_bucket == "low" and dem_bucket == "high":
            return (
                f"Low current livability with strong tenant demand creates "
                f"high value-add opportunity: ~{roi_pct:.1f}% ROI via "
                f"{strategy.lower()}, payback ~{payback:.1f}y."
            )
        return (
            f"{strategy} delivers ~{roi_pct:.1f}% ROI with payback "
            f"~{payback:.1f}y; {dem_bucket} demand and {liv_bucket} "
            f"livability support the uplift."
        )

    if decision == "HOLD":
        if roi >= _ROI_STRONG_MIN and payback > _PAYBACK_SLOW_MIN:
            return (
                f"Strong ~{roi_pct:.1f}% ROI but slow payback "
                f"(~{payback:.1f}y) erodes time-value; hold until rents "
                f"or demand firm up."
            )
        return (
            f"Borderline economics at ~{roi_pct:.1f}% ROI and "
            f"~{payback:.1f}y payback; {strategy.lower()} is defensible "
            f"only if costs come in at the low end of the band."
        )

    if potential == "LOW":
        return (
            f"Already-high livability leaves little headroom for uplift; "
            f"~{roi_pct:.1f}% ROI does not justify renovation capital."
        )
    return (
        f"Projected ~{roi_pct:.1f}% ROI falls below the renovation "
        f"hurdle for {dem_bucket}-demand / {liv_bucket}-livability areas; skip."
    )


def _payback_years(
    annual_rent_before: float,
    uplift_fraction: float,
    cost_aed: float,
) -> float:
    if cost_aed <= 0.0:
        return 0.0
    if annual_rent_before <= 0.0 or uplift_fraction <= 0.0:
        return 99.0
    delta_rent = annual_rent_before * uplift_fraction
    if delta_rent <= 0.0:
        return 99.0
    years = cost_aed / delta_rent
    if years > 99.0:
        return 99.0
    return years


def evaluate_renovation(
    area: str,
    value_before: float,
    livability_score: float,
    demand_score: float,
    gross_rental_yield_pct: float,
    recommended_buy_price: Optional[float] = None,
) -> RenovationEvaluation:
    """Full renovation evaluation for a single area.

    When ``recommended_buy_price`` is provided (from the acquisition
    layer), the result is enriched with ``total_investment`` and
    ``post_renovation_roi`` so the full deal can be evaluated end-to-end.
    """
    value_before = max(0.0, value_before)
    potential = classify_renovation_potential(livability_score, demand_score)
    strategy = _strategy_for(potential)

    cost_fraction = _cost_fraction(potential, demand_score)
    uplift_fraction = _uplift_fraction(potential, livability_score, demand_score)

    estimated_cost = value_before * cost_fraction
    value_after = value_before * (1.0 + uplift_fraction)

    if estimated_cost > 0.0:
        roi = (value_after - value_before - estimated_cost) / estimated_cost
    else:
        roi = 0.0

    annual_rent_before = value_before * max(0.0, gross_rental_yield_pct) / 100.0
    payback = _payback_years(annual_rent_before, uplift_fraction, estimated_cost)

    decision = _decide(roi, payback)
    reason = _reason(
        decision, potential, livability_score, demand_score, roi, payback, strategy,
    )
    drivers = _value_drivers(
        livability_score, demand_score, gross_rental_yield_pct, potential, roi,
    )

    total_investment: Optional[float] = None
    post_renovation_roi: Optional[float] = None
    if recommended_buy_price is not None and recommended_buy_price > 0.0:
        total_investment = recommended_buy_price + estimated_cost
        if total_investment > 0.0:
            post_renovation_roi = (value_after - total_investment) / total_investment

    return RenovationEvaluation(
        area=area,
        renovation_potential=potential,
        strategy=strategy,
        estimated_cost=round(estimated_cost, 2),
        value_before=round(value_before, 2),
        value_after=round(value_after, 2),
        roi=round(roi, 4),
        payback_period=round(payback, 2),
        decision=decision,
        reason=reason,
        value_drivers=drivers,
        total_investment=(
            round(total_investment, 2) if total_investment is not None else None
        ),
        post_renovation_roi=(
            round(post_renovation_roi, 4)
            if post_renovation_roi is not None
            else None
        ),
    )
