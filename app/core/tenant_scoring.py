"""Core scoring engine for the Dubai Tenant Recommendation Engine (v3-DXB.2).

This module is intentionally IO-free and framework-free. Every function is
pure and deterministic: same inputs always produce the same outputs, no
randomness, no database access, no HTTP calls, no filesystem access, no
global mutable state.

v3-DXB.2 refinements (fixes over v3-DXB.1):
  - Final score uses a MULTIPLICATIVE penalty: final = raw * (1 - penalty).
  - Accessibility: commute gating is applied UNIFORMLY to all three
    components (metro, road, walkability). Rationale: in Dubai, even
    walkability is conditioned on how reachable the destination zone is
    during peak commute; applying the gate everywhere keeps the pillar
    internally consistent and avoids a walkability-only escape hatch.
  - Metro weight ceiling reduced from 0.85 -> 0.65 to prevent the metro
    factor from dominating the pillar in extreme tenant configurations.
  - Availability is now integrated as a multiplicative factor on the final
    score, which also hard-excludes areas with no viable inventory.
  - Quality weights: when the caller provides partial overrides, only the
    NON-OVERRIDDEN slots are rebalanced (user intent is preserved).
  - Cluster weights: each adaptive weight is clamped to a sane [lo, hi]
    band BEFORE normalization to prevent extreme drift.
  - Confidence: a small explicit penalty is applied when the result is
    flagged as low-confidence (surfaced in the penalty bundle).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional


# ---------------------------------------------------------------------------
# Constants (finalized Dubai v3-DXB.2 logic).
# ---------------------------------------------------------------------------

COMMUTE_TAU: Dict[str, float] = {
    "low": 65.0,
    "med": 35.0,
    "high": 18.0,
}

TRAFFIC_CONGESTION_COEFF: float = 0.6

HOUSEHOLD_QUALITY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "single":  {"amenities": 0.22, "lifestyle": 0.28, "safety": 0.18, "connectivity": 0.14, "family": 0.08, "parking": 0.10},
    "couple":  {"amenities": 0.22, "lifestyle": 0.22, "safety": 0.18, "connectivity": 0.14, "family": 0.14, "parking": 0.10},
    "family":  {"amenities": 0.18, "lifestyle": 0.10, "safety": 0.22, "connectivity": 0.14, "family": 0.26, "parking": 0.10},
}

METRO_BASE_WEIGHT: float = 0.50
METRO_NO_CAR_BONUS: float = 0.20
METRO_PRIORITY_ADJUST: Dict[str, float] = {
    "low": -0.15,
    "med": 0.00,
    "high": 0.20,
}
METRO_WEIGHT_FLOOR: float = 0.10
METRO_WEIGHT_CEIL: float = 0.65  # reduced from 0.85 to prevent metro dominance

ROAD_SHARE_OWNS_CAR: float = 0.70
ROAD_SHARE_NO_CAR: float = 0.30

AFFORD_COMFORT_MAX: float = 0.60
AFFORD_LINEAR_MAX: float = 1.00
AFFORD_EXP_MAX: float = 1.30
AFFORD_LINEAR_END_SCORE: float = 0.70
AFFORD_EXP_DECAY_K: float = 7.0

PARKING_BONUS_OWNS_CAR: float = 0.05
PARKING_BONUS_NO_CAR: float = 0.00

AVAIL_EXCLUDE_BELOW: int = 3
AVAIL_PENALTY_BAND_MAX: int = 6
AVAIL_PENALTY_BAND_FACTOR: float = 0.65

FRESHNESS_TIERS = (
    (30,  0.00),
    (90,  0.05),
    (180, 0.10),
    (365, 0.15),
)
FRESHNESS_MAX_PENALTY: float = 0.20

FALLBACK_AREA_PENALTY: float = 0.05
FALLBACK_FEATURE_PENALTY_PER_FIELD: float = 0.02
FALLBACK_MAX_PENALTY: float = 0.15

LOW_CONFIDENCE_THRESHOLD: float = 0.55
LOW_CONFIDENCE_PENALTY: float = 0.05  # small penalty surfaced when flagged

CLUSTER_BASE_WEIGHTS: Dict[str, float] = {
    "accessibility": 0.33,
    "quality": 0.34,
    "affordability": 0.33,
}
COMMUTE_SENSITIVITY_ACCESS_BIAS: Dict[str, float] = {
    "low": -0.08,
    "med": 0.00,
    "high": 0.12,
}
HOUSEHOLD_QUALITY_BIAS: Dict[str, float] = {
    "single": -0.02,
    "couple": 0.02,
    "family": 0.10,
}
BUDGET_PRESSURE_AFFORD_BIAS: float = 0.18

# Clamp bands for cluster weights BEFORE normalization (prevents drift).
CLUSTER_WEIGHT_FLOOR: float = 0.15
CLUSTER_WEIGHT_CEIL: float = 0.60

_EPS: float = 1e-9


# ---------------------------------------------------------------------------
# Result dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AffordabilityResult:
    score: float
    rent_ratio: float
    regime: str
    raw_fit: float
    parking_bonus: float


@dataclass(frozen=True)
class CommuteResult:
    score: float
    tau: float
    peak_commute_time: float
    traffic_congestion_index: float
    effective_minutes: float


@dataclass(frozen=True)
class AccessibilityResult:
    score: float
    metro_weight: float
    road_weight: float
    walkability_weight: float
    metro_component: float
    road_component: float
    walkability_component: float
    commute: CommuteResult


@dataclass(frozen=True)
class QualityResult:
    score: float
    weights: Dict[str, float]
    subscores: Dict[str, float]


@dataclass(frozen=True)
class AvailabilityResult:
    score: float
    viable: bool
    excluded: bool
    in_penalty_band: bool
    listings_in_budget: int
    total_listings: int


@dataclass(frozen=True)
class ClusterWeightsResolved:
    accessibility: float
    quality: float
    affordability: float
    source: str


@dataclass(frozen=True)
class FinalScoreResult:
    final_score: float
    raw_blended: float
    penalty_applied: float
    pillars: Dict[str, float]
    cluster_weights: Dict[str, float]
    freshness_penalty: float
    fallback_penalty: float
    low_confidence_penalty: float
    availability_factor: float


@dataclass(frozen=True)
class ConfidenceResult:
    score: float
    low_confidence: bool
    kappa: float
    stability: float
    freshness: float
    components: Dict[str, float] = field(default_factory=dict)


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


# ---------------------------------------------------------------------------
# 1. Affordability (piecewise: comfort / linear / exponential / excluded).
# ---------------------------------------------------------------------------


def compute_affordability_score(
    area_rent_aed: float,
    budget_aed: float,
    owns_car: bool,
    parking_available: Optional[bool] = None,
) -> AffordabilityResult:
    if budget_aed <= 0.0:
        raise ValueError("budget_aed must be positive.")
    if area_rent_aed < 0.0:
        raise ValueError("area_rent_aed must be non-negative.")

    r = area_rent_aed / budget_aed

    if r <= AFFORD_COMFORT_MAX:
        regime = "comfort"
        raw_fit = 1.0
    elif r <= AFFORD_LINEAR_MAX:
        regime = "linear"
        span = AFFORD_LINEAR_MAX - AFFORD_COMFORT_MAX
        t = (r - AFFORD_COMFORT_MAX) / span
        raw_fit = 1.0 - t * (1.0 - AFFORD_LINEAR_END_SCORE)
    elif r < AFFORD_EXP_MAX:
        regime = "exponential"
        overshoot = r - AFFORD_LINEAR_MAX
        span = AFFORD_EXP_MAX - AFFORD_LINEAR_MAX
        floor = math.exp(-AFFORD_EXP_DECAY_K * span)
        shape = (math.exp(-AFFORD_EXP_DECAY_K * overshoot) - floor) / (1.0 - floor)
        raw_fit = AFFORD_LINEAR_END_SCORE * max(0.0, shape)
    else:
        regime = "excluded"
        raw_fit = 0.0

    parking_bonus = 0.0
    if owns_car and parking_available is True:
        parking_bonus = PARKING_BONUS_OWNS_CAR
    elif not owns_car:
        parking_bonus = PARKING_BONUS_NO_CAR

    score = _clamp(raw_fit + parking_bonus)
    if regime == "excluded":
        score = 0.0

    return AffordabilityResult(
        score=score,
        rent_ratio=r,
        regime=regime,
        raw_fit=raw_fit,
        parking_bonus=parking_bonus,
    )


# ---------------------------------------------------------------------------
# 2. Commute (peak time + congestion index).
# ---------------------------------------------------------------------------


def compute_commute_score(
    peak_commute_time: float,
    traffic_congestion_index: float,
    commute_sensitivity: str,
) -> CommuteResult:
    if peak_commute_time < 0.0:
        raise ValueError("peak_commute_time must be non-negative.")
    if not (0.0 <= traffic_congestion_index <= 1.0):
        raise ValueError("traffic_congestion_index must be in [0, 1].")
    if commute_sensitivity not in COMMUTE_TAU:
        raise ValueError(f"commute_sensitivity must be in {list(COMMUTE_TAU)}.")

    tau = COMMUTE_TAU[commute_sensitivity]
    t_eff = peak_commute_time * (1.0 + TRAFFIC_CONGESTION_COEFF * traffic_congestion_index)
    score = _clamp(math.exp(-t_eff / tau))

    return CommuteResult(
        score=score,
        tau=tau,
        peak_commute_time=peak_commute_time,
        traffic_congestion_index=traffic_congestion_index,
        effective_minutes=t_eff,
    )


# ---------------------------------------------------------------------------
# 3. Accessibility (dynamic metro weighting; UNIFORM commute gating).
# ---------------------------------------------------------------------------


def _dynamic_metro_weight(owns_car: bool, commute_priority: str) -> float:
    if commute_priority not in METRO_PRIORITY_ADJUST:
        raise ValueError("commute_priority must be low|med|high.")
    w = METRO_BASE_WEIGHT
    if not owns_car:
        w += METRO_NO_CAR_BONUS
    w += METRO_PRIORITY_ADJUST[commute_priority]
    return _clamp(w, METRO_WEIGHT_FLOOR, METRO_WEIGHT_CEIL)


def compute_accessibility_score(
    metro_access_score: float,
    road_subscore: float,
    walkability_subscore: float,
    commute: CommuteResult,
    owns_car: bool,
    commute_priority: str,
    metro_weight_override: Optional[float] = None,
) -> AccessibilityResult:
    for name, v in (
        ("metro_access_score", metro_access_score),
        ("road_subscore", road_subscore),
        ("walkability_subscore", walkability_subscore),
    ):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1] (got {v}).")

    if metro_weight_override is not None:
        metro_w = _clamp(metro_weight_override, METRO_WEIGHT_FLOOR, METRO_WEIGHT_CEIL)
    else:
        metro_w = _dynamic_metro_weight(owns_car, commute_priority)

    remainder = 1.0 - metro_w
    road_share = ROAD_SHARE_OWNS_CAR if owns_car else ROAD_SHARE_NO_CAR
    road_w = remainder * road_share
    walk_w = remainder * (1.0 - road_share)

    # Uniform commute gating: in Dubai, sprawl and heat mean even local
    # walkability depends on peak-hour reachability. Applying the gate to
    # all three components keeps the pillar internally consistent and
    # prevents walkability from acting as an escape hatch for isolated
    # areas that look walkable locally but are unreachable in rush hour.
    gate = commute.score
    metro_component = metro_access_score * gate
    road_component = road_subscore * gate
    walk_component = walkability_subscore * gate

    score = _clamp(
        metro_w * metro_component
        + road_w * road_component
        + walk_w * walk_component
    )

    return AccessibilityResult(
        score=score,
        metro_weight=metro_w,
        road_weight=road_w,
        walkability_weight=walk_w,
        metro_component=metro_component,
        road_component=road_component,
        walkability_component=walk_component,
        commute=commute,
    )


# ---------------------------------------------------------------------------
# 4. Quality (partial overrides preserve user intent).
# ---------------------------------------------------------------------------


def _resolve_quality_weights(
    household_type: str,
    overrides: Dict[str, Optional[float]],
) -> Dict[str, float]:
    """Blend household defaults with caller overrides.

    Behavior:
      - If NO overrides are provided, return the household defaults (already
        sum to 1.0 by construction, but normalized defensively).
      - If ALL slots are overridden, normalize the override set.
      - If PARTIAL overrides are provided, honor the overridden values as-is
        and rebalance ONLY the non-overridden slots proportionally so the
        total sums to 1.0. This preserves user intent on the explicit slots.
    """
    if household_type not in HOUSEHOLD_QUALITY_WEIGHTS:
        raise ValueError(
            f"household_type must be one of {list(HOUSEHOLD_QUALITY_WEIGHTS)}."
        )
    defaults = HOUSEHOLD_QUALITY_WEIGHTS[household_type]

    provided = {k: v for k, v in overrides.items() if v is not None}
    for k, v in provided.items():
        if v < 0.0:
            raise ValueError(f"{k} weight must be non-negative (got {v}).")

    if not provided:
        return _normalize(defaults)

    if len(provided) == len(defaults):
        return _normalize(provided)

    # Partial: keep provided values, rebalance the rest.
    provided_sum = sum(provided.values())
    if provided_sum >= 1.0:
        # User already exceeds 1.0; non-overridden slots collapse to 0
        # (preserves user intent) and we normalize the provided slots.
        resolved = {k: provided.get(k, 0.0) for k in defaults}
        return _normalize(resolved)

    remaining_budget = 1.0 - provided_sum
    non_overridden = {k: defaults[k] for k in defaults if k not in provided}
    default_remaining_sum = sum(non_overridden.values())

    resolved: Dict[str, float] = {}
    for k in defaults:
        if k in provided:
            resolved[k] = provided[k]
        else:
            if default_remaining_sum <= _EPS:
                resolved[k] = remaining_budget / max(1, len(non_overridden))
            else:
                resolved[k] = non_overridden[k] * (remaining_budget / default_remaining_sum)
    return resolved


def compute_quality_score(
    amenities_subscore: float,
    lifestyle_subscore: float,
    safety_subscore: float,
    connectivity_subscore: float,
    family_subscore: float,
    parking_subscore: float,
    household_type: str,
    amenities_weight: Optional[float] = None,
    lifestyle_weight: Optional[float] = None,
    safety_weight: Optional[float] = None,
    connectivity_weight: Optional[float] = None,
    family_weight: Optional[float] = None,
    parking_weight: Optional[float] = None,
) -> QualityResult:
    subs = {
        "amenities":    amenities_subscore,
        "lifestyle":    lifestyle_subscore,
        "safety":       safety_subscore,
        "connectivity": connectivity_subscore,
        "family":       family_subscore,
        "parking":      parking_subscore,
    }
    for name, v in subs.items():
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name}_subscore must be in [0, 1] (got {v}).")

    weights = _resolve_quality_weights(
        household_type=household_type,
        overrides={
            "amenities": amenities_weight,
            "lifestyle": lifestyle_weight,
            "safety": safety_weight,
            "connectivity": connectivity_weight,
            "family": family_weight,
            "parking": parking_weight,
        },
    )

    score = _clamp(sum(weights[k] * subs[k] for k in weights))
    return QualityResult(score=score, weights=weights, subscores=dict(subs))


# ---------------------------------------------------------------------------
# 5. Availability (exclude / penalty band / viable).
# ---------------------------------------------------------------------------


def compute_availability_score(
    listings_in_budget: int,
    total_listings: int,
    exclude_below: int = AVAIL_EXCLUDE_BELOW,
    penalty_band_max: int = AVAIL_PENALTY_BAND_MAX,
    penalty_band_factor: float = AVAIL_PENALTY_BAND_FACTOR,
) -> AvailabilityResult:
    if listings_in_budget < 0 or total_listings < 0:
        raise ValueError("listing counts must be non-negative.")
    if total_listings == 0:
        return AvailabilityResult(
            score=0.0, viable=False, excluded=True, in_penalty_band=False,
            listings_in_budget=listings_in_budget, total_listings=0,
        )

    ratio = listings_in_budget / total_listings

    if listings_in_budget < exclude_below:
        return AvailabilityResult(
            score=0.0, viable=False, excluded=True, in_penalty_band=False,
            listings_in_budget=listings_in_budget, total_listings=total_listings,
        )

    if listings_in_budget <= penalty_band_max:
        return AvailabilityResult(
            score=_clamp(ratio * penalty_band_factor),
            viable=True, excluded=False, in_penalty_band=True,
            listings_in_budget=listings_in_budget, total_listings=total_listings,
        )

    return AvailabilityResult(
        score=_clamp(ratio),
        viable=True, excluded=False, in_penalty_band=False,
        listings_in_budget=listings_in_budget, total_listings=total_listings,
    )


# ---------------------------------------------------------------------------
# 6. Cluster weight resolution (clamped before normalization).
# ---------------------------------------------------------------------------


def compute_cluster_weights(
    commute_sensitivity: str,
    household_type: str,
    budget_pressure: float,
    accessibility_override: Optional[float] = None,
    quality_override: Optional[float] = None,
    affordability_override: Optional[float] = None,
) -> ClusterWeightsResolved:
    if commute_sensitivity not in COMMUTE_SENSITIVITY_ACCESS_BIAS:
        raise ValueError("commute_sensitivity must be low|med|high.")
    if household_type not in HOUSEHOLD_QUALITY_BIAS:
        raise ValueError(f"household_type must be in {list(HOUSEHOLD_QUALITY_BIAS)}.")
    if not (0.0 <= budget_pressure <= 1.0):
        raise ValueError("budget_pressure must be in [0, 1].")

    adaptive = {
        "accessibility": CLUSTER_BASE_WEIGHTS["accessibility"]
                         + COMMUTE_SENSITIVITY_ACCESS_BIAS[commute_sensitivity],
        "quality":       CLUSTER_BASE_WEIGHTS["quality"]
                         + HOUSEHOLD_QUALITY_BIAS[household_type],
        "affordability": CLUSTER_BASE_WEIGHTS["affordability"]
                         + BUDGET_PRESSURE_AFFORD_BIAS * budget_pressure,
    }

    provided = [accessibility_override, quality_override, affordability_override]
    n_provided = sum(1 for v in provided if v is not None)
    if n_provided == 3:
        source = "override"
    elif n_provided == 0:
        source = "adaptive"
    else:
        source = "mixed"

    resolved = {
        "accessibility": accessibility_override if accessibility_override is not None else adaptive["accessibility"],
        "quality":       quality_override       if quality_override       is not None else adaptive["quality"],
        "affordability": affordability_override if affordability_override is not None else adaptive["affordability"],
    }

    # Clamp BEFORE normalization to prevent extreme drift from biases.
    clamped = {
        k: _clamp(v, CLUSTER_WEIGHT_FLOOR, CLUSTER_WEIGHT_CEIL)
        for k, v in resolved.items()
    }
    norm = _normalize(clamped)
    return ClusterWeightsResolved(
        accessibility=norm["accessibility"],
        quality=norm["quality"],
        affordability=norm["affordability"],
        source=source,
    )


# ---------------------------------------------------------------------------
# 7. Final score + penalties (multiplicative; availability integrated).
# ---------------------------------------------------------------------------


def _compute_freshness_penalty(data_age_days: float) -> float:
    if data_age_days < 0.0:
        raise ValueError("data_age_days must be non-negative.")
    penalty = FRESHNESS_MAX_PENALTY
    for max_days, p in FRESHNESS_TIERS:
        if data_age_days <= max_days:
            penalty = p
            break
    return penalty


def _compute_fallback_penalty(
    area_fallback: bool, fallback_feature_count: int
) -> float:
    if fallback_feature_count < 0:
        raise ValueError("fallback_feature_count must be non-negative.")
    raw = 0.0
    if area_fallback:
        raw += FALLBACK_AREA_PENALTY
    raw += fallback_feature_count * FALLBACK_FEATURE_PENALTY_PER_FIELD
    return min(FALLBACK_MAX_PENALTY, raw)


def compute_final_score(
    accessibility_score: float,
    quality_score: float,
    affordability_score: float,
    cluster_weights: ClusterWeightsResolved,
    data_age_days: float,
    area_fallback: bool = False,
    fallback_feature_count: int = 0,
    availability: Optional[AvailabilityResult] = None,
    low_confidence: bool = False,
) -> FinalScoreResult:
    for name, v in (
        ("accessibility_score", accessibility_score),
        ("quality_score", quality_score),
        ("affordability_score", affordability_score),
    ):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1] (got {v}).")

    pillars = {
        "accessibility": accessibility_score,
        "quality":       quality_score,
        "affordability": affordability_score,
    }
    w = {
        "accessibility": cluster_weights.accessibility,
        "quality":       cluster_weights.quality,
        "affordability": cluster_weights.affordability,
    }
    raw_blended = sum(w[k] * pillars[k] for k in pillars)

    freshness = _compute_freshness_penalty(data_age_days)
    fallback = _compute_fallback_penalty(area_fallback, fallback_feature_count)
    low_conf_pen = LOW_CONFIDENCE_PENALTY if low_confidence else 0.0
    penalty = _clamp(freshness + fallback + low_conf_pen, 0.0, 1.0)

    # Availability as a multiplicative gate: excluded areas collapse to 0,
    # viable areas scale final by their (penalty-band aware) availability.
    if availability is not None:
        if availability.excluded:
            avail_factor = 0.0
        else:
            avail_factor = _clamp(availability.score)
    else:
        avail_factor = 1.0

    # Multiplicative penalty application (replaces subtraction).
    final = _clamp(raw_blended * (1.0 - penalty) * avail_factor)

    return FinalScoreResult(
        final_score=final,
        raw_blended=raw_blended,
        penalty_applied=penalty,
        pillars=pillars,
        cluster_weights=w,
        freshness_penalty=freshness,
        fallback_penalty=fallback,
        low_confidence_penalty=low_conf_pen,
        availability_factor=avail_factor,
    )


# ---------------------------------------------------------------------------
# 8. Confidence.
# ---------------------------------------------------------------------------


def compute_confidence_score(
    kappa: float,
    stability: float,
    data_age_days: float,
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> ConfidenceResult:
    for name, v in (("kappa", kappa), ("stability", stability)):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1] (got {v}).")
    if data_age_days < 0.0:
        raise ValueError("data_age_days must be non-negative.")
    if not (0.0 <= low_confidence_threshold <= 1.0):
        raise ValueError("low_confidence_threshold must be in [0, 1].")

    tiered = _compute_freshness_penalty(data_age_days)
    freshness = _clamp(1.0 - (tiered / FRESHNESS_MAX_PENALTY if FRESHNESS_MAX_PENALTY > 0 else 0.0))

    w_kappa, w_stab, w_fresh = 0.4, 0.3, 0.3
    score = _clamp(w_kappa * kappa + w_stab * stability + w_fresh * freshness)
    low = score < low_confidence_threshold

    return ConfidenceResult(
        score=score,
        low_confidence=low,
        kappa=kappa,
        stability=stability,
        freshness=freshness,
        components={
            "kappa_weight":     w_kappa,
            "stability_weight": w_stab,
            "freshness_weight": w_fresh,
            "low_confidence_penalty": LOW_CONFIDENCE_PENALTY if low else 0.0,
        },
    )
