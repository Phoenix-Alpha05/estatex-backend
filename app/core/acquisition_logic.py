"""Acquisition Decision Logic (deterministic, rules-only).

This module is a *pure* decision layer stacked on top of the existing
investment engine. It does NOT:
  - re-score areas,
  - implement any ML or statistical prediction,
  - call external APIs,
  - mutate global state.

Given an already-scored area (ROI estimate + risk score + unit price
+ gross rental yield), it produces:
  - a proxy ``market_price`` (rent multiplier model),
  - a ``recommended_buy_price`` after an ROI- and risk-driven discount,
  - a plain-English ``decision`` (BUY / HOLD / PASS) with a reason.

All thresholds are deterministic constants. Running the same inputs
twice produces identical outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Decision = Literal["BUY", "HOLD", "PASS"]


# Rent-multiplier bounds used to derive a proxy market valuation from
# annual rent. 15x..18x is the stated range; the midpoint is used as
# the point estimate (deterministic, no randomness).
_RENT_MULTIPLIER_LOW = 15.0
_RENT_MULTIPLIER_HIGH = 18.0
_RENT_MULTIPLIER_MID = (_RENT_MULTIPLIER_LOW + _RENT_MULTIPLIER_HIGH) / 2.0


# ROI buckets (ROI expressed as a decimal, e.g. 0.08 = 8% / year).
_ROI_HIGH = 0.08
_ROI_MED_LOW = 0.05

# Discount bands per ROI bucket, expressed as (min, max) fractions.
_DISCOUNT_BAND_HIGH_ROI = (0.00, 0.05)   # 0% .. 5%
_DISCOUNT_BAND_MED_ROI  = (0.05, 0.12)   # 5% .. 12%
_DISCOUNT_BAND_LOW_ROI  = (0.12, 0.25)   # 12% .. 25%

# Decision thresholds.
_DECISION_BUY_ROI_MIN = 0.08
_DECISION_BUY_RISK_MAX = 0.4
_DECISION_HOLD_ROI_MIN = 0.05
_DECISION_HOLD_ROI_MAX = 0.08
_DECISION_PASS_RISK_MAX = 0.7


@dataclass(frozen=True)
class AcquisitionDecision:
    area: str
    market_price: float
    recommended_buy_price: float
    discount_required: float  # fraction of market_price, 0..1
    roi_estimate: float       # decimal (e.g. 0.072 = 7.2% / yr)
    risk_score: float         # 0..1
    decision: Decision
    reason: str
    pricing_logic: str


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _annual_rent_aed(avg_unit_price_aed: float, gross_rental_yield_pct: float) -> float:
    """Derive annual rent from price and gross yield (both observed features)."""
    yield_decimal = max(0.0, gross_rental_yield_pct) / 100.0
    return max(0.0, avg_unit_price_aed) * yield_decimal


def estimate_market_price(
    avg_unit_price_aed: float,
    gross_rental_yield_pct: float,
) -> float:
    """Rent-multiplier proxy valuation.

    market_price = annual_rent * mid(15..18).

    If rent information is missing / zero, falls back to the observed
    average unit price so downstream math stays defined.
    """
    annual_rent = _annual_rent_aed(avg_unit_price_aed, gross_rental_yield_pct)
    if annual_rent <= 0.0:
        return max(0.0, avg_unit_price_aed)
    return annual_rent * _RENT_MULTIPLIER_MID


def _discount_for_roi(roi_estimate: float) -> tuple[float, float]:
    """Return (band_low, band_high) discount fractions for the ROI bucket."""
    if roi_estimate >= _ROI_HIGH:
        return _DISCOUNT_BAND_HIGH_ROI
    if roi_estimate >= _ROI_MED_LOW:
        return _DISCOUNT_BAND_MED_ROI
    return _DISCOUNT_BAND_LOW_ROI


def compute_required_discount(roi_estimate: float, risk_score: float) -> float:
    """Deterministic discount fraction (0..1) derived from ROI and risk.

    Within the ROI bucket's allowed band, risk linearly interpolates
    between the band's low and high end. risk 0.0 -> low end,
    risk 1.0 -> high end. ROI above 12%/yr nudges the discount toward
    the low end even within its band.
    """
    roi = max(0.0, roi_estimate)
    risk = _clamp(risk_score, 0.0, 1.0)

    band_low, band_high = _discount_for_roi(roi)
    base = band_low + (band_high - band_low) * risk

    # Exceptional-ROI tightening: if ROI is well above the high bucket,
    # trim any positive discount by a small deterministic amount.
    if roi >= 0.12 and base > band_low:
        base -= (base - band_low) * 0.4

    return _clamp(base, 0.0, 0.25)


def _decide(roi_estimate: float, risk_score: float) -> Decision:
    if roi_estimate < _DECISION_HOLD_ROI_MIN or risk_score > _DECISION_PASS_RISK_MAX:
        return "PASS"
    if roi_estimate >= _DECISION_BUY_ROI_MIN and risk_score < _DECISION_BUY_RISK_MAX:
        return "BUY"
    return "HOLD"


def _risk_label(risk_score: float) -> str:
    if risk_score < 0.35:
        return "controlled"
    if risk_score < 0.55:
        return "moderate"
    if risk_score <= 0.7:
        return "elevated"
    return "high"


def _roi_label(roi_estimate: float) -> str:
    if roi_estimate >= 0.10:
        return "strong"
    if roi_estimate >= _DECISION_BUY_ROI_MIN:
        return "healthy"
    if roi_estimate >= _DECISION_HOLD_ROI_MIN:
        return "moderate"
    return "thin"


def _reason(decision: Decision, roi_estimate: float, risk_score: float, discount: float) -> str:
    roi_pct = roi_estimate * 100.0
    disc_pct = discount * 100.0
    roi_label = _roi_label(roi_estimate)
    risk_label = _risk_label(risk_score)

    if decision == "BUY":
        if disc_pct < 2.0:
            tail = "this asset supports acquisition close to market value with minimal discount."
        else:
            tail = (
                f"this asset supports acquisition at roughly a "
                f"{disc_pct:.1f}% discount to estimated market value."
            )
        return (
            f"At ~{roi_pct:.1f}% projected ROI with {risk_label} risk exposure, "
            f"{tail}"
        )

    if decision == "HOLD":
        return (
            f"Projected ROI of ~{roi_pct:.1f}% at {risk_label} risk is "
            f"borderline: acquisition is defensible only near a "
            f"{disc_pct:.1f}% discount to estimated market value, "
            f"otherwise hold for a better entry."
        )

    # PASS variants
    if roi_estimate < _DECISION_HOLD_ROI_MIN and risk_score > _DECISION_PASS_RISK_MAX:
        return (
            f"Combination of {roi_label} ROI (~{roi_pct:.1f}%) and "
            f"{risk_label} risk fails the return-for-risk hurdle; "
            f"pass at current prices."
        )
    if roi_estimate < _DECISION_HOLD_ROI_MIN:
        return (
            f"Projected ROI of ~{roi_pct:.1f}% is {roi_label} relative "
            f"to capital commitment and discount required "
            f"(~{disc_pct:.1f}%) is not sufficient to justify entry; pass."
        )
    return (
        f"Risk exposure is {risk_label} ({risk_score:.2f}) against a "
        f"~{roi_pct:.1f}% ROI profile; reward does not compensate risk. Pass."
    )


def _pricing_logic(roi_estimate: float, risk_score: float, discount: float) -> str:
    """Short, explainable narrative of *why* this discount is required."""
    roi_pct = roi_estimate * 100.0
    disc_pct = discount * 100.0
    risk_label = _risk_label(risk_score)

    if roi_estimate >= _ROI_HIGH:
        bucket = (
            f"Strong ROI (~{roi_pct:.1f}%) clears the return threshold; "
            f"only a small entry discount (~{disc_pct:.1f}%) is needed "
            f"given {risk_label} risk."
        )
    elif roi_estimate >= _ROI_MED_LOW:
        bucket = (
            f"Moderate ROI (~{roi_pct:.1f}%) and {risk_label} risk "
            f"require a ~{disc_pct:.1f}% entry discount to meet the "
            f"return threshold."
        )
    else:
        bucket = (
            f"Thin ROI (~{roi_pct:.1f}%) combined with {risk_label} risk "
            f"requires a deep ~{disc_pct:.1f}% discount to estimated "
            f"market value before acquisition is defensible."
        )
    return (
        f"{bucket} Estimated market value is derived as an income-based "
        f"proxy (annual rent capitalised at a 15x-18x rent multiplier)."
    )


def evaluate_acquisition(
    area: str,
    avg_unit_price_aed: float,
    gross_rental_yield_pct: float,
    roi_estimate: float,
    risk_score: float,
) -> AcquisitionDecision:
    """Full acquisition evaluation for a single area.

    All pricing/decision math is deterministic and pure. Scoring values
    (ROI, risk) are read as-is from the investment engine and never
    recomputed here.
    """
    market_price = estimate_market_price(avg_unit_price_aed, gross_rental_yield_pct)
    discount = compute_required_discount(roi_estimate, risk_score)
    recommended = max(0.0, market_price * (1.0 - discount))
    decision = _decide(roi_estimate, risk_score)
    reason = _reason(decision, roi_estimate, risk_score, discount)
    pricing_logic = _pricing_logic(roi_estimate, risk_score, discount)
    return AcquisitionDecision(
        area=area,
        market_price=round(market_price, 2),
        recommended_buy_price=round(recommended, 2),
        discount_required=round(discount, 4),
        roi_estimate=round(roi_estimate, 4),
        risk_score=round(risk_score, 4),
        decision=decision,
        reason=reason,
        pricing_logic=pricing_logic,
    )
