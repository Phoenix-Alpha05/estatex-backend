"""Renovation / Value-Add Service (orchestration only).

Reuses signals produced by the existing tenant and investment engines.
This service does NOT:
  - re-score areas,
  - duplicate any scoring math,
  - mutate the existing tenant / investment services.

Flow:
  1. Call ``analyze_investments`` to obtain demand_score, avg_unit_price
     and gross_rental_yield per area (investment engine output).
  2. Call ``recommend_areas`` with a neutral, budget-relaxed TenantInput
     to obtain an area-level livability proxy (tenant engine's
     ``final_score``, reused verbatim).
  3. For every area present in BOTH outputs, derive ``value_before``
     via the existing acquisition-layer market-price proxy and feed
     the trio into the deterministic renovation evaluator.
  4. Return a list of renovation evaluations plus the shared data-source
     status block the other endpoints already emit.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from app.core.acquisition_logic import (
    compute_required_discount,
    estimate_market_price,
)
from app.core.renovation_logic import RenovationEvaluation, evaluate_renovation
from app.models.investment_input import InvestmentInput
from app.models.tenant_input import (
    CommutePriority,
    CommuteSensitivity,
    HouseholdType,
    TenantInput,
    TenantProfile,
)
from app.services.investment_service import (
    analyze_investments,
    get_data_source_status,
)
from app.services.tenant_service import recommend_areas


# A fixed, budget-relaxed tenant request used purely as a vehicle to
# reuse the tenant engine's livability scoring across all areas. Using
# a neutral (non-biased) profile and a very high budget means no area
# is excluded for affordability, so ``final_score`` reflects the area's
# intrinsic liveability rather than a specific tenant's fit.
_NEUTRAL_LIVABILITY_REQUEST = TenantInput(
    budget_aed=1_000_000.0,
    commute_priority=CommutePriority.MED,
    commute_sensitivity=CommuteSensitivity.MED,
    owns_car=True,
    household_type=HouseholdType.FAMILY,
    tenant_profile=TenantProfile.FAMILY,
)


def _livability_scores_by_area() -> Dict[str, float]:
    """Map of area -> tenant engine final_score (livability proxy)."""
    tenant_results = recommend_areas(
        _NEUTRAL_LIVABILITY_REQUEST,
        include_excluded=True,
    )
    return {t.area: float(t.final_score) for t in tenant_results}


def analyze_renovations(request: InvestmentInput) -> List[Dict[str, Any]]:
    """Run the deterministic renovation decision layer over all viable areas."""
    investment_results = analyze_investments(request)
    livability_by_area = _livability_scores_by_area()

    evaluations: List[RenovationEvaluation] = []
    for inv in investment_results:
        livability_score = livability_by_area.get(inv.area)
        if livability_score is None:
            # Area scored by the investment engine but not present in the
            # tenant dataset: fall back to a neutral 0.5 livability so
            # the pipeline never silently drops an area.
            livability_score = 0.5

        demand_score = float(inv.pillars.demand_score)
        value_before = estimate_market_price(
            avg_unit_price_aed=inv.avg_unit_price_aed,
            gross_rental_yield_pct=inv.gross_rental_yield_pct,
        )

        # Reuse the acquisition layer's pricing logic (no duplication) to
        # derive the recommended buy price, enabling combined deal ROI.
        discount = compute_required_discount(
            roi_estimate=float(inv.roi_estimate),
            risk_score=float(inv.risk_score),
        )
        recommended_buy_price = max(0.0, value_before * (1.0 - discount))

        evaluations.append(
            evaluate_renovation(
                area=inv.area,
                value_before=value_before,
                livability_score=livability_score,
                demand_score=demand_score,
                gross_rental_yield_pct=inv.gross_rental_yield_pct,
                recommended_buy_price=recommended_buy_price,
            )
        )

    _rank = {"RENOVATE": 0, "HOLD": 1, "SKIP": 2}
    evaluations.sort(
        key=lambda e: (
            _rank.get(e.decision, 3),
            -e.roi,
            e.payback_period,
            e.area,
        )
    )
    return [asdict(e) for e in evaluations]


def get_renovation_data_source_status() -> Dict[str, Any]:
    """Reuse the investment service's data-source status verbatim."""
    return get_data_source_status()
