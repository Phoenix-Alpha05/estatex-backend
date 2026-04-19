"""Acquisition Decision Service (orchestration only).

Reuses the investment engine's outputs verbatim. This service does NOT:
  - re-score areas,
  - duplicate any ROI / risk / yield computation,
  - mutate the existing investment service.

Flow:
  1. Call ``analyze_investments`` from the investment service.
  2. For each ``InvestmentOutput``, feed its ROI, risk, unit price, and
     gross yield into the deterministic ``evaluate_acquisition`` layer.
  3. Return a flat list of acquisition decisions plus the same
     data-source metadata block the investment endpoint already emits.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from app.core.acquisition_logic import AcquisitionDecision, evaluate_acquisition
from app.models.investment_input import InvestmentInput
from app.services.investment_service import (
    analyze_investments,
    get_data_source_status,
)


def analyze_acquisitions(request: InvestmentInput) -> List[Dict[str, Any]]:
    """Run the acquisition decision layer over all viable investment results."""
    investment_results = analyze_investments(request)

    decisions: List[AcquisitionDecision] = []
    for inv in investment_results:
        decisions.append(
            evaluate_acquisition(
                area=inv.area,
                avg_unit_price_aed=inv.avg_unit_price_aed,
                gross_rental_yield_pct=inv.gross_rental_yield_pct,
                roi_estimate=inv.roi_estimate,
                risk_score=inv.risk_score,
            )
        )

    _rank = {"BUY": 0, "HOLD": 1, "PASS": 2}
    decisions.sort(
        key=lambda d: (
            _rank.get(d.decision, 3),
            -d.roi_estimate,
            d.risk_score,
            d.area,
        )
    )
    return [asdict(d) for d in decisions]


def get_acquisition_data_source_status() -> Dict[str, Any]:
    """Reuse the investment service's data-source status as-is."""
    return get_data_source_status()
