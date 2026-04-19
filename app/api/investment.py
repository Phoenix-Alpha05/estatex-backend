"""Investment Recommendation API layer.

Routing-only. This module:
  - declares the HTTP surface for investment analysis,
  - delegates all computation to ``app.services.investment_service``,
  - performs no scoring, weighting, or data transformation of its own.

The router is mounted by ``app.api.router`` under the application's
versioned prefix (``settings.API_V1_STR``), so the final path exposed
to clients is ``/api/v1/investment/analyze``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, status

from app.models.investment_input import InvestmentInput
from app.models.investment_output import (
    InvestmentAnalyzeResponse,
    InvestmentDataSourceStatus,
)
from app.services.investment_service import (
    analyze_investments,
    get_data_source_status,
)

logger = logging.getLogger("app.api.investment")

router = APIRouter(tags=["investment"])


@router.post(
    "/investment/analyze",
    response_model=InvestmentAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Dubai areas for an investment request.",
    description=(
        "Runs the investment recommendation pipeline over the Dubai "
        "investment dataset and returns a ranked list of InvestmentOutput "
        "records together with a data_source metadata block describing "
        "the origin and health of the dataset used for the analysis."
    ),
)
def analyze(request: InvestmentInput) -> InvestmentAnalyzeResponse:
    logger.info(
        "investment.analyze:request budget=%s risk=%s horizon=%s",
        getattr(request, "budget_aed", None),
        getattr(request, "risk_level", None),
        getattr(request, "investment_horizon", None),
    )

    results = analyze_investments(request)
    status_snapshot = get_data_source_status()

    count = len(results) if results else 0
    logger.info(
        "investment.analyze:response count=%d data_source=%s degraded=%s",
        count,
        status_snapshot.get("source"),
        status_snapshot.get("degraded"),
    )

    return InvestmentAnalyzeResponse(
        results=results or [],
        data_source=InvestmentDataSourceStatus(**status_snapshot),
    )
