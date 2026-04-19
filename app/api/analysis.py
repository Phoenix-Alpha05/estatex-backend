"""Dual Intelligence API layer.

Thin FastAPI router for /analysis/compare. Delegates all work to
``app.services.analysis_service``. Performs no scoring of its own.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, status

from app.models.analysis_output import CompareRequest, CompareResponse
from app.services.analysis_service import compare_areas

logger = logging.getLogger("app.api.analysis")

router = APIRouter(tags=["analysis"])


@router.post(
    "/analysis/compare",
    response_model=CompareResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare Dubai areas from tenant and investor perspectives.",
    description=(
        "Runs the tenant and investment engines against the same set of "
        "Dubai areas and returns a fused, deterministic cross-stakeholder "
        "view. Each result includes a tenant score, an investor score, a "
        "combined score, a tradeoff delta and a plain-language insight."
    ),
)
def compare(request: CompareRequest) -> CompareResponse:
    logger.info(
        "analysis.compare:request tenant_budget=%s invest_budget=%s risk=%s horizon=%s",
        request.tenant_budget_aed,
        request.investment_budget_aed,
        request.risk_level,
        request.investment_horizon,
    )
    response = compare_areas(request)
    logger.info(
        "analysis.compare:response count=%d tenant_degraded=%s invest_degraded=%s",
        len(response.results),
        response.tenant_data_source.degraded,
        response.investment_data_source.degraded,
    )
    return response
