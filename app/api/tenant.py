"""Tenant Recommendation API layer.

Routing-only. This module:
  - declares the HTTP surface for tenant area recommendations,
  - delegates all computation to ``app.services.tenant_service``,
  - performs no scoring, weighting, or data transformation of its own.

The router is mounted by ``app.api.router`` under the application's
versioned prefix (``settings.API_V1_STR``), so the final path exposed
to clients is ``/api/v1/tenant/recommend``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, status

from app.models.tenant_input import TenantInput
from app.models.tenant_output import (
    DataSourceStatus,
    TenantRecommendResponse,
)
from app.services.tenant_service import get_data_source_status, recommend_areas

logger = logging.getLogger("app.api.tenant")

router = APIRouter(tags=["tenant"])


@router.post(
    "/tenant/recommend",
    response_model=TenantRecommendResponse,
    status_code=status.HTTP_200_OK,
    summary="Recommend Dubai areas for a tenant request.",
    description=(
        "Runs the tenant recommendation pipeline over the Dubai area "
        "dataset and returns a ranked list of TenantOutput records "
        "together with a data_source metadata block describing the "
        "origin and health of the dataset used for the scoring run."
    ),
)
def recommend(request: TenantInput) -> TenantRecommendResponse:
    logger.info(
        "tenant.recommend:request budget=%s lifestyle=%s priorities=%s",
        getattr(request, "budget", None),
        getattr(request, "lifestyle", None),
        getattr(request, "priorities", None),
    )

    results = recommend_areas(request)
    status_snapshot = get_data_source_status()

    count = len(results) if results else 0
    logger.info(
        "tenant.recommend:response count=%d data_source=%s degraded=%s",
        count,
        status_snapshot.get("source"),
        status_snapshot.get("degraded"),
    )

    return TenantRecommendResponse(
        results=results or [],
        data_source=DataSourceStatus(**status_snapshot),
    )
