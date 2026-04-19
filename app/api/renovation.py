"""Renovation / Value-Add API layer.

Routing-only. Delegates all computation to
``app.services.renovation_service``. The underlying scoring is
performed by the existing tenant and investment engines; this endpoint
adds a deterministic value-add decision layer on top.

Mounted under the application's versioned prefix, so the public path is
``/api/v1/renovation/analyze``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from app.models.investment_input import InvestmentInput
from app.models.investment_output import InvestmentDataSourceStatus
from app.services.renovation_service import (
    analyze_renovations,
    get_renovation_data_source_status,
)

logger = logging.getLogger("app.api.renovation")

router = APIRouter(tags=["renovation"])


class RenovationEvaluationOut(BaseModel):
    area: str
    renovation_potential: Literal["LOW", "MEDIUM", "HIGH"]
    strategy: Literal[
        "Cosmetic Upgrade",
        "Layout Optimization",
        "Full Refurbishment",
    ]
    estimated_cost: float = Field(..., description="Estimated renovation spend in AED.")
    value_before: float = Field(..., description="Estimated market value before works, in AED.")
    value_after: float = Field(..., description="Estimated market value after works, in AED.")
    roi: float = Field(
        ...,
        description=(
            "Return on renovation spend: "
            "(value_after - value_before - cost) / cost, as a decimal."
        ),
    )
    payback_period: float = Field(
        ...,
        description=(
            "Years to recoup renovation spend through incremental rent "
            "(uplift_fraction * annual rent)."
        ),
    )
    decision: Literal["RENOVATE", "HOLD", "SKIP"]
    reason: str
    value_drivers: List[str] = Field(
        default_factory=list,
        description="Explainable signal-level reasons the ROI exists.",
    )
    total_investment: Optional[float] = Field(
        default=None,
        description="Recommended buy price + renovation cost, in AED.",
    )
    post_renovation_roi: Optional[float] = Field(
        default=None,
        description=(
            "Combined deal ROI vs. total_investment: "
            "(value_after - total_investment) / total_investment."
        ),
    )


class RenovationAnalyzeResponse(BaseModel):
    results: List[RenovationEvaluationOut]
    data_source: InvestmentDataSourceStatus


@router.post(
    "/renovation/analyze",
    response_model=RenovationAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Quantify value-creation opportunities through renovation.",
    description=(
        "Reuses the tenant engine's livability score and the investment "
        "engine's demand pillar to classify each area's renovation "
        "potential, estimate cost and post-renovation value, compute "
        "ROI and payback, and issue a RENOVATE / HOLD / SKIP verdict. "
        "No scoring logic is duplicated."
    ),
)
def analyze(request: InvestmentInput) -> RenovationAnalyzeResponse:
    logger.info(
        "renovation.analyze:request budget=%s risk=%s horizon=%s",
        getattr(request, "budget_aed", None),
        getattr(request, "risk_level", None),
        getattr(request, "investment_horizon", None),
    )

    raw: List[Dict[str, Any]] = analyze_renovations(request)
    status_snapshot = get_renovation_data_source_status()

    logger.info(
        "renovation.analyze:response count=%d data_source=%s degraded=%s",
        len(raw),
        status_snapshot.get("source"),
        status_snapshot.get("degraded"),
    )

    return RenovationAnalyzeResponse(
        results=[RenovationEvaluationOut(**row) for row in raw],
        data_source=InvestmentDataSourceStatus(**status_snapshot),
    )
