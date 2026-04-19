"""Acquisition Decision API layer.

Routing-only. Delegates all computation to
``app.services.acquisition_service``. The underlying scoring is still
performed by the existing investment engine; this endpoint just adds a
deterministic decision / pricing layer on top.

Mounted under the application's versioned prefix, so the public path is
``/api/v1/acquisition/analyze``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from app.models.investment_input import InvestmentInput
from app.models.investment_output import InvestmentDataSourceStatus
from app.services.acquisition_service import (
    analyze_acquisitions,
    get_acquisition_data_source_status,
)

logger = logging.getLogger("app.api.acquisition")

router = APIRouter(tags=["acquisition"])


class AcquisitionDecisionOut(BaseModel):
    area: str
    market_price: float = Field(..., description="Proxy market price in AED.")
    recommended_buy_price: float = Field(
        ..., description="Price at which acquisition is recommended, in AED."
    )
    discount_required: float = Field(
        ..., description="Discount fraction vs. market_price (0..1)."
    )
    roi_estimate: float = Field(..., description="ROI decimal, e.g. 0.072 = 7.2%/yr.")
    risk_score: float = Field(..., description="Risk score from investment engine (0..1).")
    decision: Literal["BUY", "HOLD", "PASS"]
    reason: str
    pricing_logic: str = Field(
        ...,
        description=(
            "Explainable narrative of why the required discount applies, "
            "framed in terms of ROI, risk and methodology."
        ),
    )


class AcquisitionAnalyzeResponse(BaseModel):
    results: List[AcquisitionDecisionOut]
    data_source: InvestmentDataSourceStatus


@router.post(
    "/acquisition/analyze",
    response_model=AcquisitionAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Produce acquisition decisions and recommended buy prices.",
    description=(
        "Runs the existing investment recommendation pipeline and then "
        "applies a deterministic acquisition decision layer that derives "
        "a proxy market price, a recommended buy price, and a "
        "BUY / HOLD / PASS verdict per area. No scoring logic is "
        "duplicated; ROI and risk are reused from the investment engine."
    ),
)
def analyze(request: InvestmentInput) -> AcquisitionAnalyzeResponse:
    logger.info(
        "acquisition.analyze:request budget=%s risk=%s horizon=%s",
        getattr(request, "budget_aed", None),
        getattr(request, "risk_level", None),
        getattr(request, "investment_horizon", None),
    )

    raw: List[Dict[str, Any]] = analyze_acquisitions(request)
    status_snapshot = get_acquisition_data_source_status()

    logger.info(
        "acquisition.analyze:response count=%d data_source=%s degraded=%s",
        len(raw),
        status_snapshot.get("source"),
        status_snapshot.get("degraded"),
    )

    return AcquisitionAnalyzeResponse(
        results=[AcquisitionDecisionOut(**row) for row in raw],
        data_source=InvestmentDataSourceStatus(**status_snapshot),
    )
