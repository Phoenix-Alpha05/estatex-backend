"""HTTP surface for the Operational Workflow Integration Layer.

Exposes ``POST /workflow/run`` which fans out to the existing
investment / acquisition / renovation services and returns a unified
operational pipeline view: one record per area with priority, stage,
next action, confidence and a plain-English summary.

No business logic lives here; this module is purely request/response
shaping.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from app.models.investment_input import InvestmentInput
from app.services.workflow_service import (
    get_workflow_data_source_status,
    run_workflow,
)

router = APIRouter()
logger = logging.getLogger("app.api.workflow")


class WorkflowRecordOut(BaseModel):
    area: str
    acquisition_decision: Optional[Literal["BUY", "HOLD", "PASS"]] = None
    recommended_buy_price: Optional[float] = None
    renovation_decision: Optional[Literal["RENOVATE", "HOLD", "SKIP"]] = None
    renovation_potential: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    strategy: Optional[
        Literal["Cosmetic Upgrade", "Layout Optimization", "Full Refurbishment"]
    ] = None
    estimated_cost: Optional[float] = None
    renovation_roi: Optional[float] = Field(
        default=None,
        description="ROI on renovation spend only.",
    )
    payback_period: Optional[float] = None
    total_investment: Optional[float] = Field(
        default=None,
        description="Recommended buy price + renovation cost (AED).",
    )
    post_renovation_roi: Optional[float] = Field(
        default=None,
        description="Combined deal ROI vs. total_investment.",
    )
    value_drivers: List[str] = Field(default_factory=list)
    investment_score: float
    roi_estimate: float
    risk_score: float
    viable: bool
    priority: Literal["HIGH", "MEDIUM", "LOW", "DROP"]
    next_action: str
    stage: Literal["LEAD", "UNDER_REVIEW", "APPROVED"]
    confidence: float
    summary: str


class WorkflowResponse(BaseModel):
    count: int
    counts_by_priority: Dict[str, int]
    data_source: Dict[str, Any]
    results: List[WorkflowRecordOut]


@router.post(
    "/workflow/run",
    response_model=WorkflowResponse,
    status_code=status.HTTP_200_OK,
    summary="Run the unified investment -> acquisition -> renovation workflow",
)
def workflow_run_endpoint(request: InvestmentInput) -> WorkflowResponse:
    logger.info(
        "workflow:run budget=%s risk=%s horizon=%s",
        request.budget_aed, request.risk_level, request.investment_horizon,
    )
    records, counts = run_workflow(request)
    data_source = get_workflow_data_source_status()
    return WorkflowResponse(
        count=len(records),
        counts_by_priority=counts,
        data_source=data_source,
        results=[WorkflowRecordOut(**r) for r in records],
    )
