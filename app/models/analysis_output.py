"""Dual Intelligence analysis models.

Shared response schema for the cross-engine comparison endpoint
(/analysis/compare). The endpoint evaluates each Dubai area from
both the tenant and investor perspectives and surfaces the tradeoffs.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CompareDataSourceStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    degraded: bool = Field(default=False)
    source: str = Field(default="unknown")
    reason: str = Field(default="")


class DualAreaScore(BaseModel):
    """Per-area combined view merging tenant and investor signals."""

    model_config = ConfigDict(extra="forbid")

    area: str = Field(..., min_length=1)

    tenant_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    tenant_viable: bool = Field(default=False)
    tenant_headline: Optional[str] = Field(default=None, max_length=320)
    tenant_low_confidence: bool = Field(default=False)

    investment_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    investment_viable: bool = Field(default=False)
    investment_headline: Optional[str] = Field(default=None, max_length=320)
    investment_low_confidence: bool = Field(default=False)
    roi_estimate: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    avg_unit_price_aed: Optional[float] = Field(default=None, ge=0.0)
    gross_rental_yield_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)

    combined_score: float = Field(..., ge=0.0, le=1.0)
    tradeoff_delta: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="investment_score - tenant_score. Positive => investor-leaning.",
    )
    persona: str = Field(
        ...,
        description=(
            "Classification: 'balanced', 'tenant_leaning', 'investor_leaning', "
            "'tenant_only', 'investor_only', or 'weak'."
        ),
    )
    classification: str = Field(
        ...,
        description=(
            "Business decision label: 'LIVE + HOLD', 'LIVE (INVEST LATER)', "
            "'HOLD (RENT ELSEWHERE)', 'LIVE ONLY', 'HOLD ONLY', or 'AVOID'."
        ),
    )
    classification_reason: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description=(
            "Explainable justification for the classification, grounded in "
            "livability, rental yield, ROI, risk and unit price."
        ),
    )
    insight: str = Field(
        ...,
        min_length=1,
        max_length=320,
        description="One-line deterministic sentence summarising the tradeoff.",
    )


class CompareRequest(BaseModel):
    """Request payload for the dual-intelligence compare endpoint."""

    model_config = ConfigDict(extra="forbid")

    tenant_budget_aed: float = Field(
        default=12000.0,
        gt=0.0,
        le=1_000_000.0,
        description="Monthly rental budget used to run the tenant engine.",
    )
    investment_budget_aed: float = Field(
        default=2_000_000.0,
        gt=0.0,
        le=500_000_000.0,
        description="Capital budget used to run the investment engine.",
    )
    household_type: str = Field(default="single")
    tenant_profile: str = Field(default="young_professional")
    commute_priority: str = Field(default="med")
    commute_sensitivity: str = Field(default="med")
    owns_car: bool = Field(default=False)
    risk_level: str = Field(default="medium")
    investment_horizon: str = Field(default="medium")


class CompareResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: List[DualAreaScore] = Field(default_factory=list)
    tenant_data_source: CompareDataSourceStatus = Field(
        default_factory=CompareDataSourceStatus
    )
    investment_data_source: CompareDataSourceStatus = Field(
        default_factory=CompareDataSourceStatus
    )
    summary: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="Plain-language headline describing the cross-stakeholder landscape.",
    )
