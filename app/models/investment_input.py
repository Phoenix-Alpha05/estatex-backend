"""Investment recommendation engine input models.

Public request schema for the Dubai Investment Recommendation Engine.
Pure Pydantic v2 - no business logic, no side effects.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InvestmentHorizon(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class PillarWeights(BaseModel):
    """Optional per-pillar weight overrides for the investment engine."""

    model_config = ConfigDict(extra="forbid")

    rental_yield: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    appreciation: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    demand: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    liquidity: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class InvestmentInput(BaseModel):
    """Request payload for the Dubai Investment Recommendation Engine."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    budget_aed: float = Field(
        ...,
        gt=0.0,
        le=500_000_000.0,
        description="Total capital budget in AED (unit price cap).",
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Investor's tolerance for capital risk.",
    )
    investment_horizon: InvestmentHorizon = Field(
        ...,
        description="Planned holding period for the investment.",
    )
    preferred_locations: Optional[List[str]] = Field(
        default=None,
        max_length=50,
        description="Optional short-list of Dubai areas to bias toward.",
    )
    min_yield: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="Optional minimum gross rental yield in percent.",
    )
    max_risk: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional maximum acceptable aggregate risk score [0,1].",
    )
    budget_headroom: float = Field(
        default=1.15,
        ge=1.0,
        le=2.0,
        description=(
            "Multiplier on budget_aed used by the hard budget gate. "
            "Default 1.15 allows areas up to 15% over budget to stay viable."
        ),
    )
    pillar_weights: PillarWeights = Field(
        default_factory=PillarWeights,
        description="Optional pillar weight overrides.",
    )
    freshness_threshold_days: int = Field(
        default=90,
        ge=1,
        le=3650,
        description="Data age threshold in days beyond which freshness warnings fire.",
    )
