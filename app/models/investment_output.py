"""Investment recommendation engine output models.

Public response schema for the Dubai Investment Recommendation Engine.
Pure Pydantic v2 - no business logic.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class InvestmentPillarScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rental_yield_score: float = Field(..., ge=0.0, le=1.0)
    appreciation_score: float = Field(..., ge=0.0, le=1.0)
    demand_score: float = Field(..., ge=0.0, le=1.0)
    liquidity_score: float = Field(..., ge=0.0, le=1.0)


class InvestmentDriver(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str = Field(..., min_length=1)
    pillar: str = Field(..., min_length=1)
    contribution: float = Field(..., ge=-1.0, le=1.0)
    note: str = Field(..., min_length=1)


class InvestmentExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str = Field(..., min_length=1, max_length=320)
    details: Optional[str] = Field(default=None, max_length=4000)


class InvestmentOutput(BaseModel):
    """Scored, explainable investment record for a single Dubai area."""

    model_config = ConfigDict(extra="forbid")

    area: str = Field(..., min_length=1)
    roi_estimate: float = Field(..., ge=-1.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    investment_score: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    low_confidence: bool = Field(default=False)
    viable: bool = Field(default=True)
    avg_unit_price_aed: float = Field(..., ge=0.0)
    gross_rental_yield_pct: float = Field(..., ge=0.0, le=100.0)
    penalty_applied: float = Field(..., ge=0.0, le=1.0)
    pillars: InvestmentPillarScores
    top_positive_drivers: List[InvestmentDriver] = Field(default_factory=list, max_length=10)
    top_negative_factor: Optional[InvestmentDriver] = Field(default=None)
    explanation: InvestmentExplanation
    warnings: List[str] = Field(default_factory=list, max_length=20)

    @property
    def liquidity_score(self) -> float:
        return self.pillars.liquidity_score

    @property
    def key_strengths(self) -> List[str]:
        return [f"{d.feature}" for d in self.top_positive_drivers]

    @property
    def key_risks(self) -> List[str]:
        risks: List[str] = list(self.warnings)
        if self.top_negative_factor is not None:
            risks.insert(0, self.top_negative_factor.feature)
        return risks


class InvestmentDataSourceStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    degraded: bool = Field(default=False)
    source: str = Field(default="unknown")
    reason: str = Field(default="")


class InvestmentAnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: List[InvestmentOutput] = Field(default_factory=list)
    data_source: InvestmentDataSourceStatus = Field(default_factory=InvestmentDataSourceStatus)
