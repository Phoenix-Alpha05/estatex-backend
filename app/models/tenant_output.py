"""Tenant recommendation engine output models.

Defines the Pydantic v2 response schema returned by the scoring pipeline.
Only models live here. No business logic.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PillarScores(BaseModel):
    """Per-cluster (pillar) contribution to the final recommendation score.

    Each value is on a 0.0-1.0 scale after internal normalization.
    """

    model_config = ConfigDict(extra="forbid")

    accessibility: float = Field(
        ..., ge=0.0, le=1.0, description="Accessibility cluster score."
    )
    quality: float = Field(
        ..., ge=0.0, le=1.0, description="Quality cluster score."
    )
    affordability: float = Field(
        ..., ge=0.0, le=1.0, description="Affordability cluster score."
    )


class Driver(BaseModel):
    """Single human-readable contribution to the final score.

    Aligned with the architecture: each driver ties a feature to its
    originating cluster and a signed contribution magnitude.
    """

    model_config = ConfigDict(extra="forbid")

    feature: str = Field(
        ..., min_length=1, description="Feature name, e.g. 'metro_access'."
    )
    cluster: str = Field(
        ...,
        min_length=1,
        description="Originating cluster: 'accessibility' | 'quality' | 'affordability'.",
    )
    contribution: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Signed contribution to the final score, in [-1, 1].",
    )
    note: Optional[str] = Field(
        default=None, description="Optional short explanation of the driver."
    )


class Explanation(BaseModel):
    """Structured plain-language rationale for the recommendation."""

    model_config = ConfigDict(extra="forbid")

    headline: str = Field(
        ...,
        min_length=1,
        max_length=240,
        description="One-line summary of why this area was recommended.",
    )
    details: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional longer-form rationale, deterministic.",
    )


class TenantOutput(BaseModel):
    """Finalized recommendation record for a single Dubai area."""

    model_config = ConfigDict(extra="forbid")

    area: str = Field(..., min_length=1, description="Dubai area name.")
    final_score: float = Field(
        ..., ge=0.0, le=1.0, description="Blended recommendation score in [0, 1]."
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence (kappa/stability/freshness) in [0, 1].",
    )
    low_confidence: bool = Field(
        ...,
        description="True when confidence_score falls below the engine threshold.",
    )
    penalty_applied: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate penalty subtracted from the raw blended score.",
    )
    viable: bool = Field(
        ..., description="Whether the area passes hard gates (budget, availability)."
    )
    availability_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of listings in budget over total listings in the area.",
    )
    listings_in_budget: int = Field(
        ...,
        ge=0,
        description="Raw count of listings within the tenant's budget envelope.",
    )
    pillars: PillarScores = Field(
        ..., description="Per-cluster contribution breakdown."
    )
    top_positive_drivers: List[Driver] = Field(
        default_factory=list,
        max_length=5,
        description="Top-N positive factors lifting this area's score.",
    )
    top_negative_factor: Optional[Driver] = Field(
        default=None,
        description="Single strongest negative factor, if any.",
    )
    explanation: Explanation = Field(
        ...,
        description="Structured deterministic rationale.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Fallback and freshness warnings surfaced to the caller.",
    )


class DataSourceStatus(BaseModel):
    """Visibility signal describing how the dataset was served."""

    model_config = ConfigDict(extra="forbid")

    degraded: bool = Field(
        ..., description="True when the dataset was served from a degraded source."
    )
    source: str = Field(
        ...,
        min_length=1,
        description="Origin of the dataset: 'db' | 'mock_fallback' | 'unknown'.",
    )
    reason: str = Field(
        default="",
        description="Short machine-readable reason when degraded; empty otherwise.",
    )


class TenantRecommendResponse(BaseModel):
    """Wrapper response exposing recommendations plus data source metadata."""

    model_config = ConfigDict(extra="forbid")

    results: List[TenantOutput] = Field(
        default_factory=list,
        description="Ranked list of TenantOutput records.",
    )
    data_source: DataSourceStatus = Field(
        ..., description="Metadata describing the dataset origin and health."
    )
