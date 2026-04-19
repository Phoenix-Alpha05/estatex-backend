"""Tenant recommendation engine input models.

Defines the Pydantic v2 schema for incoming recommendation requests against
the finalized Dubai Tenant Recommendation Engine (v3-DXB.1).

Only models live here. No business logic, scoring, or side effects.
Defaults for preference and cluster weights are intentionally None; they
are resolved downstream from tenant_profile, household_type, and owns_car
so the model layer stays neutral.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CommutePriority(str, Enum):
    """Importance of a short work/city commute for this tenant."""

    LOW = "low"
    MED = "med"
    HIGH = "high"


class CommuteSensitivity(str, Enum):
    """How steeply perceived utility decays with commute minutes.

    Maps to the decay constant tau in {65, 35, 18}.
    """

    LOW = "low"
    MED = "med"
    HIGH = "high"


class HouseholdType(str, Enum):
    """Structure of the household occupying the unit."""

    SINGLE = "single"
    COUPLE = "couple"
    FAMILY = "family"


class TenantProfile(str, Enum):
    """High-level tenant archetype used to nudge default weights."""

    YOUNG_PROFESSIONAL = "young_professional"
    FAMILY = "family"
    BUDGET = "budget"


class PreferredLocation(BaseModel):
    """Optional anchor point for locality scoring (WGS84)."""

    model_config = ConfigDict(extra="forbid")

    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude in degrees.")
    lng: float = Field(..., ge=-180.0, le=180.0, description="Longitude in degrees.")


class Preferences(BaseModel):
    """Quality-cluster sub-weights.

    All fields default to None. The engine resolves unset weights from
    household_type, owns_car, and tenant_profile before scoring. Explicit
    values are always respected.
    """

    model_config = ConfigDict(extra="forbid")

    amenities_weight: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Weight for area amenities."
    )
    lifestyle_weight: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Weight for lifestyle/nightlife fit."
    )
    safety_weight: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Weight for safety score."
    )
    connectivity_weight: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Weight for general connectivity."
    )
    family_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Weight for family-friendliness. Defaults by household_type.",
    )
    parking_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Weight for parking availability. Defaults by owns_car.",
    )
    metro_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional explicit override of the dynamic accessibility-cluster "
            "metro share. When None, engine computes it from owns_car and "
            "commute_priority."
        ),
    )


class ClusterWeights(BaseModel):
    """Top-level cluster blend weights.

    All fields default to None. Partial inputs are resolved from
    tenant_profile defaults downstream. If every field is provided, the
    three values must sum to approximately 1.0 (tolerance 0.98-1.02) to
    absorb float precision.
    """

    model_config = ConfigDict(extra="forbid")

    accessibility: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Weight of the accessibility cluster."
    )
    quality: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Weight of the quality cluster."
    )
    affordability: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Weight of the affordability cluster."
    )

    @model_validator(mode="after")
    def _sum_close_to_one(self) -> "ClusterWeights":
        values = (self.accessibility, self.quality, self.affordability)
        if all(v is not None for v in values):
            total = sum(values)  # type: ignore[arg-type]
            if not (0.98 <= total <= 1.02):
                raise ValueError(
                    f"cluster_weights must sum to ~1.0 (got {total:.4f})."
                )
        return self


class TenantInput(BaseModel):
    """Finalized request payload for the Dubai Tenant Recommendation Engine.

    Declares every variable consumed by the scoring logic, with defensible
    defaults so a minimal request (budget only) is still scorable.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    budget_aed: float = Field(
        ...,
        gt=0.0,
        le=1_000_000.0,
        description="Monthly rental budget in AED. Required.",
    )
    commute_priority: CommutePriority = Field(
        default=CommutePriority.MED,
        description="Tenant's stated importance of short commute.",
    )
    commute_sensitivity: CommuteSensitivity = Field(
        default=CommuteSensitivity.MED,
        description="Decay sharpness for commute minutes -> utility.",
    )
    preferred_location: Optional[PreferredLocation] = Field(
        default=None,
        description="Optional geographic anchor for locality scoring.",
    )
    owns_car: bool = Field(
        default=False,
        description="Whether the tenant owns/drives a car in Dubai.",
    )
    household_type: HouseholdType = Field(
        default=HouseholdType.SINGLE,
        description="Household structure occupying the unit.",
    )
    tenant_profile: TenantProfile = Field(
        default=TenantProfile.YOUNG_PROFESSIONAL,
        description="Archetype used to nudge defaults without overriding inputs.",
    )
    preferences: Preferences = Field(
        default_factory=Preferences,
        description="Quality-cluster sub-weights. Unset fields resolved downstream.",
    )
    cluster_weights: ClusterWeights = Field(
        default_factory=ClusterWeights,
        description="Top-level blend. Unset fields resolved from tenant_profile.",
    )
    freshness_threshold_days: int = Field(
        default=90,
        ge=1,
        le=3650,
        description="Age (days) beyond which area data is considered stale.",
    )
