"""Tenant Recommendation Service Layer (v3-DXB.3).

Orchestration-only. This module does NOT:
  - expose HTTP endpoints (no FastAPI here),
  - re-implement any scoring logic (core functions are called as-is),
  - mutate global state.

It glues together the pure functions in ``app.core.tenant_scoring`` with
the Pydantic models in ``app.models.tenant_input`` / ``tenant_output``,
producing a sorted, explainable list of recommendations.

The dataset is loaded from the Supabase ``dubai_areas`` table via
``_load_area_dataset``. An in-code mock dataset is retained strictly
as a resilience fallback if the DB is unreachable or returns no rows,
so the scoring pipeline never sees an empty input.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as _urlreq
from urllib.error import HTTPError, URLError

logger = logging.getLogger("app.services.tenant_service")

from app.core.tenant_scoring import (
    AccessibilityResult,
    AffordabilityResult,
    AvailabilityResult,
    ClusterWeightsResolved,
    CommuteResult,
    ConfidenceResult,
    FinalScoreResult,
    QualityResult,
    compute_accessibility_score,
    compute_affordability_score,
    compute_availability_score,
    compute_cluster_weights,
    compute_commute_score,
    compute_confidence_score,
    compute_final_score,
    compute_quality_score,
)
from app.models.tenant_input import TenantInput
from app.models.tenant_output import (
    Driver,
    Explanation,
    PillarScores,
    TenantOutput,
)


# ---------------------------------------------------------------------------
# Mock data source (10 Dubai areas).
# A Supabase-backed loader can replace _load_area_dataset() without any
# pipeline changes: same return type, same semantics.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AreaRecord:
    """Raw feature bundle for a single Dubai area."""

    area_name: str
    avg_rent_aed: float
    peak_commute_time: float
    traffic_congestion_index: float
    metro_access_score: float
    road_score: float
    walkability_score: float
    amenities_score: float
    lifestyle_score: float
    safety_score: float
    connectivity_score: float
    family_score: float
    parking_score: float
    listings_in_budget: int
    total_listings: int
    last_updated_days: int
    parking_available: bool = True


_MOCK_AREAS: Tuple[AreaRecord, ...] = (
    AreaRecord(
        area_name="Downtown Dubai",
        avg_rent_aed=12500.0,
        peak_commute_time=22.0,
        traffic_congestion_index=0.72,
        metro_access_score=0.95,
        road_score=0.70,
        walkability_score=0.88,
        amenities_score=0.95,
        lifestyle_score=0.97,
        safety_score=0.90,
        connectivity_score=0.92,
        family_score=0.55,
        parking_score=0.45,
        listings_in_budget=180,
        total_listings=420,
        last_updated_days=14,
        parking_available=True,
    ),
    AreaRecord(
        area_name="Dubai Marina",
        avg_rent_aed=11000.0,
        peak_commute_time=30.0,
        traffic_congestion_index=0.78,
        metro_access_score=0.80,
        road_score=0.65,
        walkability_score=0.82,
        amenities_score=0.88,
        lifestyle_score=0.94,
        safety_score=0.85,
        connectivity_score=0.82,
        family_score=0.58,
        parking_score=0.55,
        listings_in_budget=210,
        total_listings=510,
        last_updated_days=21,
        parking_available=True,
    ),
    AreaRecord(
        area_name="Jumeirah Village Circle",
        avg_rent_aed=6200.0,
        peak_commute_time=38.0,
        traffic_congestion_index=0.55,
        metro_access_score=0.25,
        road_score=0.68,
        walkability_score=0.52,
        amenities_score=0.68,
        lifestyle_score=0.60,
        safety_score=0.80,
        connectivity_score=0.58,
        family_score=0.78,
        parking_score=0.85,
        listings_in_budget=340,
        total_listings=720,
        last_updated_days=45,
        parking_available=True,
    ),
    AreaRecord(
        area_name="Business Bay",
        avg_rent_aed=9800.0,
        peak_commute_time=24.0,
        traffic_congestion_index=0.70,
        metro_access_score=0.82,
        road_score=0.68,
        walkability_score=0.74,
        amenities_score=0.82,
        lifestyle_score=0.86,
        safety_score=0.86,
        connectivity_score=0.88,
        family_score=0.55,
        parking_score=0.50,
        listings_in_budget=260,
        total_listings=580,
        last_updated_days=18,
        parking_available=True,
    ),
    AreaRecord(
        area_name="Dubai Silicon Oasis",
        avg_rent_aed=5400.0,
        peak_commute_time=42.0,
        traffic_congestion_index=0.48,
        metro_access_score=0.20,
        road_score=0.72,
        walkability_score=0.45,
        amenities_score=0.60,
        lifestyle_score=0.48,
        safety_score=0.84,
        connectivity_score=0.52,
        family_score=0.82,
        parking_score=0.90,
        listings_in_budget=150,
        total_listings=330,
        last_updated_days=60,
        parking_available=True,
    ),
    AreaRecord(
        area_name="Mirdif",
        avg_rent_aed=7200.0,
        peak_commute_time=40.0,
        traffic_congestion_index=0.52,
        metro_access_score=0.18,
        road_score=0.74,
        walkability_score=0.50,
        amenities_score=0.70,
        lifestyle_score=0.55,
        safety_score=0.90,
        connectivity_score=0.55,
        family_score=0.92,
        parking_score=0.92,
        listings_in_budget=120,
        total_listings=280,
        last_updated_days=75,
        parking_available=True,
    ),
    AreaRecord(
        area_name="Deira",
        avg_rent_aed=4800.0,
        peak_commute_time=28.0,
        traffic_congestion_index=0.65,
        metro_access_score=0.90,
        road_score=0.60,
        walkability_score=0.78,
        amenities_score=0.72,
        lifestyle_score=0.62,
        safety_score=0.70,
        connectivity_score=0.84,
        family_score=0.65,
        parking_score=0.40,
        listings_in_budget=310,
        total_listings=640,
        last_updated_days=35,
        parking_available=False,
    ),
    AreaRecord(
        area_name="Al Barsha",
        avg_rent_aed=6800.0,
        peak_commute_time=26.0,
        traffic_congestion_index=0.58,
        metro_access_score=0.78,
        road_score=0.70,
        walkability_score=0.62,
        amenities_score=0.78,
        lifestyle_score=0.68,
        safety_score=0.86,
        connectivity_score=0.74,
        family_score=0.84,
        parking_score=0.75,
        listings_in_budget=200,
        total_listings=450,
        last_updated_days=28,
        parking_available=True,
    ),
    AreaRecord(
        area_name="Palm Jumeirah",
        avg_rent_aed=22000.0,
        peak_commute_time=34.0,
        traffic_congestion_index=0.62,
        metro_access_score=0.30,
        road_score=0.55,
        walkability_score=0.60,
        amenities_score=0.90,
        lifestyle_score=0.96,
        safety_score=0.95,
        connectivity_score=0.60,
        family_score=0.80,
        parking_score=0.90,
        listings_in_budget=40,
        total_listings=190,
        last_updated_days=22,
        parking_available=True,
    ),
    AreaRecord(
        area_name="Dubai Sports City",
        avg_rent_aed=5000.0,
        peak_commute_time=44.0,
        traffic_congestion_index=0.50,
        metro_access_score=0.15,
        road_score=0.66,
        walkability_score=0.42,
        amenities_score=0.58,
        lifestyle_score=0.50,
        safety_score=0.82,
        connectivity_score=0.48,
        family_score=0.78,
        parking_score=0.92,
        listings_in_budget=4,
        total_listings=160,
        last_updated_days=110,
        parking_available=True,
    ),
)


# ---------------------------------------------------------------------------
# Supabase data source.
# Reads the ``dubai_areas`` table via the PostgREST endpoint using stdlib
# only (urllib + json) so no new Python dependency is introduced. The
# pipeline contract is preserved: returns a Tuple[AreaRecord, ...].
# ---------------------------------------------------------------------------


_DUBAI_AREAS_TABLE = "dubai_areas"

_FLOAT_SUBSCORE_FIELDS: Tuple[str, ...] = (
    "metro_access_score",
    "road_score",
    "walkability_score",
    "amenities_score",
    "lifestyle_score",
    "safety_score",
    "connectivity_score",
    "family_score",
    "parking_score",
    "traffic_congestion_index",
)

_FALLBACK_NEUTRAL_SUBSCORE = 0.50
_FALLBACK_RENT_AED = 0.0
_FALLBACK_COMMUTE_MIN = 30.0
_FALLBACK_INT_ZERO = 0


def _load_env_file_into_os_environ() -> None:
    """Best-effort load of the project-root .env file so VITE_SUPABASE_*
    variables are visible to this process without adding a dependency.

    Silent no-op if the file is missing or unreadable.
    """
    here = Path(__file__).resolve()
    for candidate in (
        here.parents[3] / ".env",
        here.parents[2] / ".env",
        Path.cwd() / ".env",
    ):
        try:
            if not candidate.is_file():
                continue
            for raw in candidate.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            return
        except OSError:
            continue


def _resolve_supabase_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Resolve (url, anon_key) for PostgREST access.

    Security: ONLY the anon key is ever used. The service-role key is
    intentionally NOT read here. RLS is assumed to be enforced on the
    ``dubai_areas`` table.
    """
    _load_env_file_into_os_environ()
    url = (
        os.environ.get("SUPABASE_URL")
        or os.environ.get("VITE_SUPABASE_URL")
    )
    anon_key = (
        os.environ.get("SUPABASE_ANON_KEY")
        or os.environ.get("VITE_SUPABASE_ANON_KEY")
        or os.environ.get("VITE_SUPABASE_SUPABASE_ANON_KEY")
    )
    if url:
        url = url.rstrip("/")
    return url, anon_key


# Module-level status surface so callers (e.g., the API layer) can
# detect when the dataset was served degraded. This is a visibility
# signal only; it does not alter the scoring pipeline contract.
DATA_SOURCE_STATUS: Dict[str, Any] = {
    "source": "unknown",     # "supabase" | "mock_fallback" | "unknown"
    "degraded": False,        # true when DB failed / empty
    "error": None,            # last error string (if any)
    "row_count": 0,
    "fallback_field_count": 0,
}


def _count_missing_fields(row: Dict[str, Any]) -> int:
    """Count how many expected fields are null/missing in a DB row."""
    expected = (
        "avg_rent_aed",
        "peak_commute_time",
        "traffic_congestion_index",
        "metro_access_score",
        "road_score",
        "walkability_score",
        "amenities_score",
        "lifestyle_score",
        "safety_score",
        "connectivity_score",
        "family_score",
        "parking_score",
        "listings_in_budget",
        "total_listings",
        "last_updated_days",
        "parking_available",
    )
    return sum(1 for f in expected if row.get(f) is None)


def _safe_float(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_clamped_unit(value: Any, fallback: float = _FALLBACK_NEUTRAL_SUBSCORE) -> float:
    v = _safe_float(value, fallback)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _safe_int(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return fallback


def _safe_bool(value: Any, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "t", "1", "yes", "y"):
            return True
        if low in ("false", "f", "0", "no", "n"):
            return False
    return fallback


def _row_to_area_record(row: Dict[str, Any]) -> Optional[AreaRecord]:
    """Convert a single DB row to AreaRecord, applying null-safe fallbacks.

    Returns None if the row is missing its primary key (``area_name``)
    and therefore cannot be meaningfully scored.
    """
    area_name = row.get("area_name")
    if not area_name or not isinstance(area_name, str):
        return None

    return AreaRecord(
        area_name=area_name,
        avg_rent_aed=_safe_float(row.get("avg_rent_aed"), _FALLBACK_RENT_AED),
        peak_commute_time=_safe_float(row.get("peak_commute_time"), _FALLBACK_COMMUTE_MIN),
        traffic_congestion_index=_safe_clamped_unit(row.get("traffic_congestion_index")),
        metro_access_score=_safe_clamped_unit(row.get("metro_access_score")),
        road_score=_safe_clamped_unit(row.get("road_score")),
        walkability_score=_safe_clamped_unit(row.get("walkability_score")),
        amenities_score=_safe_clamped_unit(row.get("amenities_score")),
        lifestyle_score=_safe_clamped_unit(row.get("lifestyle_score")),
        safety_score=_safe_clamped_unit(row.get("safety_score")),
        connectivity_score=_safe_clamped_unit(row.get("connectivity_score")),
        family_score=_safe_clamped_unit(row.get("family_score")),
        parking_score=_safe_clamped_unit(row.get("parking_score")),
        listings_in_budget=_safe_int(row.get("listings_in_budget"), _FALLBACK_INT_ZERO),
        total_listings=_safe_int(row.get("total_listings"), _FALLBACK_INT_ZERO),
        last_updated_days=_safe_int(row.get("last_updated_days"), _FALLBACK_INT_ZERO),
        parking_available=_safe_bool(row.get("parking_available"), True),
    )


def _fetch_dubai_areas_rows(
    timeout_seconds: Optional[float] = None,
    max_rows: Optional[int] = None,
    max_attempts: int = 2,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Fetch rows from ``dubai_areas`` via Supabase PostgREST (anon key).

    Security: uses the anon key ONLY. RLS is assumed enforced server-side.

    Behaviour:
      - Configurable timeout (env ``DUBAI_AREAS_TIMEOUT_SECONDS`` or arg),
        defaulting to 8.0s.
      - Configurable row limit (env ``DUBAI_AREAS_ROW_LIMIT`` or arg),
        defaulting to 100.
      - Simple retry: up to ``max_attempts`` attempts (default 2) with a
        small linear back-off.

    Returns a tuple ``(rows, error)``. On success ``error`` is ``None``.
    On failure ``rows`` is ``[]`` and ``error`` is a short description.
    Never raises.
    """
    if timeout_seconds is None:
        timeout_seconds = _safe_float(
            os.environ.get("DUBAI_AREAS_TIMEOUT_SECONDS"), 8.0
        )
    if max_rows is None:
        max_rows = _safe_int(os.environ.get("DUBAI_AREAS_ROW_LIMIT"), 100)
    if max_rows <= 0:
        max_rows = 100
    if max_attempts <= 0:
        max_attempts = 1

    url, anon_key = _resolve_supabase_credentials()
    if not url or not anon_key:
        err = "credentials_missing"
        logger.error("supabase:%s table=%s", err, _DUBAI_AREAS_TABLE)
        return [], err

    endpoint = (
        f"{url}/rest/v1/{_DUBAI_AREAS_TABLE}"
        "?select=area_name,avg_rent_aed,peak_commute_time,"
        "traffic_congestion_index,metro_access_score,road_score,"
        "walkability_score,amenities_score,lifestyle_score,safety_score,"
        "connectivity_score,family_score,parking_score,parking_available,"
        "listings_in_budget,total_listings,last_updated_days"
        "&order=area_name.asc"
        f"&limit={int(max_rows)}"
    )
    headers = {
        "apikey": anon_key,
        "Authorization": f"Bearer {anon_key}",
        "Accept": "application/json",
    }

    last_error: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        try:
            req = _urlreq.Request(endpoint, headers=headers, method="GET")
            with _urlreq.urlopen(req, timeout=timeout_seconds) as resp:
                payload = resp.read()
            data = json.loads(payload.decode("utf-8"))
            if not isinstance(data, list):
                last_error = "unexpected_payload_shape"
                logger.error(
                    "supabase:%s table=%s attempt=%d",
                    last_error, _DUBAI_AREAS_TABLE, attempt,
                )
                continue
            rows = [r for r in data if isinstance(r, dict)]
            if attempt > 1:
                logger.info(
                    "supabase:fetch_recovered table=%s attempt=%d rows=%d",
                    _DUBAI_AREAS_TABLE, attempt, len(rows),
                )
            return rows, None
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            last_error = f"{type(exc).__name__}:{exc}"
            logger.warning(
                "supabase:fetch_attempt_failed table=%s attempt=%d/%d err=%s",
                _DUBAI_AREAS_TABLE, attempt, max_attempts, last_error,
            )

    logger.error(
        "supabase:fetch_failed_all_attempts table=%s attempts=%d err=%s",
        _DUBAI_AREAS_TABLE, max_attempts, last_error,
    )
    return [], last_error or "unknown_error"


def _load_area_dataset() -> Tuple[AreaRecord, ...]:
    """Return the raw Dubai area dataset from Supabase.

    Source: ``dubai_areas`` table via PostgREST with the anon key. Rows
    are mapped to ``AreaRecord`` with per-field null safety and type
    coercion. Per-row fallback usage is logged.

    Visibility:
      - On DB failure or empty result, logs at ERROR level.
      - ``DATA_SOURCE_STATUS`` is updated so the API layer can surface
        a warning flag in the HTTP response. Mock data is ONLY used as
        a last-resort safety net and is explicitly marked degraded.
    """
    rows, fetch_error = _fetch_dubai_areas_rows()

    records: List[AreaRecord] = []
    total_fallback_fields = 0
    for row in rows:
        missing = _count_missing_fields(row)
        if missing > 0:
            total_fallback_fields += missing
            logger.warning(
                "dubai_areas:row_fallback area=%s missing_fields=%d",
                row.get("area_name", "<unknown>"), missing,
            )
        rec = _row_to_area_record(row)
        if rec is None:
            logger.warning(
                "dubai_areas:row_rejected reason=missing_area_name row_keys=%s",
                list(row.keys()),
            )
            continue
        records.append(rec)

    if records:
        DATA_SOURCE_STATUS.update(
            source="supabase",
            degraded=False,
            error=None,
            row_count=len(records),
            fallback_field_count=total_fallback_fields,
        )
        logger.info(
            "dubai_areas:loaded source=supabase count=%d fallback_fields=%d",
            len(records), total_fallback_fields,
        )
        return tuple(records)

    # No usable DB rows: surface the failure loudly and fall back only as
    # a last-resort safety net so the pipeline does not crash.
    reason = fetch_error or "empty_result"
    logger.error(
        "dubai_areas:degraded reason=%s using_mock_fallback count=%d",
        reason, len(_MOCK_AREAS),
    )
    DATA_SOURCE_STATUS.update(
        source="mock_fallback",
        degraded=True,
        error=reason,
        row_count=len(_MOCK_AREAS),
        fallback_field_count=0,
    )
    return _MOCK_AREAS


# ---------------------------------------------------------------------------
# Fallback handling for missing values.
# ---------------------------------------------------------------------------


_SUBSCORE_FIELDS: Tuple[str, ...] = (
    "metro_access_score",
    "road_score",
    "walkability_score",
    "amenities_score",
    "lifestyle_score",
    "safety_score",
    "connectivity_score",
    "family_score",
    "parking_score",
)

_NEUTRAL_FALLBACK: float = 0.50


def _coerce_subscores(area: AreaRecord) -> Tuple[Dict[str, float], int]:
    """Clamp [0,1] subscores and count fallback substitutions.

    A value of None or an out-of-band value is replaced with a neutral
    0.5. The count of substitutions is surfaced to the final-score
    penalty machinery.
    """
    resolved: Dict[str, float] = {}
    fallback_count = 0
    for field_name in _SUBSCORE_FIELDS:
        value = getattr(area, field_name, None)
        if value is None:
            resolved[field_name] = _NEUTRAL_FALLBACK
            fallback_count += 1
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            resolved[field_name] = _NEUTRAL_FALLBACK
            fallback_count += 1
            continue
        if v < 0.0 or v > 1.0:
            resolved[field_name] = _NEUTRAL_FALLBACK
            fallback_count += 1
            continue
        resolved[field_name] = v
    return resolved, fallback_count


# ---------------------------------------------------------------------------
# Confidence signal derivation.
# Keeps orchestration stateless: derives kappa / stability from the area
# record itself (listings depth, data age, household consistency).
# ---------------------------------------------------------------------------


def _derive_confidence_signals(
    area: AreaRecord,
    availability: AvailabilityResult,
) -> Tuple[float, float]:
    """Return (kappa, stability) in [0, 1].

    kappa: data agreement proxy -> listings depth (more listings in
           budget relative to total => more agreement on the price band).
    stability: volatility proxy -> freshness-linked; older data, less
           stable estimate. Bounded to [0.3, 1.0] so stale but non-empty
           data is not assigned zero stability.
    """
    if availability.total_listings <= 0:
        kappa = 0.0
    else:
        ratio = availability.listings_in_budget / availability.total_listings
        kappa = max(0.0, min(1.0, 0.5 + 0.5 * ratio))

    age = max(0, int(area.last_updated_days))
    if age <= 30:
        stability = 1.0
    elif age <= 90:
        stability = 0.85
    elif age <= 180:
        stability = 0.65
    elif age <= 365:
        stability = 0.45
    else:
        stability = 0.30
    return kappa, stability


# ---------------------------------------------------------------------------
# Budget pressure derivation for the cluster-weights step.
# ---------------------------------------------------------------------------


def _budget_pressure(area_rent: float, budget: float) -> float:
    """Ratio-based pressure in [0, 1].

    0.0  -> rent is a tiny fraction of budget (no pressure).
    1.0  -> rent at or above budget (max pressure).
    """
    if budget <= 0.0:
        return 1.0
    return max(0.0, min(1.0, area_rent / budget))


# ---------------------------------------------------------------------------
# Driver extraction (top positive / single worst negative).
# ---------------------------------------------------------------------------


_CLUSTER_OF_PILLAR: Dict[str, str] = {
    "accessibility": "accessibility",
    "quality": "quality",
    "affordability": "affordability",
}


def _build_drivers(
    pillars: Dict[str, float],
    cluster_weights: Dict[str, float],
) -> Tuple[List[Driver], Optional[Driver]]:
    """Derive top positive drivers and the single worst negative factor.

    Contribution = weight * (pillar_score - mean). Because pillar_score
    and weight are each in [0, 1] and mean is in [0, 1], the result is
    already naturally bounded within [-1, 1]; we do NOT apply a hard
    clamp here so that relative magnitudes between pillars are preserved
    exactly as the math produced them.
    """
    if not pillars:
        return [], None

    mean = sum(pillars.values()) / len(pillars)
    contributions: List[Tuple[str, float]] = []
    for name, score in pillars.items():
        weight = cluster_weights.get(name, 0.0)
        centered = (score - mean) * weight
        contributions.append((name, centered))

    contributions.sort(key=lambda kv: kv[1], reverse=True)

    positives: List[Driver] = []
    for name, contrib in contributions:
        if contrib <= 0.0:
            continue
        positives.append(
            Driver(
                feature=name,
                cluster=_CLUSTER_OF_PILLAR[name],
                contribution=contrib,
                note=f"{name} pillar lifts this area above its own average.",
            )
        )
        if len(positives) >= 3:
            break

    worst_name, worst_contrib = contributions[-1]
    negative: Optional[Driver] = None
    if worst_contrib < 0.0:
        negative = Driver(
            feature=worst_name,
            cluster=_CLUSTER_OF_PILLAR[worst_name],
            contribution=worst_contrib,
            note=f"{worst_name} is the weakest pillar for this area.",
        )
    return positives, negative


# ---------------------------------------------------------------------------
# Warnings surface.
# ---------------------------------------------------------------------------


def _build_warnings(
    area: AreaRecord,
    availability: AvailabilityResult,
    confidence: ConfidenceResult,
    fallback_feature_count: int,
    freshness_threshold_days: int,
    affordability: AffordabilityResult,
) -> List[str]:
    warnings: List[str] = []
    if availability.excluded:
        warnings.append("excluded:no_viable_inventory_in_budget")
    elif availability.in_penalty_band:
        warnings.append("availability:thin_inventory_in_budget")

    if confidence.low_confidence:
        warnings.append("confidence:low_confidence_result")

    if fallback_feature_count > 0:
        warnings.append(f"fallback:{fallback_feature_count}_feature_fields_substituted")

    if area.last_updated_days > freshness_threshold_days:
        warnings.append(
            f"freshness:data_older_than_threshold_{freshness_threshold_days}d"
        )

    if affordability.regime == "excluded":
        warnings.append("affordability:rent_exceeds_budget_envelope")
    elif affordability.regime == "exponential":
        warnings.append("affordability:rent_above_comfort_band")

    return warnings


# ---------------------------------------------------------------------------
# Explanation string (simple, deterministic placeholder).
# ---------------------------------------------------------------------------


def _build_explanation(
    area_name: str,
    final: FinalScoreResult,
    viable: bool,
    top_positive: List[Driver],
    top_negative: Optional[Driver],
) -> Explanation:
    if not viable:
        headline = (
            f"{area_name} is not viable at this budget or has no inventory."
        )
    elif top_positive:
        lead = top_positive[0]
        headline = (
            f"{area_name} scores {final.final_score:.2f}, lifted most by "
            f"{lead.feature} (+{lead.contribution:.2f})."
        )
    else:
        top_pillar = max(final.pillars.items(), key=lambda kv: kv[1])[0]
        headline = (
            f"{area_name} scores {final.final_score:.2f} overall, led by its "
            f"{top_pillar} pillar."
        )

    positive_frag = (
        ", ".join(
            f"{d.feature} (+{d.contribution:.2f})" for d in top_positive
        )
        if top_positive
        else "no pillar stands out above the area mean"
    )
    negative_frag = (
        f"{top_negative.feature} ({top_negative.contribution:+.2f})"
        if top_negative is not None
        else "no material drag"
    )
    details = (
        f"Strengths: {positive_frag}. Weakness: {negative_frag}. "
        f"Raw blended={final.raw_blended:.3f}, "
        f"penalty={final.penalty_applied:.3f}, "
        f"availability_factor={final.availability_factor:.3f}."
    )
    return Explanation(headline=headline, details=details)


# ---------------------------------------------------------------------------
# Per-area processing (STRICT 8-step order).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ScoredArea:
    """Internal bundle carrying a scored area through sorting/output."""

    output: TenantOutput
    final_score: float
    excluded: bool


def _score_one_area(area: AreaRecord, req: TenantInput) -> _ScoredArea:
    subs, fallback_feature_count = _coerce_subscores(area)

    # Step 1: affordability
    affordability: AffordabilityResult = compute_affordability_score(
        area_rent_aed=area.avg_rent_aed,
        budget_aed=req.budget_aed,
        owns_car=req.owns_car,
        parking_available=area.parking_available,
    )

    # Step 2: commute
    commute: CommuteResult = compute_commute_score(
        peak_commute_time=area.peak_commute_time,
        traffic_congestion_index=area.traffic_congestion_index,
        commute_sensitivity=req.commute_sensitivity,
    )

    # Step 3: accessibility
    accessibility: AccessibilityResult = compute_accessibility_score(
        metro_access_score=subs["metro_access_score"],
        road_subscore=subs["road_score"],
        walkability_subscore=subs["walkability_score"],
        commute=commute,
        owns_car=req.owns_car,
        commute_priority=req.commute_priority,
        metro_weight_override=req.preferences.metro_weight,
    )

    # Step 4: quality
    quality: QualityResult = compute_quality_score(
        amenities_subscore=subs["amenities_score"],
        lifestyle_subscore=subs["lifestyle_score"],
        safety_subscore=subs["safety_score"],
        connectivity_subscore=subs["connectivity_score"],
        family_subscore=subs["family_score"],
        parking_subscore=subs["parking_score"],
        household_type=req.household_type,
        amenities_weight=req.preferences.amenities_weight,
        lifestyle_weight=req.preferences.lifestyle_weight,
        safety_weight=req.preferences.safety_weight,
        connectivity_weight=req.preferences.connectivity_weight,
        family_weight=req.preferences.family_weight,
        parking_weight=req.preferences.parking_weight,
    )

    # Step 5: availability
    availability: AvailabilityResult = compute_availability_score(
        listings_in_budget=area.listings_in_budget,
        total_listings=area.total_listings,
    )

    # Step 6: cluster weights
    cluster_weights: ClusterWeightsResolved = compute_cluster_weights(
        commute_sensitivity=req.commute_sensitivity,
        household_type=req.household_type,
        budget_pressure=_budget_pressure(area.avg_rent_aed, req.budget_aed),
        accessibility_override=req.cluster_weights.accessibility,
        quality_override=req.cluster_weights.quality,
        affordability_override=req.cluster_weights.affordability,
    )

    # Step 7: confidence
    kappa, stability = _derive_confidence_signals(area, availability)
    confidence: ConfidenceResult = compute_confidence_score(
        kappa=kappa,
        stability=stability,
        data_age_days=float(area.last_updated_days),
    )

    # Step 8: final score. Core does not accept a continuous confidence
    # score, so we apply a continuous confidence adjustment to the final
    # value here (penalty proportional to (1 - confidence.score)). This
    # replaces the boolean-only treatment while preserving core purity.
    area_fallback_flag = fallback_feature_count > 0
    final: FinalScoreResult = compute_final_score(
        accessibility_score=accessibility.score,
        quality_score=quality.score,
        affordability_score=affordability.score,
        cluster_weights=cluster_weights,
        data_age_days=float(area.last_updated_days),
        area_fallback=area_fallback_flag,
        fallback_feature_count=fallback_feature_count,
        availability=availability,
        low_confidence=confidence.low_confidence,
    )

    _CONF_ADJUST_MAX = 0.10
    conf_adjust = 1.0 - (1.0 - confidence.score) * _CONF_ADJUST_MAX
    adjusted_final_score = max(0.0, min(1.0, final.final_score * conf_adjust))
    adjusted_penalty = max(
        0.0,
        min(1.0, final.penalty_applied + (1.0 - conf_adjust)),
    )

    # Output construction.
    pillars_model = PillarScores(
        accessibility=final.pillars["accessibility"],
        quality=final.pillars["quality"],
        affordability=final.pillars["affordability"],
    )
    top_positive, top_negative = _build_drivers(
        pillars=final.pillars,
        cluster_weights=final.cluster_weights,
    )
    viable = (not availability.excluded) and (affordability.regime != "excluded")
    explanation = _build_explanation(
        area.area_name,
        final,
        viable,
        top_positive,
        top_negative,
    )
    warnings = _build_warnings(
        area=area,
        availability=availability,
        confidence=confidence,
        fallback_feature_count=fallback_feature_count,
        freshness_threshold_days=req.freshness_threshold_days,
        affordability=affordability,
    )

    if availability.total_listings > 0:
        availability_ratio = (
            availability.listings_in_budget / availability.total_listings
        )
    else:
        availability_ratio = 0.0

    output = TenantOutput(
        area=area.area_name,
        final_score=adjusted_final_score,
        confidence_score=confidence.score,
        low_confidence=confidence.low_confidence,
        penalty_applied=adjusted_penalty,
        viable=viable,
        availability_ratio=availability_ratio,
        listings_in_budget=availability.listings_in_budget,
        pillars=pillars_model,
        top_positive_drivers=top_positive,
        top_negative_factor=top_negative,
        explanation=explanation,
        warnings=warnings,
    )
    return _ScoredArea(
        output=output,
        final_score=adjusted_final_score,
        excluded=availability.excluded,
    )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def recommend_areas(
    request: TenantInput,
    include_excluded: bool = False,
) -> List[TenantOutput]:
    """Run the full recommendation pipeline for a tenant request.

    Order of operations:
      1. Load dataset.
      2. For each area, run the 8-step scoring pipeline.
      3. Sort results by final_score DESC (excluded areas tie-break last).
      4. Filter out or retain excluded areas depending on ``include_excluded``.

    Returns a list of ``TenantOutput`` records safe to serialize.
    """
    dataset = _load_area_dataset()
    scored: List[_ScoredArea] = [_score_one_area(area, request) for area in dataset]

    # Sort in a single stable pass:
    #   1. viable (non-excluded) areas first (0 < 1),
    #   2. then by final_score DESC (negated so ascending sort puts
    #      higher scores first),
    #   3. then by area name ASC for deterministic tie-breaking.
    # We avoid reverse=True because it would also reverse the
    # alphabetical tie-breaker.
    scored.sort(
        key=lambda s: (
            0 if not s.excluded else 1,
            -s.final_score,
            s.output.area,
        )
    )

    if include_excluded:
        return [s.output for s in scored]
    return [s.output for s in scored if not s.excluded]


def get_data_source_status() -> Dict[str, Any]:
    """Return a normalized snapshot of the last dataset load status.

    Maps the internal ``DATA_SOURCE_STATUS`` surface to the public
    response shape: ``{degraded, source, reason}`` where ``source`` is
    one of ``"db"``, ``"mock_fallback"``, or ``"unknown"``.
    """
    raw_source = DATA_SOURCE_STATUS.get("source", "unknown")
    if raw_source == "supabase":
        source = "db"
    elif raw_source == "mock_fallback":
        source = "mock_fallback"
    else:
        source = "unknown"
    error = DATA_SOURCE_STATUS.get("error")
    return {
        "degraded": bool(DATA_SOURCE_STATUS.get("degraded", False)),
        "source": source,
        "reason": "" if error is None else str(error),
    }
