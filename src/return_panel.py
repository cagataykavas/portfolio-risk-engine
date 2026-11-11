"""Fail-closed admission for portfolio return panels."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


class ReturnPanelError(ValueError):
    """Raised when a return panel is unsafe or ambiguous to process."""


@dataclass(frozen=True)
class ReturnPanelPolicy:
    """Resource and data-quality bounds applied before risk estimation."""

    policy_id: str = "return-panel-admission-v1"
    minimum_observations: int = 60
    maximum_observations: int = 1_000_000
    maximum_assets: int = 256
    maximum_cells: int = 5_000_000

    def __post_init__(self) -> None:
        for field in (
            "minimum_observations",
            "maximum_observations",
            "maximum_assets",
            "maximum_cells",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ReturnPanelError(f"{field} must be a positive integer")
        if self.minimum_observations > self.maximum_observations:
            raise ReturnPanelError("minimum_observations exceeds maximum_observations")


@dataclass(frozen=True)
class ReturnPanelReport:
    """Privacy-reduced evidence describing the panel admitted by the engine."""

    schema_version: str
    policy_id: str
    accepted: bool
    observations: int
    asset_count: int
    index_type: str
    period_start: str
    period_end: str
    content_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReturnPanelAdmission:
    """Validated return data and its audit report."""

    panel: pd.DataFrame
    report: ReturnPanelReport


def _validate_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 128:
        raise ReturnPanelError(f"invalid {field}")
    if any(ord(character) < 32 for character in value):
        raise ReturnPanelError(f"invalid {field}")
    return value


def _canonical_index_value(value: Any) -> dict[str, str]:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise ReturnPanelError("index contains missing values")
        return {"type": "timestamp", "value": value.isoformat()}
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, (np.integer, int)) and not isinstance(value, (np.bool_, bool)):
        return {"type": "integer", "value": str(int(value))}
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ReturnPanelError("index contains non-finite values")
        return {"type": "float", "value": number.hex()}
    if isinstance(value, str) and len(value) <= 256 and value:
        return {"type": "string", "value": value}
    raise ReturnPanelError("unsupported index value")


def _content_digest(panel: pd.DataFrame) -> str:
    digest = hashlib.sha256()

    def update_field(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    header = json.dumps(
        {
            "schema_version": "return-panel-content-v1",
            "columns": list(panel.columns),
            "shape": list(panel.shape),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    update_field(header)
    values = panel.to_numpy(dtype="<f8", copy=False)
    for index_value, row in zip(panel.index, values, strict=True):
        canonical_index = json.dumps(
            _canonical_index_value(index_value),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        update_field(canonical_index)
        update_field(np.asarray(row, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def admit_return_panel(
    returns: pd.DataFrame,
    assets: Sequence[str],
    weights: Sequence[float],
    *,
    policy: ReturnPanelPolicy | None = None,
) -> ReturnPanelAdmission:
    """Validate and select an ordered return panel before any risk calculation."""

    active_policy = policy or ReturnPanelPolicy()
    if not isinstance(returns, pd.DataFrame):
        raise ReturnPanelError("returns must be a pandas DataFrame")
    if returns.columns.has_duplicates:
        raise ReturnPanelError("return columns must be unique")

    asset_ids = [_validate_identifier(asset, "asset identifier") for asset in assets]
    if not asset_ids:
        raise ReturnPanelError("at least one asset is required")
    if len(asset_ids) != len(set(asset_ids)):
        raise ReturnPanelError("asset identifiers must be unique")
    if len(asset_ids) > active_policy.maximum_assets:
        raise ReturnPanelError("asset count exceeds policy limit")

    missing = [asset for asset in asset_ids if asset not in returns.columns]
    if missing:
        raise ReturnPanelError(f"missing return columns: {missing}")

    try:
        numeric_weights = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ReturnPanelError("weights must be numeric") from exc
    if numeric_weights.ndim != 1 or len(numeric_weights) != len(asset_ids):
        raise ReturnPanelError("weights must be one-dimensional and match assets")
    if not np.isfinite(numeric_weights).all():
        raise ReturnPanelError("weights contain non-finite values")
    if math.isclose(float(numeric_weights.sum()), 0.0, abs_tol=np.finfo(float).eps):
        raise ReturnPanelError("weights must not sum to zero")

    observations = len(returns)
    if observations < active_policy.minimum_observations:
        raise ReturnPanelError(
            f"return panel requires at least {active_policy.minimum_observations} observations"
        )
    if observations > active_policy.maximum_observations:
        raise ReturnPanelError("observation count exceeds policy limit")
    if observations * len(asset_ids) > active_policy.maximum_cells:
        raise ReturnPanelError("return panel exceeds cell budget")
    if isinstance(returns.index, pd.MultiIndex):
        raise ReturnPanelError("multi-level indexes are not supported")
    if returns.index.has_duplicates:
        raise ReturnPanelError("return index must be unique")
    try:
        monotonic = returns.index.is_monotonic_increasing
    except TypeError as exc:
        raise ReturnPanelError("return index values are not comparable") from exc
    if not monotonic:
        raise ReturnPanelError("return index must be ordered ascending")

    selected = returns.loc[:, asset_ids].copy()
    try:
        values = selected.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ReturnPanelError("returns must be numeric") from exc
    if not np.isfinite(values).all():
        raise ReturnPanelError("returns contain missing or non-finite values")
    if (values < -1.0).any():
        raise ReturnPanelError("simple returns cannot be below -100 percent")
    selected = pd.DataFrame(values, index=selected.index.copy(), columns=asset_ids)

    canonical_index = [_canonical_index_value(value) for value in selected.index]
    report = ReturnPanelReport(
        schema_version="1.0",
        policy_id=active_policy.policy_id,
        accepted=True,
        observations=observations,
        asset_count=len(asset_ids),
        index_type=type(selected.index).__name__,
        period_start=canonical_index[0]["value"],
        period_end=canonical_index[-1]["value"],
        content_sha256=_content_digest(selected),
    )
    return ReturnPanelAdmission(panel=selected, report=report)
