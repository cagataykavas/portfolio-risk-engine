from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_ASSETS = 256


class RiskAllocationError(ValueError):
    """Raised when a risk-allocation artifact or policy is malformed."""


@dataclass(frozen=True)
class Position:
    asset_id: str
    weight: float


@dataclass(frozen=True)
class RiskAllocationArtifact:
    generated_at: datetime
    portfolio_id: str
    risk_model_digest: str
    covariance_snapshot_digest: str
    positions: tuple[Position, ...]
    covariance: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": _format_timestamp(self.generated_at),
            "portfolio_id": self.portfolio_id,
            "risk_model_digest": self.risk_model_digest,
            "covariance_snapshot_digest": self.covariance_snapshot_digest,
            "positions": [asdict(position) for position in self.positions],
            "covariance": [list(row) for row in self.covariance],
        }


@dataclass(frozen=True)
class AllocationPolicy:
    max_age_seconds: int = 86_400
    max_future_skew_seconds: int = 60
    max_gross_leverage: float = 2.0
    net_exposure_target: float = 1.0
    net_exposure_tolerance: float = 1e-8
    max_absolute_risk_share: float = 0.55
    min_effective_risk_positions: float = 2.0
    max_euler_relative_error: float = 1e-10
    max_gradient_relative_error: float = 1e-6
    finite_difference_step: float = 1e-6
    symmetry_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        for name in ("max_age_seconds", "max_future_skew_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RiskAllocationError(f"{name} must be a non-negative integer")
        for name in (
            "max_gross_leverage",
            "net_exposure_tolerance",
            "max_euler_relative_error",
            "max_gradient_relative_error",
            "finite_difference_step",
            "symmetry_tolerance",
        ):
            value = _policy_float(getattr(self, name), name)
            if value <= 0:
                raise RiskAllocationError(f"{name} must be greater than zero")
        _policy_float(self.net_exposure_target, "net_exposure_target")
        if not 0.0 < _policy_float(self.max_absolute_risk_share, "max_absolute_risk_share") <= 1.0:
            raise RiskAllocationError("max_absolute_risk_share must be within (0, 1]")
        if _policy_float(self.min_effective_risk_positions, "min_effective_risk_positions") < 1.0:
            raise RiskAllocationError("min_effective_risk_positions must be at least one")


@dataclass(frozen=True)
class AssetRiskAllocation:
    asset_ref: str
    marginal_volatility: float
    component_volatility: float
    signed_risk_share: float
    absolute_risk_budget_share: float
    finite_difference_marginal: float
    gradient_relative_error: float


@dataclass(frozen=True)
class RiskAllocationReport:
    schema_version: str
    accepted: bool
    reasons: tuple[str, ...]
    evidence_digest: str
    portfolio_ref: str
    risk_model_digest: str
    covariance_snapshot_digest: str
    generated_at: str
    asset_count: int
    net_exposure: float
    gross_leverage: float
    portfolio_volatility: float
    component_sum: float
    euler_relative_error: float
    max_gradient_relative_error: float
    max_absolute_risk_share: float
    risk_budget_hhi: float
    effective_risk_positions: float
    negative_component_count: int
    allocations: tuple[AssetRiskAllocation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "reasons": list(self.reasons),
            "allocations": [asdict(allocation) for allocation in self.allocations],
        }


def _policy_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RiskAllocationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RiskAllocationError(f"{name} must be finite")
    return result


def _canonical_json(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise RiskAllocationError("artifact contains a non-canonical value") from error
    return rendered.encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validate_identifier(value: object, path: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value):
        raise RiskAllocationError(f"{path} must be a bounded identifier")
    return value


def _validate_digest(value: object, path: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise RiskAllocationError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _validate_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RiskAllocationError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RiskAllocationError(f"{path} must be finite")
    return result


def _require_object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RiskAllocationError(f"{path} must be an object")
    return value


def _require_keys(value: dict[str, object], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise RiskAllocationError(f"{path} has missing or unexpected fields")


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise RiskAllocationError("generated_at must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        raise RiskAllocationError("generated_at must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RiskAllocationError("generated_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RiskAllocationError("generated_at must include a timezone")
    return parsed.astimezone(UTC)


def parse_artifact(value: object) -> RiskAllocationArtifact:
    root = _require_object(value, "artifact")
    _require_keys(
        root,
        {
            "generated_at",
            "portfolio_id",
            "risk_model_digest",
            "covariance_snapshot_digest",
            "positions",
            "covariance",
        },
        "artifact",
    )
    raw_positions = root["positions"]
    if (
        not isinstance(raw_positions, list)
        or len(raw_positions) < 2
        or len(raw_positions) > MAX_ASSETS
    ):
        raise RiskAllocationError("positions must contain between 2 and 256 assets")
    positions: list[Position] = []
    seen_assets: set[str] = set()
    for index, value_row in enumerate(raw_positions):
        path = f"positions[{index}]"
        row = _require_object(value_row, path)
        _require_keys(row, {"asset_id", "weight"}, path)
        asset_id = _validate_identifier(row["asset_id"], f"{path}.asset_id")
        if asset_id in seen_assets:
            raise RiskAllocationError("position asset IDs must be unique")
        seen_assets.add(asset_id)
        positions.append(
            Position(asset_id=asset_id, weight=_validate_float(row["weight"], f"{path}.weight"))
        )

    size = len(positions)
    raw_covariance = root["covariance"]
    if not isinstance(raw_covariance, list) or len(raw_covariance) != size:
        raise RiskAllocationError("covariance must be a square matrix matching positions")
    covariance: list[tuple[float, ...]] = []
    for row_index, raw_row in enumerate(raw_covariance):
        if not isinstance(raw_row, list) or len(raw_row) != size:
            raise RiskAllocationError("covariance must be a square matrix matching positions")
        covariance.append(
            tuple(
                _validate_float(item, f"covariance[{row_index}][{column_index}]")
                for column_index, item in enumerate(raw_row)
            )
        )
    return RiskAllocationArtifact(
        generated_at=_parse_timestamp(root["generated_at"]),
        portfolio_id=_validate_identifier(root["portfolio_id"], "portfolio_id"),
        risk_model_digest=_validate_digest(root["risk_model_digest"], "risk_model_digest"),
        covariance_snapshot_digest=_validate_digest(
            root["covariance_snapshot_digest"], "covariance_snapshot_digest"
        ),
        positions=tuple(positions),
        covariance=tuple(covariance),
    )


def _canonical_artifact(artifact: RiskAllocationArtifact) -> dict[str, Any]:
    order = sorted(
        range(len(artifact.positions)), key=lambda index: artifact.positions[index].asset_id
    )
    covariance = np.asarray(artifact.covariance, dtype=float)
    return {
        "generated_at": _format_timestamp(artifact.generated_at),
        "portfolio_id": artifact.portfolio_id,
        "risk_model_digest": artifact.risk_model_digest,
        "covariance_snapshot_digest": artifact.covariance_snapshot_digest,
        "positions": [asdict(artifact.positions[index]) for index in order],
        "covariance": [[float(covariance[i, j]) for j in order] for i in order],
    }


def _portfolio_volatility(weights: np.ndarray, covariance: np.ndarray) -> float:
    variance = float(weights @ covariance @ weights)
    if not math.isfinite(variance) or variance <= 0.0:
        raise RiskAllocationError("portfolio variance must be finite and positive")
    return math.sqrt(variance)


def _finite_difference_marginals(
    weights: np.ndarray,
    covariance: np.ndarray,
    step: float,
) -> np.ndarray:
    marginal = np.empty_like(weights)
    for index, weight in enumerate(weights):
        bump = step * max(1.0, abs(float(weight)))
        upper = weights.copy()
        lower = weights.copy()
        upper[index] += bump
        lower[index] -= bump
        marginal[index] = (
            _portfolio_volatility(upper, covariance) - _portfolio_volatility(lower, covariance)
        ) / (2.0 * bump)
    return marginal


def audit_risk_allocation(
    artifact: RiskAllocationArtifact | object,
    *,
    now: datetime,
    policy: AllocationPolicy | None = None,
) -> RiskAllocationReport:
    parsed = artifact if isinstance(artifact, RiskAllocationArtifact) else parse_artifact(artifact)
    if now.tzinfo is None or now.utcoffset() is None:
        raise RiskAllocationError("now must be timezone-aware")
    effective_policy = policy or AllocationPolicy()
    order = sorted(range(len(parsed.positions)), key=lambda index: parsed.positions[index].asset_id)
    positions = tuple(parsed.positions[index] for index in order)
    raw_covariance = np.asarray(parsed.covariance, dtype=float)
    covariance = raw_covariance[np.ix_(order, order)]
    weights = np.asarray([position.weight for position in positions], dtype=float)
    symmetry_error = float(np.max(np.abs(covariance - covariance.T)))
    if symmetry_error > effective_policy.symmetry_tolerance:
        raise RiskAllocationError("covariance exceeds the symmetry tolerance")

    volatility = _portfolio_volatility(weights, covariance)
    analytic_marginal = covariance @ weights / volatility
    finite_difference = _finite_difference_marginals(
        weights, covariance, effective_policy.finite_difference_step
    )
    component = weights * analytic_marginal
    component_sum = float(component.sum())
    euler_relative_error = abs(component_sum - volatility) / max(volatility, 1e-15)
    gradient_errors = np.abs(finite_difference - analytic_marginal) / np.maximum(
        np.abs(analytic_marginal), 1e-12
    )

    absolute_components = np.abs(component)
    absolute_total = float(absolute_components.sum())
    if absolute_total <= 0.0:
        raise RiskAllocationError("absolute component risk must be positive")
    absolute_shares = absolute_components / absolute_total
    signed_shares = component / volatility
    risk_budget_hhi = float(absolute_shares @ absolute_shares)
    effective_positions = 1.0 / risk_budget_hhi
    gross_leverage = float(np.abs(weights).sum())
    net_exposure = float(weights.sum())
    max_absolute_share = float(absolute_shares.max())
    max_gradient_error = float(gradient_errors.max())

    reasons: set[str] = set()
    age_seconds = (now.astimezone(UTC) - parsed.generated_at).total_seconds()
    if age_seconds > effective_policy.max_age_seconds:
        reasons.add("stale_allocation_evidence")
    if age_seconds < -effective_policy.max_future_skew_seconds:
        reasons.add("future_generated_at")
    if gross_leverage > effective_policy.max_gross_leverage:
        reasons.add("gross_leverage_budget_exceeded")
    if (
        abs(net_exposure - effective_policy.net_exposure_target)
        > effective_policy.net_exposure_tolerance
    ):
        reasons.add("net_exposure_mismatch")
    if max_absolute_share > effective_policy.max_absolute_risk_share:
        reasons.add("component_risk_concentration_exceeded")
    if effective_positions < effective_policy.min_effective_risk_positions:
        reasons.add("insufficient_effective_risk_positions")
    if euler_relative_error > effective_policy.max_euler_relative_error:
        reasons.add("euler_reconciliation_failed")
    if max_gradient_error > effective_policy.max_gradient_relative_error:
        reasons.add("marginal_gradient_check_failed")

    allocations = tuple(
        sorted(
            (
                AssetRiskAllocation(
                    asset_ref=hashlib.sha256(position.asset_id.encode("utf-8")).hexdigest()[:16],
                    marginal_volatility=float(analytic_marginal[index]),
                    component_volatility=float(component[index]),
                    signed_risk_share=float(signed_shares[index]),
                    absolute_risk_budget_share=float(absolute_shares[index]),
                    finite_difference_marginal=float(finite_difference[index]),
                    gradient_relative_error=float(gradient_errors[index]),
                )
                for index, position in enumerate(positions)
            ),
            key=lambda allocation: allocation.asset_ref,
        )
    )
    ordered_reasons = tuple(sorted(reasons))
    return RiskAllocationReport(
        schema_version="1.0",
        accepted=not ordered_reasons,
        reasons=ordered_reasons,
        evidence_digest=_digest(_canonical_artifact(parsed)),
        portfolio_ref=hashlib.sha256(parsed.portfolio_id.encode("utf-8")).hexdigest()[:16],
        risk_model_digest=parsed.risk_model_digest,
        covariance_snapshot_digest=parsed.covariance_snapshot_digest,
        generated_at=_format_timestamp(parsed.generated_at),
        asset_count=len(parsed.positions),
        net_exposure=net_exposure,
        gross_leverage=gross_leverage,
        portfolio_volatility=volatility,
        component_sum=component_sum,
        euler_relative_error=euler_relative_error,
        max_gradient_relative_error=max_gradient_error,
        max_absolute_risk_share=max_absolute_share,
        risk_budget_hhi=risk_budget_hhi,
        effective_risk_positions=effective_positions,
        negative_component_count=int(np.count_nonzero(component < 0.0)),
        allocations=allocations,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RiskAllocationError("JSON objects must not contain duplicate fields")
        result[key] = value
    return result


def load_artifact(path: Path) -> RiskAllocationArtifact:
    try:
        if path.stat().st_size > MAX_INPUT_BYTES:
            raise RiskAllocationError("input exceeds the JSON byte limit")
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RiskAllocationError("JSON must not contain non-finite numbers")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RiskAllocationError("input is not valid UTF-8 JSON") from error
    return parse_artifact(payload)


def _write_json(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    if output is None:
        print(rendered, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Euler component-volatility allocation")
    parser.add_argument("artifact", type=Path, help="portfolio allocation JSON artifact")
    parser.add_argument("--output", type=Path, help="atomically write the audit report")
    parser.add_argument("--now", required=True, help="UTC-aware audit timestamp")
    parser.add_argument("--max-gross-leverage", type=float, default=2.0)
    parser.add_argument("--max-absolute-risk-share", type=float, default=0.55)
    parser.add_argument("--min-effective-risk-positions", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report = audit_risk_allocation(
            load_artifact(arguments.artifact),
            now=_parse_timestamp(arguments.now),
            policy=AllocationPolicy(
                max_gross_leverage=arguments.max_gross_leverage,
                max_absolute_risk_share=arguments.max_absolute_risk_share,
                min_effective_risk_positions=arguments.min_effective_risk_positions,
            ),
        )
        _write_json(report.to_dict(), arguments.output)
    except (RiskAllocationError, OSError) as error:
        _write_json(
            {"accepted": False, "error": "malformed_artifact", "detail": str(error)},
            arguments.output,
        )
        return 2
    return 0 if report.accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
