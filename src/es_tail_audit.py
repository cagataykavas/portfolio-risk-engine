from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class ArtifactMalformed(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ESTailPolicy:
    max_input_bytes: int = 8_388_608
    min_observations: int = 250
    max_observations: int = 100_000
    min_expected_tail_observations: float = 10.0
    min_actual_tail_observations: int = 8
    significance_level: float = 0.05
    min_mean_tail_buffer: float = 1e-6
    max_relative_tail_bias: float = 0.25
    min_material_relative_tail_bias: float = 0.10
    max_abs_value: float = 1_000_000.0
    max_hac_lag: int = 1_000
    max_age_seconds: int = 86_400
    max_future_skew_seconds: int = 60
    max_findings: int = 128

    def __post_init__(self) -> None:
        checks = (
            (1 <= self.max_input_bytes <= 67_108_864, "max_input_bytes"),
            (2 <= self.min_observations <= self.max_observations, "observation_range"),
            (self.max_observations <= 1_000_000, "max_observations"),
            (self.min_expected_tail_observations > 0.0, "min_expected_tail_observations"),
            (
                1 <= self.min_actual_tail_observations <= self.max_observations,
                "min_actual_tail_observations",
            ),
            (0.0 < self.significance_level < 0.5, "significance_level"),
            (self.min_mean_tail_buffer > 0.0, "min_mean_tail_buffer"),
            (self.max_relative_tail_bias >= 0.0, "max_relative_tail_bias"),
            (
                0.0 <= self.min_material_relative_tail_bias <= self.max_relative_tail_bias,
                "min_material_relative_tail_bias",
            ),
            (self.max_abs_value > 0.0, "max_abs_value"),
            (1 <= self.max_hac_lag <= 10_000, "max_hac_lag"),
            (1 <= self.max_age_seconds <= 604_800, "max_age_seconds"),
            (0 <= self.max_future_skew_seconds <= 3_600, "max_future_skew_seconds"),
            (1 <= self.max_findings <= 10_000, "max_findings"),
        )
        for valid, name in checks:
            if not valid:
                raise ValueError(f"{name} outside supported range")


@dataclass(frozen=True)
class Finding:
    scope: str
    subject_sha256: str
    code: str


@dataclass(frozen=True)
class ESTailReport:
    schema_version: int
    status: str
    accepted: bool
    artifact_sha256: str
    policy_sha256: str
    audit_sha256: str
    observation_count: int
    expected_tail_observations: float
    actual_tail_observations: int
    loss_above_expected_shortfall_count: int
    mean_identification_residual: float
    mean_tail_buffer: float
    relative_tail_bias: float
    hac_lag: int
    long_run_variance: float
    standard_error: float
    underestimation_z: float
    underestimation_p_value: float
    finding_count: int
    findings_truncated: bool
    findings: tuple[Finding, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["findings"] = [asdict(item) for item in self.findings]
        return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactMalformed("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ArtifactMalformed("NON_FINITE_NUMBER")


def load_artifact(raw: bytes, policy: ESTailPolicy | None = None) -> dict[str, Any]:
    selected_policy = policy or ESTailPolicy()
    if len(raw) > selected_policy.max_input_bytes:
        raise ArtifactMalformed("INPUT_TOO_LARGE")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactMalformed("INVALID_UTF8") from exc
    try:
        artifact = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ArtifactMalformed:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ArtifactMalformed("INVALID_JSON") from exc
    if not isinstance(artifact, dict):
        raise ArtifactMalformed("ROOT_NOT_OBJECT")
    return artifact


def audit_es_tail(
    artifact: dict[str, Any],
    *,
    as_of: datetime,
    policy: ESTailPolicy | None = None,
) -> ESTailReport:
    selected_policy = policy or ESTailPolicy()
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    as_of = as_of.astimezone(UTC)
    _expect_keys(
        artifact,
        {
            "schema_version",
            "audit_id",
            "model_sha256",
            "forecast_config_sha256",
            "created_at",
            "confidence",
            "forecast_horizon",
            "hac_lag",
            "observations",
        },
        "ROOT_FIELDS",
    )
    if artifact["schema_version"] != 1 or isinstance(artifact["schema_version"], bool):
        raise ArtifactMalformed("SCHEMA_VERSION")
    audit_id = _identifier(artifact["audit_id"], "AUDIT_ID")
    _digest(artifact["model_sha256"], "MODEL_SHA256")
    _digest(artifact["forecast_config_sha256"], "FORECAST_CONFIG_SHA256")
    created_at = _timestamp(artifact["created_at"], "CREATED_AT")
    if created_at > as_of + timedelta(seconds=selected_policy.max_future_skew_seconds):
        raise ArtifactMalformed("EVIDENCE_FROM_FUTURE")
    if as_of - created_at > timedelta(seconds=selected_policy.max_age_seconds):
        raise ArtifactMalformed("STALE_EVIDENCE")

    confidence = _number(artifact["confidence"], "CONFIDENCE", selected_policy)
    if not 0.5 < confidence < 1.0:
        raise ArtifactMalformed("CONFIDENCE")
    tail_probability = 1.0 - confidence
    forecast_horizon = _integer(artifact["forecast_horizon"], "FORECAST_HORIZON")
    if not 1 <= forecast_horizon <= 10_000:
        raise ArtifactMalformed("FORECAST_HORIZON")
    hac_lag = _integer(artifact["hac_lag"], "HAC_LAG")
    if not 0 <= hac_lag <= selected_policy.max_hac_lag:
        raise ArtifactMalformed("HAC_LAG")
    if hac_lag < forecast_horizon - 1:
        raise ArtifactMalformed("HAC_LAG_BELOW_HORIZON")

    raw_observations = artifact["observations"]
    if not isinstance(raw_observations, list) or not raw_observations:
        raise ArtifactMalformed("OBSERVATIONS_REQUIRED")
    if len(raw_observations) > selected_policy.max_observations:
        raise ArtifactMalformed("OBSERVATION_BUDGET_EXCEEDED")
    if hac_lag >= len(raw_observations):
        raise ArtifactMalformed("HAC_LAG_NOT_IDENTIFIABLE")

    timestamps: set[datetime] = set()
    previous_timestamp: datetime | None = None
    residuals: list[float] = []
    tail_buffers: list[float] = []
    canonical_observations: list[dict[str, Any]] = []
    tail_count = 0
    above_es_count = 0

    for raw_observation in raw_observations:
        observation = _parse_observation(raw_observation, selected_policy)
        timestamp = observation["timestamp"]
        if timestamp > created_at:
            raise ArtifactMalformed("OBSERVATION_AFTER_EVIDENCE")
        if timestamp in timestamps:
            raise ArtifactMalformed("DUPLICATE_TIMESTAMP")
        timestamps.add(timestamp)
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ArtifactMalformed("TIMESTAMPS_NOT_STRICTLY_ORDERED")
        previous_timestamp = timestamp

        loss = observation["loss"]
        value_at_risk = observation["value_at_risk"]
        expected_shortfall = observation["expected_shortfall"]
        tail_buffer = expected_shortfall - value_at_risk
        tail_excess = max(loss - value_at_risk, 0.0)
        residuals.append(tail_excess / tail_probability - tail_buffer)
        tail_buffers.append(tail_buffer)
        tail_count += loss > value_at_risk
        above_es_count += loss > expected_shortfall
        canonical_observations.append(
            {
                "timestamp": _canonical_timestamp(timestamp),
                "loss": loss,
                "value_at_risk": value_at_risk,
                "expected_shortfall": expected_shortfall,
            }
        )

    expected_tail_count = len(residuals) * tail_probability
    mean_residual = math.fsum(residuals) / len(residuals)
    mean_tail_buffer = math.fsum(tail_buffers) / len(tail_buffers)
    if mean_tail_buffer >= selected_policy.min_mean_tail_buffer:
        relative_tail_bias = mean_residual / mean_tail_buffer
        if not math.isfinite(relative_tail_bias):
            relative_tail_bias = math.copysign(selected_policy.max_abs_value, mean_residual)
    else:
        relative_tail_bias = math.copysign(selected_policy.max_abs_value, mean_residual)
    long_run_variance = _newey_west_long_run_variance(residuals, hac_lag)
    standard_error = math.sqrt(max(long_run_variance, 0.0) / len(residuals))
    if standard_error > 0.0:
        z_score = mean_residual / standard_error
        if not math.isfinite(z_score):
            z_score = math.copysign(selected_policy.max_abs_value, mean_residual)
        p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
    else:
        z_score = 0.0
        p_value = 1.0

    audit_hash = _sha256(audit_id.encode())
    findings: list[Finding] = []
    if len(residuals) < selected_policy.min_observations:
        findings.append(Finding("aggregate", audit_hash, "OBSERVATION_EVIDENCE_UNDERPOWERED"))
    if expected_tail_count < selected_policy.min_expected_tail_observations:
        findings.append(Finding("aggregate", audit_hash, "EXPECTED_TAIL_EVIDENCE_UNDERPOWERED"))
    if tail_count < selected_policy.min_actual_tail_observations:
        findings.append(Finding("aggregate", audit_hash, "ACTUAL_TAIL_EVIDENCE_UNDERPOWERED"))
    if mean_tail_buffer < selected_policy.min_mean_tail_buffer:
        findings.append(Finding("aggregate", audit_hash, "MEAN_TAIL_BUFFER_BELOW_POLICY"))
    if long_run_variance <= 0.0:
        findings.append(Finding("aggregate", audit_hash, "DEGENERATE_LONG_RUN_VARIANCE"))
    if relative_tail_bias > selected_policy.max_relative_tail_bias:
        findings.append(Finding("aggregate", audit_hash, "RELATIVE_TAIL_BIAS_ABOVE_POLICY"))
    if (
        relative_tail_bias > selected_policy.min_material_relative_tail_bias
        and p_value <= selected_policy.significance_level
    ):
        findings.append(Finding("aggregate", audit_hash, "TAIL_UNDERESTIMATION_SIGNIFICANT"))

    findings.sort(key=lambda item: (item.scope, item.subject_sha256, item.code))
    truncated = len(findings) > selected_policy.max_findings
    canonical_artifact = {
        "schema_version": 1,
        "audit_id": audit_id,
        "model_sha256": artifact["model_sha256"],
        "forecast_config_sha256": artifact["forecast_config_sha256"],
        "created_at": _canonical_timestamp(created_at),
        "confidence": confidence,
        "forecast_horizon": forecast_horizon,
        "hac_lag": hac_lag,
        "observations": canonical_observations,
    }
    accepted = not findings
    return ESTailReport(
        schema_version=1,
        status="accepted" if accepted else "policy_rejected",
        accepted=accepted,
        artifact_sha256=_sha256(_canonical_json(canonical_artifact)),
        policy_sha256=_sha256(_canonical_json(asdict(selected_policy))),
        audit_sha256=audit_hash,
        observation_count=len(residuals),
        expected_tail_observations=expected_tail_count,
        actual_tail_observations=tail_count,
        loss_above_expected_shortfall_count=above_es_count,
        mean_identification_residual=mean_residual,
        mean_tail_buffer=mean_tail_buffer,
        relative_tail_bias=relative_tail_bias,
        hac_lag=hac_lag,
        long_run_variance=long_run_variance,
        standard_error=standard_error,
        underestimation_z=z_score,
        underestimation_p_value=p_value,
        finding_count=len(findings),
        findings_truncated=truncated,
        findings=tuple(findings[: selected_policy.max_findings]),
    )


def _parse_observation(raw: Any, policy: ESTailPolicy) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ArtifactMalformed("OBSERVATION_NOT_OBJECT")
    _expect_keys(
        raw,
        {"timestamp", "loss", "value_at_risk", "expected_shortfall"},
        "OBSERVATION_FIELDS",
    )
    timestamp = _timestamp(raw["timestamp"], "TIMESTAMP")
    loss = _number(raw["loss"], "LOSS", policy)
    value_at_risk = _number(raw["value_at_risk"], "VALUE_AT_RISK", policy)
    expected_shortfall = _number(raw["expected_shortfall"], "EXPECTED_SHORTFALL", policy)
    if value_at_risk < 0.0:
        raise ArtifactMalformed("VALUE_AT_RISK_NEGATIVE")
    if expected_shortfall <= value_at_risk:
        raise ArtifactMalformed("EXPECTED_SHORTFALL_NOT_ABOVE_VAR")
    return {
        "timestamp": timestamp,
        "loss": loss,
        "value_at_risk": value_at_risk,
        "expected_shortfall": expected_shortfall,
    }


def _newey_west_long_run_variance(values: list[float], lag: int) -> float:
    mean_value = math.fsum(values) / len(values)
    centered = [value - mean_value for value in values]
    variance = math.fsum(value * value for value in centered) / len(centered)
    for offset in range(1, lag + 1):
        covariance = math.fsum(
            centered[index] * centered[index - offset] for index in range(offset, len(centered))
        ) / len(centered)
        variance += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    if variance < 0.0 and math.isclose(variance, 0.0, abs_tol=1e-15):
        return 0.0
    return variance


def _expect_keys(value: dict[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        raise ArtifactMalformed(code)


def _identifier(value: Any, code: str) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise ArtifactMalformed(code)
    return value


def _digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ArtifactMalformed(code)
    return value


def _integer(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactMalformed(code)
    return value


def _number(value: Any, code: str, policy: ESTailPolicy) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactMalformed(code)
    result = float(value)
    if not math.isfinite(result) or abs(result) > policy.max_abs_value:
        raise ArtifactMalformed(code)
    return result


def _timestamp(value: Any, code: str) -> datetime:
    if not isinstance(value, str):
        raise ArtifactMalformed(code)
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ArtifactMalformed(code) from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ArtifactMalformed(code)
    return timestamp.astimezone(UTC)


def _canonical_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Expected Shortfall tail calibration with a HAC identification-moment test"
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--as-of", help="timezone-aware RFC3339 timestamp; defaults to current UTC")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    policy = ESTailPolicy()
    try:
        artifact = load_artifact(args.artifact.read_bytes(), policy)
        as_of = _timestamp(args.as_of, "AS_OF") if args.as_of else datetime.now(UTC)
        report = audit_es_tail(artifact, as_of=as_of, policy=policy)
        payload = report.to_dict()
        exit_code = 0 if report.accepted else 2
    except (ArtifactMalformed, OSError) as exc:
        error = exc.code if isinstance(exc, ArtifactMalformed) else "ARTIFACT_IO_ERROR"
        payload = {"accepted": False, "error": error, "status": "malformed"}
        exit_code = 3
    if args.output:
        try:
            _write_json(args.output, payload)
        except OSError:
            return 3
    else:
        json.dump(payload, sys.stdout, sort_keys=True, separators=(",", ":"), allow_nan=False)
        sys.stdout.write("\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
