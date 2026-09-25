"""Fail-closed covariance admission and bounded positive-definite repair."""

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
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"
MAX_INPUT_BYTES = 2 * 1024 * 1024
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class CovarianceEvidenceError(ValueError):
    """Raised when a covariance artifact is structurally unsafe to process."""


@dataclass(frozen=True)
class CovariancePolicy:
    policy_id: str = "covariance-admission-v1"
    min_assets: int = 2
    max_assets: int = 128
    min_observations: int = 30
    min_observation_asset_ratio: float = 5.0
    minimum_variance: float = 1e-12
    maximum_covariance_magnitude: float = 1.0
    max_asymmetry: float = 1e-10
    max_absolute_correlation: float = 1.000000001
    eigenvalue_floor_ratio: float = 1e-8
    max_relative_repair: float = 0.05
    max_correlation_shift: float = 0.05
    max_repaired_condition_number: float = 1e9

    def validate(self) -> None:
        _identifier(self.policy_id, "policy_id")
        for field in ("min_assets", "max_assets", "min_observations"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CovarianceEvidenceError(f"invalid policy field: {field}")
        if self.min_assets > self.max_assets:
            raise CovarianceEvidenceError("min_assets exceeds max_assets")
        for field in (
            "min_observation_asset_ratio",
            "minimum_variance",
            "maximum_covariance_magnitude",
            "max_asymmetry",
            "max_absolute_correlation",
            "eigenvalue_floor_ratio",
            "max_relative_repair",
            "max_correlation_shift",
            "max_repaired_condition_number",
        ):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise CovarianceEvidenceError(f"invalid policy field: {field}")
            if value < 0:
                raise CovarianceEvidenceError(f"negative policy field: {field}")
        if self.max_absolute_correlation < 1:
            raise CovarianceEvidenceError(
                "max_absolute_correlation must be at least one"
            )
        if self.max_repaired_condition_number < 1:
            raise CovarianceEvidenceError("condition-number limit must be at least one")
        if self.maximum_covariance_magnitude < self.minimum_variance:
            raise CovarianceEvidenceError(
                "covariance magnitude limit is below minimum variance"
            )


@dataclass(frozen=True)
class CovarianceReport:
    schema_version: str
    accepted: bool
    repaired: bool
    reason_codes: tuple[str, ...]
    policy: dict[str, Any]
    estimator_id: str
    asset_count: int
    observation_count: int
    observation_asset_ratio: float
    maximum_asymmetry: float
    input_minimum_eigenvalue: float
    repaired_minimum_eigenvalue: float
    input_condition_number: float | None
    repaired_condition_number: float
    relative_repair_norm: float
    maximum_correlation_shift: float
    input_sha256: str
    repaired_covariance_sha256: str
    report_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CovarianceAdmission:
    report: CovarianceReport
    covariance: np.ndarray


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise CovarianceEvidenceError(f"invalid identifier: {field}")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _matrix_digest(matrix: np.ndarray) -> str:
    normalized = [[float(value) for value in row] for row in matrix]
    return _digest(normalized)


def _condition_number(eigenvalues: np.ndarray) -> float | None:
    maximum = float(np.max(eigenvalues))
    minimum = float(np.min(eigenvalues))
    if minimum <= 0:
        return None
    return maximum / minimum


def admit_covariance(
    covariance: Any,
    *,
    asset_ids: list[str] | tuple[str, ...],
    observation_count: int,
    estimator_id: str,
    policy: CovariancePolicy | None = None,
) -> CovarianceAdmission:
    """Validate and condition a covariance matrix without changing its variances."""

    active_policy = policy or CovariancePolicy()
    active_policy.validate()
    estimator = _identifier(estimator_id, "estimator_id")
    if isinstance(observation_count, bool) or not isinstance(observation_count, int):
        raise CovarianceEvidenceError("observation_count must be an integer")
    if observation_count <= 0:
        raise CovarianceEvidenceError("observation_count must be positive")
    if not isinstance(asset_ids, (list, tuple)):
        raise CovarianceEvidenceError("asset_ids must be a list")
    labels = tuple(_identifier(value, "asset_id") for value in asset_ids)
    if len(set(labels)) != len(labels):
        raise CovarianceEvidenceError("asset_ids must be unique")
    if not active_policy.min_assets <= len(labels) <= active_policy.max_assets:
        raise CovarianceEvidenceError("asset count violates processing bounds")
    try:
        matrix = np.asarray(covariance, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CovarianceEvidenceError("covariance must be numeric") from exc
    if matrix.ndim != 2 or matrix.shape != (len(labels), len(labels)):
        raise CovarianceEvidenceError("covariance shape does not match asset_ids")
    if not np.isfinite(matrix).all():
        raise CovarianceEvidenceError("covariance contains non-finite values")
    if np.max(np.abs(matrix)) > active_policy.maximum_covariance_magnitude:
        raise CovarianceEvidenceError("covariance exceeds safe magnitude bound")
    diagonal = np.diag(matrix)
    if np.any(diagonal < active_policy.minimum_variance):
        raise CovarianceEvidenceError("covariance diagonal contains invalid variance")

    maximum_asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    symmetric = (matrix + matrix.T) / 2.0
    standard_deviations = np.sqrt(np.diag(symmetric))
    scale = np.outer(standard_deviations, standard_deviations)
    input_correlation = symmetric / scale
    input_max_correlation = float(
        np.max(np.abs(input_correlation - np.eye(len(labels))))
    )
    input_eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    maximum_eigenvalue = float(np.max(input_eigenvalues))
    if maximum_eigenvalue <= 0:
        raise CovarianceEvidenceError("covariance has no positive variance direction")

    floor = max(
        active_policy.minimum_variance,
        maximum_eigenvalue * active_policy.eigenvalue_floor_ratio,
    )
    needs_eigenvalue_floor = bool(np.any(input_eigenvalues < floor))
    if needs_eigenvalue_floor:
        repaired_eigenvalues = np.maximum(input_eigenvalues, floor)
        repaired = (eigenvectors * repaired_eigenvalues) @ eigenvectors.T
        variance_rescale = np.sqrt(np.diag(symmetric) / np.diag(repaired))
        repaired = repaired * np.outer(variance_rescale, variance_rescale)
        repaired = (repaired + repaired.T) / 2.0
    else:
        repaired = symmetric.copy()
    repaired_eigenvalues_final = np.linalg.eigvalsh(repaired)
    repaired_correlation = repaired / scale
    relative_repair = float(
        np.linalg.norm(repaired - symmetric, ord="fro")
        / np.linalg.norm(symmetric, ord="fro")
    )
    maximum_correlation_shift = float(
        np.max(np.abs(repaired_correlation - input_correlation))
    )
    repaired_condition = _condition_number(repaired_eigenvalues_final)
    if repaired_condition is None:
        raise CovarianceEvidenceError(
            "repair did not produce positive-definite covariance"
        )

    ratio = observation_count / len(labels)
    reasons: list[str] = []
    if observation_count < active_policy.min_observations:
        reasons.append("INSUFFICIENT_OBSERVATIONS")
    if ratio < active_policy.min_observation_asset_ratio:
        reasons.append("INSUFFICIENT_OBSERVATION_ASSET_RATIO")
    if maximum_asymmetry > active_policy.max_asymmetry:
        reasons.append("ASYMMETRY_EXCEEDED")
    if input_max_correlation > active_policy.max_absolute_correlation:
        reasons.append("CORRELATION_BOUND_EXCEEDED")
    if relative_repair > active_policy.max_relative_repair:
        reasons.append("REPAIR_NORM_EXCEEDED")
    if maximum_correlation_shift > active_policy.max_correlation_shift:
        reasons.append("CORRELATION_SHIFT_EXCEEDED")
    if repaired_condition > active_policy.max_repaired_condition_number:
        reasons.append("CONDITION_NUMBER_EXCEEDED")

    was_repaired = needs_eigenvalue_floor or maximum_asymmetry > 0
    input_payload = {
        "schema_version": SCHEMA_VERSION,
        "estimator_id": estimator,
        "asset_ids": list(labels),
        "observation_count": observation_count,
        "covariance": [[float(value) for value in row] for row in matrix],
    }
    core = {
        "schema_version": SCHEMA_VERSION,
        "accepted": not reasons,
        "repaired": was_repaired,
        "reason_codes": tuple(reasons),
        "policy": asdict(active_policy),
        "estimator_id": estimator,
        "asset_count": len(labels),
        "observation_count": observation_count,
        "observation_asset_ratio": ratio,
        "maximum_asymmetry": maximum_asymmetry,
        "input_minimum_eigenvalue": float(np.min(input_eigenvalues)),
        "repaired_minimum_eigenvalue": float(np.min(repaired_eigenvalues_final)),
        "input_condition_number": _condition_number(input_eigenvalues),
        "repaired_condition_number": repaired_condition,
        "relative_repair_norm": relative_repair,
        "maximum_correlation_shift": maximum_correlation_shift,
        "input_sha256": _digest(input_payload),
        "repaired_covariance_sha256": _matrix_digest(repaired),
    }
    report = CovarianceReport(**core, report_sha256=_digest(core))
    return CovarianceAdmission(report=report, covariance=repaired)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CovarianceEvidenceError("duplicate JSON key")
        result[key] = value
    return result


def load_artifact(path: Path, *, max_bytes: int = MAX_INPUT_BYTES) -> dict[str, Any]:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise CovarianceEvidenceError("max_bytes must be positive")
    if path.stat().st_size > max_bytes:
        raise CovarianceEvidenceError("artifact exceeds byte budget")
    with path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise CovarianceEvidenceError("artifact exceeds byte budget")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CovarianceEvidenceError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CovarianceEvidenceError("invalid JSON artifact") from exc


def audit_artifact(
    artifact: dict[str, Any], policy: CovariancePolicy | None = None
) -> CovarianceAdmission:
    expected = {
        "schema_version",
        "estimator_id",
        "asset_ids",
        "observation_count",
        "covariance",
    }
    if not isinstance(artifact, dict) or set(artifact) != expected:
        raise CovarianceEvidenceError("artifact has invalid object shape")
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise CovarianceEvidenceError("unsupported schema_version")
    return admit_covariance(
        artifact["covariance"],
        asset_ids=artifact["asset_ids"],
        observation_count=artifact["observation_count"],
        estimator_id=artifact["estimator_id"],
        policy=policy,
    )


def _write_report(path: Path, report: CovarianceReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report.as_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        admission = audit_artifact(load_artifact(args.artifact))
        if args.output:
            _write_report(args.output, admission.report)
    except (CovarianceEvidenceError, OSError):
        print(json.dumps({"accepted": False, "error_code": "MALFORMED_ARTIFACT"}))
        return 3
    print(json.dumps(admission.report.as_dict(), sort_keys=True, allow_nan=False))
    return 0 if admission.report.accepted else 2


if __name__ == "__main__":
    sys.exit(main())
