from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.expected_shortfall_allocation import (
    ExpectedShortfallAllocationError,
    allocate_historical_expected_shortfall,
)


class ExpectedShortfallBootstrapError(ValueError):
    """Raised when bootstrap evidence or policy cannot be evaluated safely."""


@dataclass(frozen=True)
class BootstrapPolicy:
    block_size: int = 20
    replicates: int = 500
    interval_confidence: float = 0.95
    max_component_interval_width_as_es: float = 1.0
    require_sign_stability: bool = True
    seed: int = 0
    max_resampled_rows: int = 20_000_000

    def validate(self, scenario_count: int) -> None:
        if (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or not 2 <= self.block_size <= scenario_count // 2
        ):
            raise ExpectedShortfallBootstrapError(
                "block_size must retain at least two blocks"
            )
        if (
            isinstance(self.replicates, bool)
            or not isinstance(self.replicates, int)
            or not 50 <= self.replicates <= 10_000
        ):
            raise ExpectedShortfallBootstrapError(
                "replicates must be an integer in [50, 10000]"
            )
        if (
            isinstance(self.interval_confidence, bool)
            or not isinstance(self.interval_confidence, (int, float))
            or not math.isfinite(float(self.interval_confidence))
            or not 0.5 <= float(self.interval_confidence) < 1.0
        ):
            raise ExpectedShortfallBootstrapError(
                "interval_confidence must be finite and in [0.5, 1)"
            )
        if (
            isinstance(self.max_component_interval_width_as_es, bool)
            or not isinstance(self.max_component_interval_width_as_es, (int, float))
            or not math.isfinite(float(self.max_component_interval_width_as_es))
            or float(self.max_component_interval_width_as_es) <= 0.0
        ):
            raise ExpectedShortfallBootstrapError(
                "max_component_interval_width_as_es must be finite and positive"
            )
        if not isinstance(self.require_sign_stability, bool):
            raise ExpectedShortfallBootstrapError(
                "require_sign_stability must be boolean"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 2**32 - 1
        ):
            raise ExpectedShortfallBootstrapError(
                "seed must be an integer in [0, 2**32 - 1]"
            )
        if (
            isinstance(self.max_resampled_rows, bool)
            or not isinstance(self.max_resampled_rows, int)
            or self.max_resampled_rows < 1
        ):
            raise ExpectedShortfallBootstrapError(
                "max_resampled_rows must be a positive integer"
            )
        if scenario_count * self.replicates > self.max_resampled_rows:
            raise ExpectedShortfallBootstrapError(
                "bootstrap work exceeds max_resampled_rows"
            )


@dataclass(frozen=True)
class BootstrapFinding:
    code: str
    asset_ref: str | None = None
    observed: float | int | None = None
    limit: float | int | None = None


@dataclass(frozen=True)
class AssetBootstrapInterval:
    asset_ref: str
    point_component_expected_shortfall: float
    lower_component_expected_shortfall: float
    upper_component_expected_shortfall: float
    interval_width_as_portfolio_es: float
    sign_stable: bool


@dataclass(frozen=True)
class ExpectedShortfallBootstrapReport:
    schema_version: str
    accepted: bool
    scenario_count: int
    confidence: float
    block_size: int
    replicates_requested: int
    replicates_valid: int
    interval_confidence: float
    point_expected_shortfall: float
    lower_expected_shortfall: float
    upper_expected_shortfall: float
    input_sha256: str
    policy_sha256: str
    intervals: tuple[AssetBootstrapInterval, ...]
    findings: tuple[BootstrapFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "intervals": [asdict(interval) for interval in self.intervals],
            "findings": [asdict(finding) for finding in self.findings],
        }


def _digest_payload(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _input_digest(
    *,
    asset_ids: tuple[str, ...],
    weights: np.ndarray,
    scenarios: np.ndarray,
    confidence: float,
    min_effective_tail_scenarios: float,
) -> str:
    order = np.argsort(np.asarray(asset_ids), kind="stable")
    digest = hashlib.sha256()
    digest.update(b"expected-shortfall-bootstrap-input/v1\0")
    digest.update(
        json.dumps(
            {
                "asset_ids": [asset_ids[index] for index in order],
                "confidence": confidence,
                "min_effective_tail_scenarios": min_effective_tail_scenarios,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(np.asarray(weights[order], dtype="<f8").tobytes())
    digest.update(np.asarray(scenarios[:, order], dtype="<f8").tobytes())
    return digest.hexdigest()


def audit_expected_shortfall_bootstrap(
    *,
    asset_ids: Sequence[str],
    weights: object,
    scenario_returns: object,
    confidence: float = 0.975,
    min_effective_tail_scenarios: float = 20.0,
    policy: BootstrapPolicy | None = None,
) -> ExpectedShortfallBootstrapReport:
    """Audit component ES uncertainty with a circular moving-block bootstrap."""

    active_policy = policy or BootstrapPolicy()
    try:
        point = allocate_historical_expected_shortfall(
            asset_ids=asset_ids,
            weights=weights,
            scenario_returns=scenario_returns,
            confidence=confidence,
            min_effective_tail_scenarios=min_effective_tail_scenarios,
        )
    except ExpectedShortfallAllocationError as error:
        raise ExpectedShortfallBootstrapError(str(error)) from error

    assets = tuple(asset_ids)
    position_weights = np.asarray(weights, dtype=float)
    scenarios = np.asarray(scenario_returns, dtype=float)
    active_policy.validate(point.scenario_count)
    blocks_per_replicate = math.ceil(point.scenario_count / active_policy.block_size)
    offsets = np.arange(active_policy.block_size)
    generator = np.random.default_rng(active_policy.seed)
    component_samples: dict[str, list[float]] = {
        allocation.asset_ref: [] for allocation in point.allocations
    }
    expected_shortfall_samples: list[float] = []
    invalid_replicates = 0

    for _ in range(active_policy.replicates):
        starts = generator.integers(0, point.scenario_count, size=blocks_per_replicate)
        indices = ((starts[:, None] + offsets) % point.scenario_count).ravel()[
            : point.scenario_count
        ]
        try:
            replicate = allocate_historical_expected_shortfall(
                asset_ids=assets,
                weights=position_weights,
                scenario_returns=scenarios[indices],
                confidence=confidence,
                min_effective_tail_scenarios=min_effective_tail_scenarios,
            )
        except ExpectedShortfallAllocationError:
            invalid_replicates += 1
            continue
        expected_shortfall_samples.append(replicate.expected_shortfall)
        for allocation in replicate.allocations:
            component_samples[allocation.asset_ref].append(
                allocation.component_expected_shortfall
            )

    valid_replicates = len(expected_shortfall_samples)
    if valid_replicates < max(2, active_policy.replicates // 2):
        raise ExpectedShortfallBootstrapError(
            "fewer than half of bootstrap replicates produced allocatable tail loss"
        )
    alpha = (1.0 - active_policy.interval_confidence) / 2.0
    lower_quantile, upper_quantile = alpha, 1.0 - alpha
    lower_es, upper_es = np.quantile(
        expected_shortfall_samples, (lower_quantile, upper_quantile)
    )
    findings: list[BootstrapFinding] = []
    if invalid_replicates:
        findings.append(
            BootstrapFinding(
                code="INVALID_BOOTSTRAP_REPLICATES",
                observed=invalid_replicates,
                limit=0,
            )
        )

    point_by_ref = {
        allocation.asset_ref: allocation for allocation in point.allocations
    }
    intervals: list[AssetBootstrapInterval] = []
    for asset_ref in sorted(component_samples):
        lower, upper = np.quantile(
            component_samples[asset_ref], (lower_quantile, upper_quantile)
        )
        width = float((upper - lower) / point.expected_shortfall)
        sign_stable = bool(lower > 0.0 or upper < 0.0)
        intervals.append(
            AssetBootstrapInterval(
                asset_ref=asset_ref,
                point_component_expected_shortfall=point_by_ref[
                    asset_ref
                ].component_expected_shortfall,
                lower_component_expected_shortfall=float(lower),
                upper_component_expected_shortfall=float(upper),
                interval_width_as_portfolio_es=width,
                sign_stable=sign_stable,
            )
        )
        if width > active_policy.max_component_interval_width_as_es:
            findings.append(
                BootstrapFinding(
                    code="COMPONENT_INTERVAL_TOO_WIDE",
                    asset_ref=asset_ref,
                    observed=width,
                    limit=active_policy.max_component_interval_width_as_es,
                )
            )
        if active_policy.require_sign_stability and not sign_stable:
            findings.append(
                BootstrapFinding(code="COMPONENT_SIGN_UNSTABLE", asset_ref=asset_ref)
            )

    findings.sort(key=lambda finding: (finding.code, finding.asset_ref or ""))
    return ExpectedShortfallBootstrapReport(
        schema_version="1.0",
        accepted=not findings,
        scenario_count=point.scenario_count,
        confidence=point.confidence,
        block_size=active_policy.block_size,
        replicates_requested=active_policy.replicates,
        replicates_valid=valid_replicates,
        interval_confidence=active_policy.interval_confidence,
        point_expected_shortfall=point.expected_shortfall,
        lower_expected_shortfall=float(lower_es),
        upper_expected_shortfall=float(upper_es),
        input_sha256=_input_digest(
            asset_ids=assets,
            weights=position_weights,
            scenarios=scenarios,
            confidence=point.confidence,
            min_effective_tail_scenarios=float(min_effective_tail_scenarios),
        ),
        policy_sha256=_digest_payload(asdict(active_policy)),
        intervals=tuple(intervals),
        findings=tuple(findings),
    )
