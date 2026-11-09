from __future__ import annotations

import json

import numpy as np
import pytest

from src.expected_shortfall_bootstrap import (
    BootstrapPolicy,
    ExpectedShortfallBootstrapError,
    audit_expected_shortfall_bootstrap,
)


def dependent_scenarios(rows: int = 500) -> np.ndarray:
    generator = np.random.default_rng(2026)
    shocks = generator.normal(0.0, 0.007, size=(rows, 3))
    result = np.empty_like(shocks)
    result[0] = shocks[0]
    for index in range(1, rows):
        result[index] = 0.55 * result[index - 1] + shocks[index]
    result[220:235] -= np.asarray((0.035, 0.025, 0.012))
    return result


def audit(matrix: np.ndarray | None = None, **overrides):
    arguments = {
        "asset_ids": ("equity-secret", "credit-secret", "rates-secret"),
        "weights": (0.50, 0.35, 0.15),
        "scenario_returns": dependent_scenarios() if matrix is None else matrix,
        "confidence": 0.95,
        "min_effective_tail_scenarios": 10,
        "policy": BootstrapPolicy(
            block_size=10,
            replicates=79,
            max_component_interval_width_as_es=5.0,
        ),
    }
    arguments.update(overrides)
    return audit_expected_shortfall_bootstrap(**arguments)


def finding_codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


def test_dependent_tail_allocation_has_deterministic_intervals() -> None:
    first = audit()
    second = audit()

    assert first.to_dict() == second.to_dict()
    assert first.accepted is True
    assert first.replicates_valid == 79
    assert first.lower_expected_shortfall < first.point_expected_shortfall
    assert first.upper_expected_shortfall > first.lower_expected_shortfall
    assert all(interval.sign_stable for interval in first.intervals)


def test_asset_permutation_preserves_report_and_input_commitment() -> None:
    matrix = dependent_scenarios()
    original = audit(matrix)
    order = [2, 0, 1]
    permuted = audit_expected_shortfall_bootstrap(
        asset_ids=tuple(
            np.asarray(("equity-secret", "credit-secret", "rates-secret"))[order]
        ),
        weights=np.asarray((0.50, 0.35, 0.15))[order],
        scenario_returns=matrix[:, order],
        confidence=0.95,
        min_effective_tail_scenarios=10,
        policy=BootstrapPolicy(
            block_size=10,
            replicates=79,
            max_component_interval_width_as_es=5.0,
        ),
    )

    assert permuted.to_dict() == original.to_dict()


def test_sequence_order_is_bound_because_blocks_preserve_dependence() -> None:
    matrix = dependent_scenarios()
    shuffled = matrix[np.random.default_rng(7).permutation(len(matrix))]

    assert audit(matrix).input_sha256 != audit(shuffled).input_sha256


def test_tight_width_budget_rejects_uncertain_component_allocation() -> None:
    report = audit(
        policy=BootstrapPolicy(
            block_size=10,
            replicates=79,
            max_component_interval_width_as_es=0.01,
            require_sign_stability=False,
        )
    )

    assert report.accepted is False
    assert "COMPONENT_INTERVAL_TOO_WIDE" in finding_codes(report)
    finding = next(
        item for item in report.findings if item.code == "COMPONENT_INTERVAL_TOO_WIDE"
    )
    assert finding.observed > finding.limit


def test_zero_weight_component_fails_sign_stability_gate() -> None:
    report = audit(weights=(0.65, 0.35, 0.0))

    assert report.accepted is False
    assert "COMPONENT_SIGN_UNSTABLE" in finding_codes(report)


def test_report_hashes_asset_ids_and_omits_scenario_rows() -> None:
    report = audit()
    rendered = json.dumps(report.to_dict())

    assert "secret" not in rendered
    assert "scenario_returns" not in rendered
    assert len(report.input_sha256) == 64
    assert len(report.policy_sha256) == 64


def test_policy_digest_changes_with_prespecified_seed() -> None:
    first = audit()
    second = audit(
        policy=BootstrapPolicy(
            block_size=10,
            replicates=79,
            seed=1,
            max_component_interval_width_as_es=5.0,
        )
    )

    assert first.policy_sha256 != second.policy_sha256
    assert first.input_sha256 == second.input_sha256


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (BootstrapPolicy(block_size=1, replicates=50), "two blocks"),
        (BootstrapPolicy(block_size=251, replicates=50), "two blocks"),
        (BootstrapPolicy(replicates=49), "replicates"),
        (BootstrapPolicy(interval_confidence=1.0), "interval_confidence"),
        (
            BootstrapPolicy(max_component_interval_width_as_es=0.0),
            "interval_width",
        ),
        (BootstrapPolicy(require_sign_stability="yes"), "boolean"),
        (BootstrapPolicy(seed=-1), "seed"),
        (BootstrapPolicy(max_resampled_rows=1), "max_resampled_rows"),
    ],
)
def test_invalid_or_excessive_policy_fails_closed(policy, message) -> None:
    with pytest.raises(ExpectedShortfallBootstrapError, match=message):
        audit(policy=policy)


def test_invalid_allocation_input_is_sanitized_to_bootstrap_error() -> None:
    with pytest.raises(ExpectedShortfallBootstrapError, match="finite"):
        audit(matrix=np.full((500, 3), np.nan))
