from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from src.risk_allocation_audit import (
    AllocationPolicy,
    RiskAllocationError,
    audit_risk_allocation,
    parse_artifact,
)
from src.synthetic import synthetic_returns

NOW = datetime(2026, 10, 1, 11, 0, tzinfo=UTC)
MODEL_DIGEST = "a" * 64
COVARIANCE_DIGEST = "b" * 64


def artifact(
    weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
    covariance: list[list[float]] | None = None,
) -> dict:
    if covariance is None:
        covariance = [
            [0.040, 0.004, 0.002, 0.001],
            [0.004, 0.030, 0.003, 0.001],
            [0.002, 0.003, 0.025, 0.002],
            [0.001, 0.001, 0.002, 0.020],
        ]
    return {
        "generated_at": "2026-10-01T11:00:00Z",
        "portfolio_id": "book-alpha",
        "risk_model_digest": MODEL_DIGEST,
        "covariance_snapshot_digest": COVARIANCE_DIGEST,
        "positions": [
            {"asset_id": f"asset-{index}", "weight": weight} for index, weight in enumerate(weights)
        ],
        "covariance": covariance,
    }


def permissive_policy(**overrides) -> AllocationPolicy:
    values = {
        "max_absolute_risk_share": 1.0,
        "min_effective_risk_positions": 1.0,
        "max_gross_leverage": 10.0,
    }
    values.update(overrides)
    return AllocationPolicy(**values)


def test_balanced_portfolio_passes_euler_and_gradient_checks() -> None:
    report = audit_risk_allocation(artifact(), now=NOW)

    assert report.accepted
    assert report.reasons == ()
    assert report.asset_count == 4
    assert report.component_sum == pytest.approx(report.portfolio_volatility, rel=1e-12)
    assert report.euler_relative_error < 1e-12
    assert report.max_gradient_relative_error < 1e-7
    assert sum(item.signed_risk_share for item in report.allocations) == pytest.approx(1.0)
    assert sum(item.absolute_risk_budget_share for item in report.allocations) == pytest.approx(1.0)


def test_real_synthetic_covariance_is_audited() -> None:
    returns = synthetic_returns(rows=500, seed=29)
    covariance = returns.cov().to_numpy().tolist()
    payload = artifact((0.25, 0.20, 0.20, 0.20, 0.15), covariance)

    report = audit_risk_allocation(payload, now=NOW, policy=permissive_policy())

    assert report.accepted
    assert report.portfolio_volatility > 0.0
    assert report.effective_risk_positions > 1.0


def test_concentrated_component_risk_is_rejected() -> None:
    payload = artifact((0.94, 0.02, 0.02, 0.02))

    report = audit_risk_allocation(payload, now=NOW)

    assert not report.accepted
    assert "component_risk_concentration_exceeded" in report.reasons
    assert "insufficient_effective_risk_positions" in report.reasons


def test_short_hedge_is_preserved_as_negative_component() -> None:
    covariance = [[0.04, 0.03], [0.03, 0.04]]
    payload = artifact((1.2, -0.2), covariance)
    payload["positions"] = payload["positions"][:2]
    payload["covariance"] = covariance

    report = audit_risk_allocation(payload, now=NOW, policy=permissive_policy())

    assert report.accepted
    assert report.negative_component_count == 1
    assert any(item.component_volatility < 0.0 for item in report.allocations)


def test_gross_and_net_exposure_policies_are_independent() -> None:
    gross = artifact((1.5, -0.5, 0.0, 0.0))
    net = artifact((0.2, 0.2, 0.2, 0.2))

    gross_report = audit_risk_allocation(
        gross,
        now=NOW,
        policy=permissive_policy(max_gross_leverage=1.5),
    )
    net_report = audit_risk_allocation(net, now=NOW, policy=permissive_policy())

    assert "gross_leverage_budget_exceeded" in gross_report.reasons
    assert "net_exposure_mismatch" in net_report.reasons


def test_stale_and_future_evidence_are_rejected() -> None:
    stale = audit_risk_allocation(
        artifact(),
        now=NOW + timedelta(seconds=101),
        policy=AllocationPolicy(max_age_seconds=100),
    )
    future = audit_risk_allocation(
        artifact(),
        now=NOW - timedelta(seconds=61),
        policy=AllocationPolicy(max_future_skew_seconds=60),
    )

    assert stale.reasons == ("stale_allocation_evidence",)
    assert future.reasons == ("future_generated_at",)


def test_semantic_asset_permutation_has_identical_evidence() -> None:
    payload = artifact()
    order = [2, 0, 3, 1]
    permuted = copy.deepcopy(payload)
    permuted["positions"] = [payload["positions"][index] for index in order]
    matrix = np.asarray(payload["covariance"])
    permuted["covariance"] = matrix[np.ix_(order, order)].tolist()

    original_report = audit_risk_allocation(payload, now=NOW)
    permuted_report = audit_risk_allocation(permuted, now=NOW)

    assert original_report.to_dict() == permuted_report.to_dict()


def test_report_does_not_expose_positions_or_covariance() -> None:
    payload = artifact()
    report = audit_risk_allocation(payload, now=NOW)
    rendered = json.dumps(report.to_dict())

    assert "book-alpha" not in rendered
    assert "asset-0" not in rendered
    assert '"weight"' not in rendered
    assert '"covariance"' not in rendered


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(extra=True), "missing or unexpected"),
        (
            lambda value: value["positions"].append(value["positions"][0]),
            "asset IDs",
        ),
        (lambda value: value["positions"][0].update(weight=float("nan")), "finite"),
        (lambda value: value.update(generated_at="2026-10-01"), "timezone"),
        (lambda value: value["covariance"].pop(), "square matrix"),
        (lambda value: value.update(risk_model_digest="A" * 64), "lowercase SHA-256"),
    ],
)
def test_malformed_artifacts_fail_closed(mutate, message: str) -> None:
    payload = artifact()
    mutate(payload)

    with pytest.raises(RiskAllocationError, match=message):
        audit_risk_allocation(payload, now=NOW)


def test_asymmetric_covariance_fails_closed() -> None:
    payload = artifact()
    payload["covariance"][0][1] += 0.001

    with pytest.raises(RiskAllocationError, match="symmetry tolerance"):
        audit_risk_allocation(payload, now=NOW)


def test_non_positive_portfolio_variance_fails_closed() -> None:
    payload = artifact(covariance=np.zeros((4, 4)).tolist())

    with pytest.raises(RiskAllocationError, match="variance"):
        audit_risk_allocation(payload, now=NOW)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_age_seconds": -1},
        {"max_gross_leverage": 0.0},
        {"max_absolute_risk_share": 1.1},
        {"min_effective_risk_positions": 0.5},
        {"finite_difference_step": float("nan")},
        {"max_future_skew_seconds": True},
    ],
)
def test_invalid_policies_fail_closed(kwargs: dict) -> None:
    with pytest.raises(RiskAllocationError):
        AllocationPolicy(**kwargs)


def test_parse_artifact_returns_typed_immutable_rows() -> None:
    parsed = parse_artifact(artifact())

    assert parsed.positions[0].asset_id == "asset-0"
    assert parsed.covariance[0][0] == 0.04


def run_cli(tmp_path: Path, raw: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(raw, encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "src.risk_allocation_audit",
            str(artifact_path),
            "--now",
            "2026-10-01T11:00:05Z",
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_accepts_and_atomically_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "allocation.json"
    completed = run_cli(tmp_path, json.dumps(artifact()), "--output", str(output))

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert json.loads(output.read_text())["accepted"] is True


def test_cli_returns_three_for_policy_rejection(tmp_path: Path) -> None:
    completed = run_cli(tmp_path, json.dumps(artifact((0.94, 0.02, 0.02, 0.02))))

    assert completed.returncode == 3
    assert json.loads(completed.stdout)["accepted"] is False


@pytest.mark.parametrize(
    "raw",
    [
        '{"generated_at":"2026-10-01T11:00:00Z","positions":[],"positions":[]}',
        '{"generated_at":NaN,"positions":[]}',
        "not-json",
    ],
)
def test_cli_returns_two_for_malformed_json(tmp_path: Path, raw: str) -> None:
    completed = run_cli(tmp_path, raw)

    assert completed.returncode == 2
    assert json.loads(completed.stdout)["error"] == "malformed_artifact"
