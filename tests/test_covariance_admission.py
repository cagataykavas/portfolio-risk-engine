from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.covariance_admission import (
    CovarianceEvidenceError,
    CovariancePolicy,
    admit_covariance,
    audit_artifact,
    load_artifact,
    main,
)
from src.engine import PortfolioRiskEngine, PortfolioSpec


def artifact() -> dict:
    return {
        "schema_version": "1.0",
        "estimator_id": "sample-covariance-v1",
        "asset_ids": ["equity", "bonds", "gold"],
        "observation_count": 300,
        "covariance": [
            [0.0400, 0.0060, 0.0040],
            [0.0060, 0.0100, 0.0015],
            [0.0040, 0.0015, 0.0225],
        ],
    }


def test_accepts_well_conditioned_covariance_without_repair():
    admission = audit_artifact(artifact())
    assert admission.report.accepted
    assert not admission.report.repaired
    assert admission.report.reason_codes == ()
    assert np.allclose(admission.covariance, artifact()["covariance"])
    assert len(admission.report.report_sha256) == 64


def test_repairs_near_singular_covariance_and_preserves_variances():
    matrix = np.array([[1.0, 0.9999999999], [0.9999999999, 1.0]])
    admission = admit_covariance(
        matrix,
        asset_ids=["left", "right"],
        observation_count=100,
        estimator_id="near-singular-v1",
    )
    assert admission.report.accepted
    assert admission.report.repaired
    assert admission.report.repaired_minimum_eigenvalue > 0
    assert np.allclose(np.diag(admission.covariance), np.diag(matrix), atol=1e-12)


def test_rejects_material_indefiniteness():
    admission = admit_covariance(
        [[0.04, 0.048], [0.048, 0.04]],
        asset_ids=["left", "right"],
        observation_count=100,
        estimator_id="broken-v1",
    )
    assert not admission.report.accepted
    assert "CORRELATION_BOUND_EXCEEDED" in admission.report.reason_codes
    assert "REPAIR_NORM_EXCEEDED" in admission.report.reason_codes


def test_rejects_excessive_asymmetry():
    admission = admit_covariance(
        [[1.0, 0.1], [0.2, 1.0]],
        asset_ids=["left", "right"],
        observation_count=100,
        estimator_id="asymmetric-v1",
    )
    assert not admission.report.accepted
    assert admission.report.reason_codes == ("ASYMMETRY_EXCEEDED",)


def test_rejects_insufficient_sample_evidence():
    admission = admit_covariance(
        np.eye(10),
        asset_ids=[f"asset-{index}" for index in range(10)],
        observation_count=30,
        estimator_id="undersampled-v1",
    )
    assert not admission.report.accepted
    assert "INSUFFICIENT_OBSERVATION_ASSET_RATIO" in admission.report.reason_codes


def test_policy_can_bound_condition_number_after_repair():
    admission = admit_covariance(
        [[1.0, 0.9999999999], [0.9999999999, 1.0]],
        asset_ids=["left", "right"],
        observation_count=100,
        estimator_id="condition-v1",
        policy=CovariancePolicy(max_repaired_condition_number=10.0),
    )
    assert "CONDITION_NUMBER_EXCEEDED" in admission.report.reason_codes


def test_report_is_deterministic_and_does_not_echo_matrix():
    first = audit_artifact(artifact()).report.as_dict()
    second = audit_artifact(artifact()).report.as_dict()
    assert first == second
    rendered = json.dumps(first, sort_keys=True)
    assert '"covariance"' not in rendered
    assert "0.006" not in rendered


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("asset_ids", ["duplicate", "duplicate", "gold"]),
        ("asset_ids", ["only-one"]),
        ("observation_count", 0),
        ("observation_count", True),
        ("estimator_id", "bad id"),
        ("covariance", [[1.0, 0.0], [0.0, 1.0]]),
        ("covariance", [[1.0, 0.0, 0.0], [0.0, float("nan"), 0.0], [0.0, 0.0, 1.0]]),
        ("covariance", [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
        ("covariance", [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
)
def test_malformed_artifacts_fail_closed(field, value):
    evidence = artifact()
    evidence[field] = value
    with pytest.raises(CovarianceEvidenceError):
        audit_artifact(evidence)


@pytest.mark.parametrize(
    "policy",
    [
        CovariancePolicy(min_assets=3, max_assets=2),
        CovariancePolicy(max_asymmetry=-1.0),
        CovariancePolicy(max_absolute_correlation=0.9),
        CovariancePolicy(max_repaired_condition_number=0.5),
        CovariancePolicy(minimum_variance=0.1, maximum_covariance_magnitude=0.01),
    ],
)
def test_invalid_policy_fails_closed(policy):
    with pytest.raises(CovarianceEvidenceError):
        audit_artifact(artifact(), policy)


def test_strict_loader_rejects_duplicate_and_non_finite_json(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"1.0","schema_version":"1.0"}')
    with pytest.raises(CovarianceEvidenceError, match="duplicate"):
        load_artifact(duplicate)
    non_finite = tmp_path / "non-finite.json"
    non_finite.write_text('{"value":NaN}')
    with pytest.raises(CovarianceEvidenceError, match="non-finite"):
        load_artifact(non_finite)


def test_loader_enforces_byte_budget_before_decoding(tmp_path):
    source = tmp_path / "artifact.json"
    source.write_text(json.dumps(artifact()))
    with pytest.raises(CovarianceEvidenceError, match="byte budget"):
        load_artifact(source, max_bytes=10)


def test_cli_has_distinct_accepted_rejected_and_malformed_exits(tmp_path, capsys):
    accepted_path = tmp_path / "accepted.json"
    accepted_path.write_text(json.dumps(artifact()))
    output_path = tmp_path / "reports" / "report.json"
    assert main([str(accepted_path), "--output", str(output_path)]) == 0
    assert json.loads(output_path.read_text())["accepted"] is True
    capsys.readouterr()

    rejected = artifact()
    rejected["observation_count"] = 3
    rejected_path = tmp_path / "rejected.json"
    rejected_path.write_text(json.dumps(rejected))
    assert main([str(rejected_path)]) == 2
    assert json.loads(capsys.readouterr().out)["accepted"] is False

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    assert main([str(malformed)]) == 3
    assert json.loads(capsys.readouterr().out)["error_code"] == "MALFORMED_ARTIFACT"


def test_engine_uses_repaired_covariance_and_exposes_evidence():
    rng = np.random.default_rng(4)
    left = rng.normal(0.0005, 0.01, 120)
    returns = pd.DataFrame({"left": left, "duplicate": left})
    result = PortfolioRiskEngine().analyze(
        returns,
        PortfolioSpec(("left", "duplicate"), (0.5, 0.5)),
        monte_carlo_scenarios=1_000,
    )
    assert result["covariance_admission"]["accepted"]
    assert result["covariance_admission"]["repaired"]
    assert all(np.isfinite(item["var"]) for item in result["risk_measures"])


def test_engine_rejects_under_sampled_covariance():
    returns = pd.DataFrame(np.eye(5), columns=[f"asset-{index}" for index in range(5)])
    spec = PortfolioSpec(tuple(returns.columns), (1, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="covariance admission rejected"):
        PortfolioRiskEngine().analyze(returns, spec, monte_carlo_scenarios=100)
