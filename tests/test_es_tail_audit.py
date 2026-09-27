from __future__ import annotations

import copy
import json
import math
from datetime import UTC, datetime, timedelta, timezone

import pytest

from src.es_tail_audit import (
    ArtifactMalformed,
    ESTailPolicy,
    _newey_west_long_run_variance,
    audit_es_tail,
    load_artifact,
    main,
)

AS_OF = datetime(2026, 9, 27, 3, 0, tzinfo=UTC)


def _artifact(
    *,
    count: int = 400,
    breach_loss: float = 0.04,
    confidence: float = 0.95,
    forecast_horizon: int = 5,
    hac_lag: int = 4,
    step_seconds: int = 60,
) -> dict[str, object]:
    start = datetime(2026, 9, 10, tzinfo=UTC)
    observations = []
    for index in range(count):
        observations.append(
            {
                "timestamp": (start + timedelta(seconds=index * step_seconds))
                .isoformat()
                .replace("+00:00", "Z"),
                "loss": breach_loss if index % 20 == 19 else 0.0,
                "value_at_risk": 0.02,
                "expected_shortfall": 0.04,
            }
        )
    return {
        "schema_version": 1,
        "audit_id": "book-a.2026-09",
        "model_sha256": "a" * 64,
        "forecast_config_sha256": "b" * 64,
        "created_at": "2026-09-27T02:59:30Z",
        "confidence": confidence,
        "forecast_horizon": forecast_horizon,
        "hac_lag": hac_lag,
        "observations": observations,
    }


def _codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


def test_calibrated_var_es_pair_is_accepted() -> None:
    report = audit_es_tail(_artifact(), as_of=AS_OF)

    assert report.accepted is True
    assert report.status == "accepted"
    assert report.observation_count == 400
    assert report.expected_tail_observations == pytest.approx(20.0)
    assert report.actual_tail_observations == 20
    assert report.loss_above_expected_shortfall_count == 0
    assert report.mean_identification_residual == pytest.approx(0.0, abs=1e-14)
    assert report.mean_tail_buffer == pytest.approx(0.02)
    assert report.relative_tail_bias == pytest.approx(0.0, abs=1e-12)
    assert report.long_run_variance > 0.0
    assert 0.0 <= report.underestimation_p_value <= 1.0


def test_tail_losses_worse_than_es_are_rejected() -> None:
    report = audit_es_tail(_artifact(breach_loss=0.06), as_of=AS_OF)

    assert report.accepted is False
    assert report.mean_identification_residual == pytest.approx(0.02)
    assert report.relative_tail_bias == pytest.approx(1.0)
    assert report.underestimation_p_value < 0.05
    assert report.loss_above_expected_shortfall_count == 20
    assert "RELATIVE_TAIL_BIAS_ABOVE_POLICY" in _codes(report)
    assert "TAIL_UNDERESTIMATION_SIGNIFICANT" in _codes(report)


def test_conservative_es_is_not_treated_as_underestimation() -> None:
    report = audit_es_tail(_artifact(breach_loss=0.03), as_of=AS_OF)

    assert report.accepted is True
    assert report.mean_identification_residual == pytest.approx(-0.01)
    assert report.relative_tail_bias == pytest.approx(-0.5)
    assert report.underestimation_p_value > 0.5


def test_identification_moment_uses_var_exceedance_not_es_exceedance() -> None:
    artifact = _artifact()
    for observation in artifact["observations"]:
        if observation["loss"] > observation["value_at_risk"]:
            observation["loss"] = 0.035

    report = audit_es_tail(artifact, as_of=AS_OF)

    assert report.actual_tail_observations == 20
    assert report.loss_above_expected_shortfall_count == 0
    assert report.mean_identification_residual < 0.0


@pytest.mark.parametrize(
    ("artifact", "policy", "code"),
    [
        (
            _artifact(count=200),
            ESTailPolicy(min_observations=250),
            "OBSERVATION_EVIDENCE_UNDERPOWERED",
        ),
        (
            _artifact(confidence=0.99),
            ESTailPolicy(),
            "EXPECTED_TAIL_EVIDENCE_UNDERPOWERED",
        ),
    ],
)
def test_underpowered_evidence_is_rejected(artifact, policy, code) -> None:
    report = audit_es_tail(artifact, as_of=AS_OF, policy=policy)

    assert report.accepted is False
    assert code in _codes(report)


def test_too_few_realized_tail_observations_is_rejected() -> None:
    artifact = _artifact()
    for observation in artifact["observations"]:
        observation["loss"] = 0.0

    report = audit_es_tail(artifact, as_of=AS_OF)

    assert "ACTUAL_TAIL_EVIDENCE_UNDERPOWERED" in _codes(report)
    assert "DEGENERATE_LONG_RUN_VARIANCE" in _codes(report)


def test_tiny_tail_buffer_is_rejected_with_finite_report() -> None:
    artifact = _artifact()
    for observation in artifact["observations"]:
        observation["expected_shortfall"] = 0.0200001

    report = audit_es_tail(artifact, as_of=AS_OF)

    assert "MEAN_TAIL_BUFFER_BELOW_POLICY" in _codes(report)
    assert math.isfinite(report.relative_tail_bias)
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("horizon", "hac_lag", "code"),
    [
        (5, 3, "HAC_LAG_BELOW_HORIZON"),
        (1, 400, "HAC_LAG_NOT_IDENTIFIABLE"),
        (0, 0, "FORECAST_HORIZON"),
        (1, -1, "HAC_LAG"),
    ],
)
def test_horizon_and_hac_contract(horizon: int, hac_lag: int, code: str) -> None:
    artifact = _artifact(forecast_horizon=horizon, hac_lag=hac_lag)

    with pytest.raises(ArtifactMalformed, match=code):
        audit_es_tail(artifact, as_of=AS_OF)


@pytest.mark.parametrize("confidence", [0.5, 1.0, -0.1, True, "0.95"])
def test_confidence_is_strictly_validated(confidence) -> None:
    artifact = _artifact()
    artifact["confidence"] = confidence

    with pytest.raises(ArtifactMalformed, match="CONFIDENCE"):
        audit_es_tail(artifact, as_of=AS_OF)


@pytest.mark.parametrize(
    ("var", "expected_shortfall", "code"),
    [
        (-0.01, 0.04, "VALUE_AT_RISK_NEGATIVE"),
        (0.04, 0.03, "EXPECTED_SHORTFALL_NOT_ABOVE_VAR"),
        (0.04, 0.04, "EXPECTED_SHORTFALL_NOT_ABOVE_VAR"),
    ],
)
def test_var_es_ordering_is_enforced(var: float, expected_shortfall: float, code: str) -> None:
    artifact = _artifact()
    artifact["observations"][0]["value_at_risk"] = var
    artifact["observations"][0]["expected_shortfall"] = expected_shortfall

    with pytest.raises(ArtifactMalformed, match=code):
        audit_es_tail(artifact, as_of=AS_OF)


@pytest.mark.parametrize("value", [True, "0.1", float("nan"), float("inf"), 1_000_001.0])
def test_financial_values_must_be_finite_bounded_numbers(value) -> None:
    artifact = _artifact()
    artifact["observations"][0]["loss"] = value

    with pytest.raises(ArtifactMalformed, match="LOSS"):
        audit_es_tail(artifact, as_of=AS_OF)


def test_duplicate_timestamps_are_rejected() -> None:
    artifact = _artifact()
    artifact["observations"][1]["timestamp"] = artifact["observations"][0]["timestamp"]

    with pytest.raises(ArtifactMalformed, match="DUPLICATE_TIMESTAMP"):
        audit_es_tail(artifact, as_of=AS_OF)


def test_observations_must_be_strictly_ordered() -> None:
    artifact = _artifact()
    artifact["observations"][0], artifact["observations"][1] = (
        artifact["observations"][1],
        artifact["observations"][0],
    )

    with pytest.raises(ArtifactMalformed, match="TIMESTAMPS_NOT_STRICTLY_ORDERED"):
        audit_es_tail(artifact, as_of=AS_OF)


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (b'{"schema_version":1,"schema_version":1}', "DUPLICATE_JSON_KEY"),
        (b'{"value":NaN}', "NON_FINITE_NUMBER"),
        (b"\xff", "INVALID_UTF8"),
        (b"[]", "ROOT_NOT_OBJECT"),
    ],
)
def test_strict_json_loader(raw: bytes, code: str) -> None:
    with pytest.raises(ArtifactMalformed, match=code):
        load_artifact(raw)


def test_input_byte_budget_is_enforced() -> None:
    with pytest.raises(ArtifactMalformed, match="INPUT_TOO_LARGE"):
        load_artifact(b"{} ", ESTailPolicy(max_input_bytes=2))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda artifact: artifact.update({"unexpected": True}),
        lambda artifact: artifact["observations"][0].update({"unexpected": True}),
    ],
)
def test_unknown_fields_are_rejected(mutate) -> None:
    artifact = _artifact()
    mutate(artifact)

    with pytest.raises(ArtifactMalformed):
        audit_es_tail(artifact, as_of=AS_OF)


@pytest.mark.parametrize(
    ("created_at", "code"),
    [
        ("2026-09-25T00:00:00Z", "STALE_EVIDENCE"),
        ("2026-09-27T03:02:00Z", "EVIDENCE_FROM_FUTURE"),
        ("not-a-time", "CREATED_AT"),
    ],
)
def test_evidence_freshness_is_enforced(created_at: str, code: str) -> None:
    artifact = _artifact()
    artifact["created_at"] = created_at

    with pytest.raises(ArtifactMalformed, match=code):
        audit_es_tail(artifact, as_of=AS_OF)


def test_naive_observation_timestamp_is_rejected() -> None:
    artifact = _artifact()
    artifact["observations"][0]["timestamp"] = "2026-09-10T00:00:00"

    with pytest.raises(ArtifactMalformed, match="TIMESTAMP"):
        audit_es_tail(artifact, as_of=AS_OF)


def test_observation_cannot_postdate_evidence_creation() -> None:
    artifact = _artifact()
    artifact["observations"][-1]["timestamp"] = "2026-09-27T03:00:00Z"

    with pytest.raises(ArtifactMalformed, match="OBSERVATION_AFTER_EVIDENCE"):
        audit_es_tail(artifact, as_of=AS_OF)


def test_observation_budget_is_enforced() -> None:
    artifact = _artifact(count=301)
    policy = ESTailPolicy(min_observations=200, max_observations=300)

    with pytest.raises(ArtifactMalformed, match="OBSERVATION_BUDGET_EXCEEDED"):
        audit_es_tail(artifact, as_of=AS_OF, policy=policy)


def test_digest_canonicalizes_timezone_representation() -> None:
    left = _artifact()
    right = copy.deepcopy(left)
    offset = timezone(timedelta(hours=3))
    right["created_at"] = datetime(2026, 9, 27, 5, 59, 30, tzinfo=offset).isoformat()
    for observation in right["observations"]:
        parsed = datetime.fromisoformat(observation["timestamp"])
        observation["timestamp"] = parsed.astimezone(offset).isoformat()

    left_report = audit_es_tail(left, as_of=AS_OF)
    right_report = audit_es_tail(right, as_of=AS_OF)

    assert left_report.artifact_sha256 == right_report.artifact_sha256
    assert left_report.to_dict() == right_report.to_dict()


def test_report_does_not_expose_raw_audit_id_timestamps_or_financial_rows() -> None:
    artifact = _artifact()
    serialized = json.dumps(audit_es_tail(artifact, as_of=AS_OF).to_dict())

    assert artifact["audit_id"] not in serialized
    assert artifact["observations"][0]["timestamp"] not in serialized
    assert '"value_at_risk"' not in serialized
    assert '"expected_shortfall"' not in serialized
    assert '"loss"' not in serialized


def test_policy_digest_changes_with_threshold() -> None:
    default = audit_es_tail(_artifact(), as_of=AS_OF)
    strict = audit_es_tail(
        _artifact(),
        as_of=AS_OF,
        policy=ESTailPolicy(max_relative_tail_bias=0.20),
    )

    assert default.policy_sha256 != strict.policy_sha256


def test_findings_are_bounded_with_full_count() -> None:
    artifact = _artifact(count=200, breach_loss=0.06)
    policy = ESTailPolicy(max_findings=2)

    report = audit_es_tail(artifact, as_of=AS_OF, policy=policy)

    assert report.findings_truncated is True
    assert report.finding_count > 2
    assert len(report.findings) == 2


def test_newey_west_variance_matches_manual_lag_one_calculation() -> None:
    values = [1.0, -1.0, 1.0, -1.0]
    gamma_zero = 1.0
    gamma_one = -3.0 / 4.0
    expected = gamma_zero + 2.0 * 0.5 * gamma_one

    assert _newey_west_long_run_variance(values, 1) == pytest.approx(expected)


def test_as_of_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        audit_es_tail(_artifact(), as_of=AS_OF.replace(tzinfo=None))


def test_cli_has_stable_exit_codes_and_atomic_output(tmp_path) -> None:
    artifact_path = tmp_path / "es.json"
    report_path = tmp_path / "report.json"
    artifact_path.write_text(json.dumps(_artifact()), encoding="utf-8")

    accepted = main(
        [str(artifact_path), "--as-of", "2026-09-27T03:00:00Z", "--output", str(report_path)]
    )
    assert accepted == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["accepted"] is True

    artifact_path.write_text(json.dumps(_artifact(breach_loss=0.06)), encoding="utf-8")
    rejected = main(
        [str(artifact_path), "--as-of", "2026-09-27T03:00:00Z", "--output", str(report_path)]
    )
    assert rejected == 2

    artifact_path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    assert main([str(artifact_path), "--output", str(report_path)]) == 3
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "malformed"
    assert list(tmp_path.glob(".report.json.*")) == []


def test_large_evidence_set_is_bounded_and_deterministic() -> None:
    artifact = _artifact(count=25_000, forecast_horizon=25, hac_lag=24, step_seconds=30)
    policy = ESTailPolicy(max_observations=25_000, max_hac_lag=24)

    first = audit_es_tail(artifact, as_of=AS_OF, policy=policy)
    second = audit_es_tail(copy.deepcopy(artifact), as_of=AS_OF, policy=policy)

    assert first.accepted is True
    assert first.observation_count == 25_000
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.to_dict() == second.to_dict()
