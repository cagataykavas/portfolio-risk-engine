from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.covariance_benchmark import (
    BenchmarkConfig,
    benchmark_covariance_estimators,
    main,
    walk_forward_forecasts,
)
from src.synthetic import synthetic_returns


def _fixture(rows: int = 180) -> tuple[pd.DataFrame, np.ndarray, BenchmarkConfig]:
    returns = synthetic_returns(rows=rows, seed=17).iloc[:, :3]
    weights = np.array([0.5, 0.3, 0.2])
    config = BenchmarkConfig(window=60, selection_observations=40, evaluation_observations=40)
    return returns, weights, config


def test_benchmark_is_deterministic_and_bounded():
    returns, weights, config = _fixture()
    first = benchmark_covariance_estimators(returns, weights, config)
    second = benchmark_covariance_estimators(returns, weights, config)

    assert first == second
    assert first["status"] == "accepted"
    assert first["selection"]["selected_estimator"] in first["metrics"]
    assert len(first["evidence_sha256"]) == 64
    assert "realized_return" not in json.dumps(first)


def test_forecasts_use_only_information_before_origin():
    returns, weights, config = _fixture()
    baseline = walk_forward_forecasts(returns, weights, config)
    target_origin = baseline["origin"].iloc[30]
    changed = returns.copy()
    changed.loc[target_origin] *= 100.0
    perturbed = walk_forward_forecasts(changed, weights, config)

    before = baseline[baseline["origin"] == target_origin].reset_index(drop=True)
    after = perturbed[perturbed["origin"] == target_origin].reset_index(drop=True)
    np.testing.assert_allclose(before["forecast_variance"], after["forecast_variance"])
    assert not np.allclose(before["realized_return"], after["realized_return"])


def test_selection_is_computed_before_evaluation():
    returns, weights, config = _fixture()
    baseline = benchmark_covariance_estimators(returns, weights, config)
    changed = returns.copy()
    changed.iloc[-config.evaluation_observations :] *= 4.0
    perturbed = benchmark_covariance_estimators(changed, weights, config)

    assert (
        baseline["selection"]["selected_estimator"] == perturbed["selection"]["selected_estimator"]
    )
    for estimator in baseline["metrics"]:
        before = baseline["metrics"][estimator]["selection"]
        after = perturbed["metrics"][estimator]["selection"]
        np.testing.assert_allclose(
            list(before.values()), list(after.values()), rtol=0.0, atol=1e-20
        )


def test_each_estimator_emits_positive_finite_variance():
    returns, weights, config = _fixture()
    forecasts = walk_forward_forecasts(returns, weights, config)

    assert set(forecasts["estimator"]) == {"sample", "diagonal_shrinkage", "ewma"}
    assert (forecasts["forecast_variance"] > 0.0).all()
    assert np.isfinite(forecasts.select_dtypes(include=[float]).to_numpy()).all()
    assert len(forecasts) == 3 * (config.selection_observations + config.evaluation_observations)


@pytest.mark.parametrize("problem", ["duplicate", "unsorted", "nan", "infinite"])
def test_malformed_return_evidence_fails_closed(problem: str):
    returns, weights, config = _fixture()
    if problem == "duplicate":
        returns.index = returns.index.where(np.arange(len(returns)) != 1, returns.index[0])
    elif problem == "unsorted":
        returns = returns.iloc[::-1]
    elif problem == "nan":
        returns.iloc[0, 0] = np.nan
    else:
        returns.iloc[0, 0] = np.inf

    with pytest.raises(ValueError):
        benchmark_covariance_estimators(returns, weights, config)


def test_short_history_and_invalid_policy_fail_closed():
    returns, weights, config = _fixture()
    with pytest.raises(ValueError, match="at least"):
        benchmark_covariance_estimators(returns.iloc[:100], weights, config)
    with pytest.raises(ValueError, match="ewma_decay"):
        benchmark_covariance_estimators(
            returns,
            weights,
            BenchmarkConfig(
                window=60,
                selection_observations=40,
                evaluation_observations=40,
                ewma_decay=1.0,
            ),
        )


def test_weights_are_validated_and_scale_invariant():
    returns, weights, config = _fixture()
    base = benchmark_covariance_estimators(returns, weights, config)
    scaled = benchmark_covariance_estimators(returns, weights * 10.0, config)
    assert base["metrics"] == scaled["metrics"]

    with pytest.raises(ValueError, match="one value per asset"):
        benchmark_covariance_estimators(returns, weights[:2], config)
    with pytest.raises(ValueError, match="sum to zero"):
        benchmark_covariance_estimators(returns, np.zeros(3), config)


def test_cli_writes_atomic_evidence(tmp_path):
    output = tmp_path / "nested" / "benchmark.json"
    exit_code = main(
        [
            "--rows",
            "140",
            "--window",
            "60",
            "--selection-observations",
            "40",
            "--evaluation-observations",
            "40",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "covariance-estimator-benchmark/v1"
    assert not list(output.parent.glob("*.tmp"))


def test_cli_uses_distinct_malformed_exit_code(tmp_path):
    assert main(["--rows", "20", "--output", str(tmp_path / "bad.json")]) == 2


def test_cli_uses_distinct_policy_rejection_exit_code(tmp_path):
    output = tmp_path / "rejected.json"
    exit_code = main(
        [
            "--rows",
            "650",
            "--max-evaluation-regret",
            "0",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 3
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "rejected"
