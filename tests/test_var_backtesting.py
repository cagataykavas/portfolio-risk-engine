from __future__ import annotations

import math

import pytest

from src.var_backtesting import backtest_var


def _losses_from_exceptions(exceptions: list[int]) -> tuple[list[float], list[float]]:
    forecasts = [1.0] * len(exceptions)
    losses = [2.0 if value else 0.0 for value in exceptions]
    return losses, forecasts


def test_well_calibrated_dispersed_exceptions_pass_coverage() -> None:
    exceptions = [int(index % 20 == 0) for index in range(200)]
    losses, forecasts = _losses_from_exceptions(exceptions)

    result = backtest_var(losses, forecasts, confidence=0.95)

    assert result.exceptions == 10
    assert result.observed_exception_rate == pytest.approx(0.05)
    assert result.coverage_pass
    assert result.independence_pass
    assert result.conditional_coverage_pass


def test_clustered_exceptions_fail_independence_despite_correct_count() -> None:
    exceptions = [0] * 190 + [1] * 10
    losses, forecasts = _losses_from_exceptions(exceptions)

    result = backtest_var(losses, forecasts, confidence=0.95)

    assert result.coverage_pass
    assert not result.independence_pass
    assert not result.conditional_coverage_pass
    assert result.transition_counts["n11"] == 9


def test_excessive_breaches_fail_unconditional_coverage() -> None:
    exceptions = [1] * 40 + [0] * 160
    losses, forecasts = _losses_from_exceptions(exceptions)

    result = backtest_var(losses, forecasts, confidence=0.95)

    assert result.observed_exception_rate == pytest.approx(0.2)
    assert not result.coverage_pass
    assert result.kupiec_p_value < 0.05


def test_zero_exception_case_remains_finite_and_fails_expected_coverage() -> None:
    losses, forecasts = _losses_from_exceptions([0] * 250)

    result = backtest_var(losses, forecasts, confidence=0.95)

    assert result.exceptions == 0
    assert math.isfinite(result.kupiec_lr)
    assert math.isfinite(result.kupiec_p_value)
    assert not result.coverage_pass


def test_result_is_json_ready() -> None:
    losses, forecasts = _losses_from_exceptions([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    payload = backtest_var(losses, forecasts, confidence=0.9).to_dict()

    assert payload["observations"] == 10
    assert set(payload["transition_counts"]) == {"n00", "n01", "n10", "n11"}
    assert isinstance(payload["conditional_coverage_pass"], bool)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="equal length"):
        backtest_var([0.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="at least two"):
        backtest_var([0.0], [1.0])
    with pytest.raises(ValueError, match="finite"):
        backtest_var([0.0, float("nan")], [1.0, 1.0])
