from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from math import erfc, exp, isfinite, log, sqrt
from typing import Any


@dataclass(frozen=True)
class VaRBacktestResult:
    observations: int
    exceptions: int
    expected_exception_rate: float
    observed_exception_rate: float
    kupiec_lr: float
    kupiec_p_value: float
    independence_lr: float
    independence_p_value: float
    conditional_coverage_lr: float
    conditional_coverage_p_value: float
    transition_counts: dict[str, int]
    coverage_pass: bool
    independence_pass: bool
    conditional_coverage_pass: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bernoulli_log_likelihood(successes: int, trials: int, probability: float) -> float:
    failures = trials - successes
    value = 0.0
    if successes:
        if probability <= 0.0:
            return float("-inf")
        value += successes * log(probability)
    if failures:
        if probability >= 1.0:
            return float("-inf")
        value += failures * log(1.0 - probability)
    return value


def _chi_square_survival(statistic: float, degrees_of_freedom: int) -> float:
    statistic = max(0.0, statistic)
    if degrees_of_freedom == 1:
        return erfc(sqrt(statistic / 2.0))
    if degrees_of_freedom == 2:
        return exp(-statistic / 2.0)
    raise ValueError("only one- and two-degree chi-square tests are supported")


def _likelihood_ratio(null_log_likelihood: float, alternative_log_likelihood: float) -> float:
    statistic = 2.0 * (alternative_log_likelihood - null_log_likelihood)
    return max(0.0, statistic)


def backtest_var(
    losses: Sequence[float],
    var_forecasts: Sequence[float],
    *,
    confidence: float = 0.99,
    significance: float = 0.05,
) -> VaRBacktestResult:
    """Run unconditional, independence and conditional-coverage VaR tests.

    Exceptions occur when realized loss is strictly greater than the forecast
    VaR. P-values use asymptotic chi-square reference distributions.
    """
    if len(losses) != len(var_forecasts):
        raise ValueError("losses and var_forecasts must have equal length")
    if len(losses) < 2:
        raise ValueError("at least two observations are required")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    if not 0.0 < significance < 1.0:
        raise ValueError("significance must be between 0 and 1")

    realized = [float(value) for value in losses]
    forecasts = [float(value) for value in var_forecasts]
    if not all(isfinite(value) for value in (*realized, *forecasts)):
        raise ValueError("losses and var_forecasts must be finite")

    exceptions = [int(loss > forecast) for loss, forecast in zip(realized, forecasts, strict=True)]
    observations = len(exceptions)
    exception_count = sum(exceptions)
    expected_rate = 1.0 - confidence
    observed_rate = exception_count / observations

    null_coverage = _bernoulli_log_likelihood(exception_count, observations, expected_rate)
    fitted_coverage = _bernoulli_log_likelihood(exception_count, observations, observed_rate)
    kupiec_lr = _likelihood_ratio(null_coverage, fitted_coverage)
    kupiec_p = _chi_square_survival(kupiec_lr, 1)

    counts = {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
    for previous, current in pairwise(exceptions):
        counts[f"n{previous}{current}"] += 1

    transitions_from_zero = counts["n00"] + counts["n01"]
    transitions_from_one = counts["n10"] + counts["n11"]
    total_transitions = transitions_from_zero + transitions_from_one
    total_next_exceptions = counts["n01"] + counts["n11"]
    common_rate = total_next_exceptions / total_transitions
    rate_after_zero = counts["n01"] / transitions_from_zero if transitions_from_zero else 0.0
    rate_after_one = counts["n11"] / transitions_from_one if transitions_from_one else 0.0

    independent_ll = _bernoulli_log_likelihood(
        total_next_exceptions, total_transitions, common_rate
    )
    markov_ll = _bernoulli_log_likelihood(counts["n01"], transitions_from_zero, rate_after_zero)
    markov_ll += _bernoulli_log_likelihood(counts["n11"], transitions_from_one, rate_after_one)
    independence_lr = _likelihood_ratio(independent_ll, markov_ll)
    independence_p = _chi_square_survival(independence_lr, 1)

    conditional_lr = kupiec_lr + independence_lr
    conditional_p = _chi_square_survival(conditional_lr, 2)
    return VaRBacktestResult(
        observations=observations,
        exceptions=exception_count,
        expected_exception_rate=expected_rate,
        observed_exception_rate=observed_rate,
        kupiec_lr=kupiec_lr,
        kupiec_p_value=kupiec_p,
        independence_lr=independence_lr,
        independence_p_value=independence_p,
        conditional_coverage_lr=conditional_lr,
        conditional_coverage_p_value=conditional_p,
        transition_counts=counts,
        coverage_pass=kupiec_p >= significance,
        independence_pass=independence_p >= significance,
        conditional_coverage_pass=conditional_p >= significance,
    )
