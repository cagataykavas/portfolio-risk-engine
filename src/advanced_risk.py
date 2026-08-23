from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from .risk import historical_var_cvar, monte_carlo_pnl, normalize_weights, portfolio_returns


@dataclass(frozen=True)
class RiskMeasure:
    method: str
    confidence: float
    var: float
    cvar: float
    horizon_days: int = 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parametric_var_cvar(
    mean_return: float,
    volatility: float,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> RiskMeasure:
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    scale = horizon_days ** 0.5
    mu = mean_return * horizon_days
    sigma = volatility * scale
    z = NormalDist().inv_cdf(confidence)
    var = -(mu - z * sigma)
    phi = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    cvar = -(mu - sigma * phi / (1.0 - confidence))
    return RiskMeasure("parametric_normal", confidence, float(var), float(cvar), horizon_days)


def historical_measure(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.95,
) -> RiskMeasure:
    p = portfolio_returns(returns, weights)
    var, cvar = historical_var_cvar(p, confidence)
    return RiskMeasure("historical", confidence, var, cvar)


def monte_carlo_measure(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.95,
    scenarios: int = 50_000,
    seed: int = 42,
) -> RiskMeasure:
    pnl = monte_carlo_pnl(
        returns.mean().to_numpy(),
        returns.cov().to_numpy(),
        weights,
        scenarios=scenarios,
        seed=seed,
    )
    losses = -pnl
    var = float(np.quantile(losses, confidence))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) else var
    return RiskMeasure("monte_carlo", confidence, var, cvar)


def finite_difference_var_contributions(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.95,
    epsilon: float = 1e-4,
) -> dict[str, float]:
    """Approximate component VaR contributions using finite differences.

    The contributions are normalized to sum to the portfolio historical VaR, making
    them easier to discuss as a decomposition rather than raw marginal derivatives.
    """
    w = normalize_weights(weights)
    base = historical_measure(returns, w, confidence).var
    marginal: list[float] = []
    for i in range(len(w)):
        bumped = w.copy()
        bumped[i] += epsilon
        bumped = normalize_weights(bumped)
        bumped_var = historical_measure(returns, bumped, confidence).var
        marginal.append((bumped_var - base) / epsilon)
    raw = w * np.asarray(marginal)
    total = float(raw.sum())
    if abs(total) < 1e-12:
        scaled = np.zeros_like(raw)
    else:
        scaled = raw * (base / total)
    return {asset: float(value) for asset, value in zip(returns.columns, scaled)}
