from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if np.isclose(total, 0.0):
        raise ValueError("weights must not sum to zero")
    return weights / total


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    w = normalize_weights(weights)
    if returns.shape[1] != len(w):
        raise ValueError("weights length must match number of assets")
    return pd.Series(returns.to_numpy() @ w, index=returns.index, name="portfolio_return")


def historical_var_cvar(portfolio_return: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    loss = -portfolio_return.dropna().to_numpy()
    var = float(np.quantile(loss, alpha))
    tail = loss[loss >= var]
    cvar = float(tail.mean()) if len(tail) else var
    return var, cvar


def max_drawdown(portfolio_return: pd.Series) -> float:
    wealth = (1.0 + portfolio_return.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def component_volatility_risk(covariance: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = normalize_weights(weights)
    covariance = np.asarray(covariance, dtype=float)
    variance = float(w @ covariance @ w)
    if variance <= 0:
        return np.zeros_like(w)
    sigma = np.sqrt(variance)
    marginal = covariance @ w / sigma
    return w * marginal


def monte_carlo_pnl(
    mean_returns: np.ndarray,
    covariance: np.ndarray,
    weights: np.ndarray,
    scenarios: int = 20000,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = normalize_weights(weights)
    simulated = rng.multivariate_normal(mean_returns, covariance, size=scenarios)
    return simulated @ w


def stress_pnl(weights: np.ndarray, shocks: np.ndarray) -> float:
    w = normalize_weights(weights)
    shocks = np.asarray(shocks, dtype=float)
    if len(w) != len(shocks):
        raise ValueError("one shock is required per asset")
    return float(w @ shocks)
