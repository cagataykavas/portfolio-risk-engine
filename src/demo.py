from __future__ import annotations

import numpy as np
import pandas as pd

from .risk import component_volatility_risk, historical_var_cvar, max_drawdown, monte_carlo_pnl, portfolio_returns, stress_pnl


def synthetic_returns(rows: int = 1000, assets: int = 4, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_cov = 0.0001 * np.array(
        [
            [1.0, 0.35, 0.20, 0.10],
            [0.35, 1.4, 0.25, 0.15],
            [0.20, 0.25, 0.8, 0.30],
            [0.10, 0.15, 0.30, 1.2],
        ]
    )[:assets, :assets]
    mean = np.linspace(0.00015, 0.00035, assets)
    data = rng.multivariate_normal(mean, base_cov, size=rows)
    return pd.DataFrame(data, columns=[f"asset_{i+1}" for i in range(assets)])


def main() -> None:
    returns = synthetic_returns()
    weights = np.array([0.30, 0.25, 0.25, 0.20])
    p = portfolio_returns(returns, weights)
    var, cvar = historical_var_cvar(p, 0.95)
    cov = returns.cov().to_numpy()
    mc = monte_carlo_pnl(returns.mean().to_numpy(), cov, weights)

    print(f"mean daily return: {p.mean():.6f}")
    print(f"daily volatility: {p.std():.6f}")
    print(f"95% historical VaR: {var:.6f}")
    print(f"95% historical CVaR: {cvar:.6f}")
    print(f"max drawdown: {max_drawdown(p):.4%}")
    print(f"risk contributions: {component_volatility_risk(cov, weights)}")
    print(f"MC 1% percentile P&L: {np.quantile(mc, .01):.6f}")
    print(f"stress P&L: {stress_pnl(weights, np.array([-0.05, -0.03, -0.07, -0.02])):.4%}")


if __name__ == "__main__":
    main()
