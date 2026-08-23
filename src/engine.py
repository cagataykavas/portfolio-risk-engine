from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .advanced_risk import (
    finite_difference_var_contributions,
    historical_measure,
    monte_carlo_measure,
    parametric_var_cvar,
)
from .risk import max_drawdown, normalize_weights, portfolio_returns
from .scenarios import DEFAULT_SCENARIOS, scenario_pnl


@dataclass(frozen=True)
class PortfolioSpec:
    assets: tuple[str, ...]
    weights: tuple[float, ...]

    def normalized(self) -> np.ndarray:
        return normalize_weights(np.asarray(self.weights, dtype=float))


class PortfolioRiskEngine:
    def analyze(
        self,
        returns: pd.DataFrame,
        spec: PortfolioSpec,
        *,
        confidence: float = 0.95,
        monte_carlo_scenarios: int = 30_000,
        seed: int = 42,
        factor_loadings: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        missing = [asset for asset in spec.assets if asset not in returns.columns]
        if missing:
            raise ValueError(f"missing return columns: {missing}")
        selected = returns.loc[:, list(spec.assets)].dropna()
        weights = spec.normalized()
        p = portfolio_returns(selected, weights)
        annualized_return = float(p.mean() * 252)
        annualized_volatility = float(p.std(ddof=1) * np.sqrt(252))
        sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0

        historical = historical_measure(selected, weights, confidence)
        parametric = parametric_var_cvar(float(p.mean()), float(p.std(ddof=1)), confidence)
        monte_carlo = monte_carlo_measure(selected, weights, confidence, monte_carlo_scenarios, seed)
        contributions = finite_difference_var_contributions(selected, weights, confidence)

        stress_results: list[dict[str, object]] = []
        if factor_loadings:
            for scenario in DEFAULT_SCENARIOS:
                stress_results.append(scenario_pnl(list(spec.assets), weights, scenario, factor_loadings))

        return {
            "assets": list(spec.assets),
            "weights": {asset: float(weight) for asset, weight in zip(spec.assets, weights)},
            "observations": len(selected),
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_zero_rf": sharpe,
            "max_drawdown": max_drawdown(p),
            "risk_measures": [historical.as_dict(), parametric.as_dict(), monte_carlo.as_dict()],
            "historical_var_contributions": contributions,
            "stress_tests": stress_results,
        }
