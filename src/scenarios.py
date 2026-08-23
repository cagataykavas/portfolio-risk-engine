from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .risk import normalize_weights


@dataclass(frozen=True)
class StressScenario:
    name: str
    shocks: dict[str, float]
    description: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SCENARIOS = (
    StressScenario(
        "global_risk_off",
        {"equity": -0.12, "rates": 0.015, "credit": -0.06, "commodity": -0.08, "fx": 0.04},
        "Broad risk-off move with equity/credit losses and defensive FX appreciation.",
    ),
    StressScenario(
        "inflation_shock",
        {"equity": -0.07, "rates": 0.025, "credit": -0.035, "commodity": 0.10, "fx": 0.02},
        "Inflation surprise with higher yields, commodity strength and weaker risk assets.",
    ),
    StressScenario(
        "liquidity_squeeze",
        {"equity": -0.10, "rates": 0.008, "credit": -0.09, "commodity": -0.05, "fx": 0.05},
        "Funding stress with spread widening and simultaneous de-risking.",
    ),
)


def scenario_pnl(
    assets: list[str],
    weights: np.ndarray,
    scenario: StressScenario,
    factor_loadings: dict[str, dict[str, float]],
) -> dict[str, object]:
    w = normalize_weights(weights)
    asset_returns: list[float] = []
    contributions: dict[str, float] = {}
    for asset, weight in zip(assets, w):
        loadings = factor_loadings.get(asset, {})
        shocked_return = sum(loadings.get(factor, 0.0) * shock for factor, shock in scenario.shocks.items())
        asset_returns.append(shocked_return)
        contributions[asset] = float(weight * shocked_return)
    portfolio = float(np.dot(w, np.asarray(asset_returns)))
    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "portfolio_return": portfolio,
        "asset_contributions": contributions,
    }
