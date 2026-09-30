from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app
from src.engine import PortfolioRiskEngine, PortfolioSpec
from src.synthetic import FACTOR_LOADINGS, synthetic_returns


def test_three_var_methods_are_finite_and_positive():
    returns = synthetic_returns(rows=400, seed=7)
    spec = PortfolioSpec(tuple(returns.columns), (0.30, 0.20, 0.25, 0.15, 0.10))
    result = PortfolioRiskEngine().analyze(
        returns,
        spec,
        confidence=0.95,
        monte_carlo_scenarios=5000,
        seed=7,
        factor_loadings=FACTOR_LOADINGS,
    )
    assert len(result["risk_measures"]) == 3
    assert all(item["var"] > 0 for item in result["risk_measures"])
    assert all(item["cvar"] >= item["var"] for item in result["risk_measures"])
    assert result["stress_tests"]


def test_contributions_cover_all_assets():
    returns = synthetic_returns(rows=300, seed=11)
    spec = PortfolioSpec(tuple(returns.columns), (1, 1, 1, 1, 1))
    result = PortfolioRiskEngine().analyze(returns, spec, monte_carlo_scenarios=2000)
    assert set(result["historical_var_contributions"]) == set(returns.columns)


def test_demo_endpoint():
    response = TestClient(app).get("/demo")
    assert response.status_code == 200
    assert response.json()["observations"] > 100
