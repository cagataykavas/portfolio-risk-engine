from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import app
from src.engine import PortfolioRiskEngine, PortfolioSpec
from src.return_panel import ReturnPanelError, ReturnPanelPolicy, admit_return_panel
from src.synthetic import synthetic_returns


def _panel(rows: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {"equity": np.linspace(-0.02, 0.02, rows), "bonds": np.linspace(0.01, -0.01, rows)},
        index=pd.date_range("2025-01-01", periods=rows, freq="B"),
    )


def test_admission_emits_stable_privacy_reduced_fingerprint():
    panel = _panel()
    first = admit_return_panel(panel, ("equity", "bonds"), (0.6, 0.4))
    second = admit_return_panel(panel.copy(), ("equity", "bonds"), (0.6, 0.4))

    assert first.report == second.report
    assert first.report.accepted is True
    assert first.report.observations == 80
    assert first.report.asset_count == 2
    assert len(first.report.content_sha256) == 64
    assert "equity" not in first.report.as_dict().values()
    assert "returns" not in first.report.as_dict()


def test_fingerprint_changes_with_value_column_order_or_index():
    panel = _panel()
    baseline = admit_return_panel(panel, ("equity", "bonds"), (0.6, 0.4)).report.content_sha256

    changed = panel.copy()
    changed.iloc[0, 0] += 1e-12
    assert admit_return_panel(changed, ("equity", "bonds"), (0.6, 0.4)).report.content_sha256 != baseline
    assert admit_return_panel(panel, ("bonds", "equity"), (0.4, 0.6)).report.content_sha256 != baseline

    reindexed = panel.copy()
    reindexed.index = reindexed.index + pd.Timedelta(days=1)
    assert admit_return_panel(reindexed, ("equity", "bonds"), (0.6, 0.4)).report.content_sha256 != baseline


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_missing_and_non_finite_returns(bad_value: float):
    panel = _panel()
    panel.iloc[10, 0] = bad_value
    with pytest.raises(ReturnPanelError, match="missing or non-finite"):
        admit_return_panel(panel, ("equity", "bonds"), (0.6, 0.4))


def test_rejects_duplicate_or_unordered_index():
    duplicate = _panel()
    duplicate.index = list(duplicate.index[:-1]) + [duplicate.index[-2]]
    with pytest.raises(ReturnPanelError, match="index must be unique"):
        admit_return_panel(duplicate, ("equity", "bonds"), (0.6, 0.4))

    unordered = _panel().iloc[::-1]
    with pytest.raises(ReturnPanelError, match="ordered ascending"):
        admit_return_panel(unordered, ("equity", "bonds"), (0.6, 0.4))


def test_rejects_duplicate_assets_and_ambiguous_columns():
    with pytest.raises(ReturnPanelError, match="asset identifiers must be unique"):
        admit_return_panel(_panel(), ("equity", "equity"), (0.5, 0.5))

    ambiguous = _panel()
    ambiguous.columns = ["equity", "equity"]
    with pytest.raises(ReturnPanelError, match="return columns must be unique"):
        admit_return_panel(ambiguous, ("equity",), (1.0,))


def test_rejects_invalid_weights_and_impossible_simple_returns():
    with pytest.raises(ReturnPanelError, match="weights contain non-finite"):
        admit_return_panel(_panel(), ("equity", "bonds"), (np.nan, 1.0))

    panel = _panel()
    panel.iloc[3, 1] = -1.01
    with pytest.raises(ReturnPanelError, match="below -100 percent"):
        admit_return_panel(panel, ("equity", "bonds"), (0.6, 0.4))


def test_policy_enforces_sample_and_resource_bounds():
    with pytest.raises(ReturnPanelError, match="at least 60 observations"):
        admit_return_panel(_panel(rows=59), ("equity", "bonds"), (0.6, 0.4))

    policy = ReturnPanelPolicy(minimum_observations=10, maximum_observations=79)
    with pytest.raises(ReturnPanelError, match="observation count exceeds"):
        admit_return_panel(_panel(), ("equity", "bonds"), (0.6, 0.4), policy=policy)

    cell_policy = ReturnPanelPolicy(maximum_cells=100)
    with pytest.raises(ReturnPanelError, match="cell budget"):
        admit_return_panel(
            _panel(), ("equity", "bonds"), (0.6, 0.4), policy=cell_policy
        )


def test_engine_reports_admission_for_exact_analyzed_panel():
    returns = synthetic_returns(rows=100, seed=19)
    spec = PortfolioSpec(tuple(returns.columns), (0.30, 0.20, 0.25, 0.15, 0.10))

    result = PortfolioRiskEngine().analyze(returns, spec, monte_carlo_scenarios=1000)
    direct = admit_return_panel(returns, spec.assets, spec.weights)

    assert result["observations"] == 100
    assert result["return_panel_admission"] == direct.report.as_dict()


def test_api_rejects_a_partial_row_instead_of_dropping_it():
    records = [{"equity": 0.01, "bonds": 0.001} for _ in range(60)]
    del records[17]["bonds"]

    response = TestClient(app).post(
        "/risk",
        json={
            "assets": ["equity", "bonds"],
            "weights": [0.6, 0.4],
            "returns": records,
            "monte_carlo_scenarios": 1000,
        },
    )

    assert response.status_code == 422
    assert "missing or non-finite" in response.json()["detail"]
