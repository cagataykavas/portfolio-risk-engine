from __future__ import annotations

import json

import numpy as np
import pytest

from src.expected_shortfall_allocation import (
    ExpectedShortfallAllocationError,
    allocate_historical_expected_shortfall,
)


def scenarios(rows: int = 1000) -> np.ndarray:
    generator = np.random.default_rng(41)
    ordinary = generator.normal(0.0004, 0.009, size=(rows, 3))
    # Preserve non-Gaussian joint losses so this exercises a case covariance-only attribution misses.
    ordinary[:30] += generator.standard_t(df=3, size=(30, 3)) * 0.025 - 0.035
    return ordinary


def allocate(matrix: np.ndarray | None = None, **overrides):
    arguments = {
        "asset_ids": ("equity", "credit", "rates"),
        "weights": (0.45, 0.35, 0.20),
        "scenario_returns": scenarios() if matrix is None else matrix,
        "confidence": 0.975,
    }
    arguments.update(overrides)
    return allocate_historical_expected_shortfall(**arguments)


def test_fat_tail_component_es_reconciles_exactly() -> None:
    report = allocate()

    assert report.scenario_count == 1000
    assert report.effective_tail_scenarios == pytest.approx(25.0)
    assert report.expected_shortfall > report.value_at_risk_cutoff > 0.0
    assert report.component_sum == pytest.approx(report.expected_shortfall, rel=1e-12)
    assert report.reconciliation_relative_error < 1e-12
    assert sum(item.signed_risk_share for item in report.allocations) == pytest.approx(
        1.0
    )
    assert sum(
        item.absolute_risk_budget_share for item in report.allocations
    ) == pytest.approx(1.0)


def test_short_hedge_retains_negative_tail_contribution() -> None:
    generator = np.random.default_rng(9)
    factor = generator.normal(0.0, 0.01, size=1000)
    matrix = np.column_stack(
        (factor, factor * 0.95 + generator.normal(0.0, 0.001, size=1000))
    )

    report = allocate_historical_expected_shortfall(
        asset_ids=("risk-asset", "hedge"),
        weights=(1.2, -0.2),
        scenario_returns=matrix,
    )

    assert report.negative_component_count == 1
    assert any(item.component_expected_shortfall < 0.0 for item in report.allocations)


def test_var_boundary_ties_share_fractional_tail_mass() -> None:
    # Ten identical worst portfolio losses straddle the 2.5-observation tail at 95%.
    matrix = np.vstack(
        (np.tile((-0.04, -0.02), (10, 1)), np.tile((0.01, 0.01), (40, 1)))
    )

    report = allocate_historical_expected_shortfall(
        asset_ids=("a", "b"),
        weights=(0.5, 0.5),
        scenario_returns=matrix,
        confidence=0.95,
        min_effective_tail_scenarios=2.0,
    )

    assert report.effective_tail_scenarios == pytest.approx(2.5)
    assert report.value_at_risk_cutoff == pytest.approx(0.03)
    assert report.expected_shortfall == pytest.approx(0.03)


def test_scenario_order_does_not_change_allocation() -> None:
    matrix = scenarios()
    shuffled = matrix[np.random.default_rng(7).permutation(len(matrix))]

    assert allocate(matrix).to_dict() == allocate(shuffled).to_dict()


def test_asset_permutation_preserves_semantic_allocations() -> None:
    matrix = scenarios()
    original = allocate(matrix)
    order = [2, 0, 1]
    permuted = allocate_historical_expected_shortfall(
        asset_ids=tuple(np.asarray(("equity", "credit", "rates"))[order]),
        weights=np.asarray((0.45, 0.35, 0.20))[order],
        scenario_returns=matrix[:, order],
        confidence=0.975,
    )

    assert permuted.to_dict() == original.to_dict()


def test_report_hashes_asset_ids_and_omits_scenarios() -> None:
    rendered = json.dumps(allocate().to_dict())

    assert "equity" not in rendered
    assert "scenario_returns" not in rendered


def test_sample_must_support_minimum_tail_mass() -> None:
    with pytest.raises(ExpectedShortfallAllocationError, match="too small"):
        allocate_historical_expected_shortfall(
            asset_ids=("a", "b"),
            weights=(0.5, 0.5),
            scenario_returns=np.full((100, 2), -0.01),
            confidence=0.99,
            min_effective_tail_scenarios=5.0,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"asset_ids": ("a", "a")}, "unique"),
        ({"weights": (1.0,)}, "one value per asset"),
        ({"scenario_returns": np.zeros((1000, 2))}, "scenario-by-asset"),
        ({"scenario_returns": np.full((1000, 3), np.nan)}, "finite"),
        ({"confidence": 1.0}, "within"),
        ({"confidence": True}, "numeric"),
        ({"min_effective_tail_scenarios": 0.0}, "positive"),
    ],
)
def test_malformed_inputs_fail_closed(overrides: dict, message: str) -> None:
    with pytest.raises(ExpectedShortfallAllocationError, match=message):
        allocate(**overrides)


def test_non_positive_tail_loss_cannot_define_risk_shares() -> None:
    with pytest.raises(ExpectedShortfallAllocationError, match="must be positive"):
        allocate(np.full((1000, 3), 0.01))
