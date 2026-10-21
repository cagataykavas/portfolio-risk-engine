from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
MAX_ASSETS = 256
MAX_SCENARIOS = 100_000


class ExpectedShortfallAllocationError(ValueError):
    """Raised when Expected Shortfall allocation inputs cannot be audited safely."""


@dataclass(frozen=True)
class AssetExpectedShortfallAllocation:
    asset_ref: str
    component_expected_shortfall: float
    signed_risk_share: float
    absolute_risk_budget_share: float


@dataclass(frozen=True)
class ExpectedShortfallAllocation:
    schema_version: str
    confidence: float
    scenario_count: int
    effective_tail_scenarios: float
    value_at_risk_cutoff: float
    expected_shortfall: float
    component_sum: float
    reconciliation_relative_error: float
    max_absolute_risk_share: float
    risk_budget_hhi: float
    effective_risk_positions: float
    negative_component_count: int
    allocations: tuple[AssetExpectedShortfallAllocation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "allocations": [asdict(allocation) for allocation in self.allocations],
        }


def _bounded_asset_ids(asset_ids: Sequence[str]) -> tuple[str, ...]:
    if not 2 <= len(asset_ids) <= MAX_ASSETS:
        raise ExpectedShortfallAllocationError(
            "asset_ids must contain between 2 and 256 assets"
        )
    result = tuple(asset_ids)
    if any(
        not isinstance(asset_id, str) or not IDENTIFIER_RE.fullmatch(asset_id)
        for asset_id in result
    ):
        raise ExpectedShortfallAllocationError("asset_ids must be bounded identifiers")
    if len(set(result)) != len(result):
        raise ExpectedShortfallAllocationError("asset_ids must be unique")
    return result


def _finite_vector(value: object, *, name: str, expected_size: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ExpectedShortfallAllocationError(f"{name} must be numeric") from error
    if result.shape != (expected_size,):
        raise ExpectedShortfallAllocationError(f"{name} must have one value per asset")
    if not np.isfinite(result).all():
        raise ExpectedShortfallAllocationError(
            f"{name} must contain only finite values"
        )
    return result


def _finite_scenarios(value: object, *, asset_count: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ExpectedShortfallAllocationError(
            "scenario_returns must be numeric"
        ) from error
    if result.ndim != 2 or result.shape[1:] != (asset_count,):
        raise ExpectedShortfallAllocationError(
            "scenario_returns must be a two-dimensional scenario-by-asset matrix"
        )
    if not 2 <= result.shape[0] <= MAX_SCENARIOS:
        raise ExpectedShortfallAllocationError(
            "scenario_returns must contain between 2 and 100000 scenarios"
        )
    if not np.isfinite(result).all():
        raise ExpectedShortfallAllocationError(
            "scenario_returns must contain only finite values"
        )
    return result


def _tail_weights(losses: np.ndarray, tail_mass: float) -> tuple[np.ndarray, float]:
    # Treat all observations tied at the empirical VaR boundary equally. This keeps the
    # allocation independent of input row order while preserving exactly the requested tail mass.
    cutoff_rank = math.ceil(tail_mass) - 1
    cutoff = float(np.sort(losses)[::-1][cutoff_rank])
    strictly_worse = losses > cutoff
    tied = losses == cutoff
    residual_mass = tail_mass - float(np.count_nonzero(strictly_worse))
    tied_count = int(np.count_nonzero(tied))
    if tied_count == 0 or not 0.0 < residual_mass <= tied_count:
        raise ExpectedShortfallAllocationError(
            "could not construct a deterministic tail sample"
        )
    weights = np.zeros_like(losses)
    weights[strictly_worse] = 1.0
    weights[tied] = residual_mass / tied_count
    return weights, cutoff


def allocate_historical_expected_shortfall(
    *,
    asset_ids: Sequence[str],
    weights: object,
    scenario_returns: object,
    confidence: float = 0.975,
    min_effective_tail_scenarios: float = 20.0,
) -> ExpectedShortfallAllocation:
    """Allocate historical Expected Shortfall to assets using the portfolio tail sample.

    Loss is defined as the negative portfolio return. At a fractional empirical quantile,
    observations tied at the VaR boundary share the remaining tail mass equally.
    """

    assets = _bounded_asset_ids(asset_ids)
    position_weights = _finite_vector(
        weights, name="weights", expected_size=len(assets)
    )
    scenarios = _finite_scenarios(scenario_returns, asset_count=len(assets))
    asset_order = np.argsort(np.asarray(assets), kind="stable")
    assets = tuple(assets[index] for index in asset_order)
    position_weights = position_weights[asset_order]
    scenarios = scenarios[:, asset_order]
    scenario_order = np.lexsort(
        tuple(scenarios[:, index] for index in reversed(range(len(assets))))
    )
    scenarios = scenarios[scenario_order]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ExpectedShortfallAllocationError("confidence must be numeric")
    effective_confidence = float(confidence)
    if not math.isfinite(effective_confidence) or not 0.0 < effective_confidence < 1.0:
        raise ExpectedShortfallAllocationError("confidence must be within (0, 1)")
    if isinstance(min_effective_tail_scenarios, bool) or not isinstance(
        min_effective_tail_scenarios, (int, float)
    ):
        raise ExpectedShortfallAllocationError(
            "min_effective_tail_scenarios must be numeric"
        )
    minimum_tail = float(min_effective_tail_scenarios)
    if not math.isfinite(minimum_tail) or minimum_tail <= 0.0:
        raise ExpectedShortfallAllocationError(
            "min_effective_tail_scenarios must be finite and positive"
        )

    tail_mass = (1.0 - effective_confidence) * scenarios.shape[0]
    if tail_mass + 1e-12 < minimum_tail:
        raise ExpectedShortfallAllocationError(
            "scenario sample is too small for the requested confidence and tail minimum"
        )

    component_losses = -(scenarios * position_weights)
    portfolio_losses = component_losses.sum(axis=1)
    tail_weights, cutoff = _tail_weights(portfolio_losses, tail_mass)
    component_es = tail_weights @ component_losses / tail_mass
    expected_shortfall = float(tail_weights @ portfolio_losses / tail_mass)
    component_sum = float(component_es.sum())
    if expected_shortfall <= 0.0:
        raise ExpectedShortfallAllocationError(
            "tail Expected Shortfall must be positive to allocate risk shares"
        )
    reconciliation_error = abs(component_sum - expected_shortfall) / expected_shortfall
    if reconciliation_error > 1e-12:
        raise ExpectedShortfallAllocationError(
            "component Expected Shortfall does not reconcile"
        )

    absolute_components = np.abs(component_es)
    absolute_total = float(absolute_components.sum())
    if absolute_total <= 0.0:
        raise ExpectedShortfallAllocationError(
            "absolute component Expected Shortfall must be positive"
        )
    absolute_shares = absolute_components / absolute_total
    signed_shares = component_es / expected_shortfall
    hhi = float(absolute_shares @ absolute_shares)

    allocations = tuple(
        sorted(
            (
                AssetExpectedShortfallAllocation(
                    asset_ref=hashlib.sha256(asset_id.encode("utf-8")).hexdigest()[:16],
                    component_expected_shortfall=float(component_es[index]),
                    signed_risk_share=float(signed_shares[index]),
                    absolute_risk_budget_share=float(absolute_shares[index]),
                )
                for index, asset_id in enumerate(assets)
            ),
            key=lambda allocation: allocation.asset_ref,
        )
    )
    return ExpectedShortfallAllocation(
        schema_version="1.0",
        confidence=effective_confidence,
        scenario_count=int(scenarios.shape[0]),
        effective_tail_scenarios=tail_mass,
        value_at_risk_cutoff=cutoff,
        expected_shortfall=expected_shortfall,
        component_sum=component_sum,
        reconciliation_relative_error=reconciliation_error,
        max_absolute_risk_share=float(absolute_shares.max()),
        risk_budget_hhi=hhi,
        effective_risk_positions=1.0 / hhi,
        negative_component_count=int(np.count_nonzero(component_es < 0.0)),
        allocations=allocations,
    )
