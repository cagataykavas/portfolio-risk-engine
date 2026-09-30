from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .risk import normalize_weights
from .synthetic import synthetic_returns

ESTIMATORS = ("sample", "diagonal_shrinkage", "ewma")
SCHEMA_VERSION = "covariance-estimator-benchmark/v1"


@dataclass(frozen=True)
class BenchmarkConfig:
    window: int = 252
    selection_observations: int = 126
    evaluation_observations: int = 126
    shrinkage: float = 0.25
    ewma_decay: float = 0.97
    variance_floor: float = 1e-12
    max_evaluation_regret: float = 0.05

    def validate(self, asset_count: int) -> None:
        minimum_window = max(20, asset_count + 2)
        if self.window < minimum_window:
            raise ValueError(f"window must be at least {minimum_window}")
        if self.selection_observations < 20 or self.evaluation_observations < 20:
            raise ValueError(
                "selection and evaluation periods must each contain at least 20 observations"
            )
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        if not 0.0 < self.ewma_decay < 1.0:
            raise ValueError("ewma_decay must be in (0, 1)")
        if not math.isfinite(self.variance_floor) or self.variance_floor <= 0.0:
            raise ValueError("variance_floor must be finite and positive")
        if not math.isfinite(self.max_evaluation_regret) or self.max_evaluation_regret < 0.0:
            raise ValueError("max_evaluation_regret must be finite and non-negative")


def _validate_inputs(
    returns: pd.DataFrame, weights: np.ndarray, config: BenchmarkConfig
) -> tuple[pd.DataFrame, np.ndarray]:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame")
    if returns.shape[1] > 128 or returns.shape[0] > 100_000:
        raise ValueError("returns exceed the 128-asset or 100,000-row resource budget")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns index must be a DatetimeIndex")
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError("returns index must be strictly increasing and unique")
    if returns.columns.has_duplicates or any(
        not isinstance(column, str) or not column for column in returns.columns
    ):
        raise ValueError("asset names must be unique, non-empty strings")
    try:
        values = returns.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must contain only numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("returns must not contain missing or non-finite values")

    raw_weights = np.asarray(weights, dtype=float)
    if raw_weights.ndim != 1 or len(raw_weights) != returns.shape[1]:
        raise ValueError("weights must contain exactly one value per asset")
    if not np.isfinite(raw_weights).all():
        raise ValueError("weights must be finite")
    normalized = normalize_weights(raw_weights)
    config.validate(returns.shape[1])
    required = config.window + config.selection_observations + config.evaluation_observations
    if len(returns) < required:
        raise ValueError(f"at least {required} return observations are required")
    return returns.astype(float, copy=False), normalized


def _estimate_covariance(window: np.ndarray, estimator: str, config: BenchmarkConfig) -> np.ndarray:
    if estimator == "sample":
        covariance = np.cov(window, rowvar=False, ddof=1)
    elif estimator == "diagonal_shrinkage":
        sample = np.cov(window, rowvar=False, ddof=1)
        covariance = (1.0 - config.shrinkage) * sample + config.shrinkage * np.diag(np.diag(sample))
    elif estimator == "ewma":
        age = np.arange(len(window) - 1, -1, -1, dtype=float)
        observation_weights = np.power(config.ewma_decay, age)
        observation_weights /= observation_weights.sum()
        mean = observation_weights @ window
        centered = window - mean
        covariance = (centered * observation_weights[:, None]).T @ centered
    else:
        raise ValueError(f"unknown estimator: {estimator}")
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    return (covariance + covariance.T) / 2.0


def walk_forward_forecasts(
    returns: pd.DataFrame,
    weights: np.ndarray,
    config: BenchmarkConfig | None = None,
) -> pd.DataFrame:
    """Build one-step variance forecasts using only observations before each origin."""
    config = config or BenchmarkConfig()
    returns, normalized_weights = _validate_inputs(returns, weights, config)
    values = returns.to_numpy()
    first_origin = len(returns) - config.selection_observations - config.evaluation_observations
    records: list[dict[str, Any]] = []

    for origin in range(first_origin, len(returns)):
        training = values[origin - config.window : origin]
        realized_return = float(values[origin] @ normalized_weights)
        split = (
            "selection" if origin < first_origin + config.selection_observations else "evaluation"
        )
        for estimator in ESTIMATORS:
            covariance = _estimate_covariance(training, estimator, config)
            variance = max(
                float(normalized_weights @ covariance @ normalized_weights), config.variance_floor
            )
            gaussian_nll = 0.5 * (math.log(variance) + realized_return**2 / variance)
            records.append(
                {
                    "origin": returns.index[origin],
                    "split": split,
                    "estimator": estimator,
                    "forecast_variance": variance,
                    "realized_return": realized_return,
                    "gaussian_nll": gaussian_nll,
                    "squared_error": (variance - realized_return**2) ** 2,
                }
            )
    return pd.DataFrame.from_records(records)


def _input_digest(returns: pd.DataFrame, weights: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(returns.columns).encode())
    digest.update(returns.index.asi8.tobytes())
    digest.update(np.ascontiguousarray(returns.to_numpy(dtype="<f8")).tobytes())
    digest.update(np.ascontiguousarray(weights.astype("<f8")).tobytes())
    return digest.hexdigest()


def benchmark_covariance_estimators(
    returns: pd.DataFrame,
    weights: np.ndarray,
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    config = config or BenchmarkConfig()
    returns, normalized_weights = _validate_inputs(returns, weights, config)
    forecasts = walk_forward_forecasts(returns, normalized_weights, config)
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for estimator in ESTIMATORS:
        metrics[estimator] = {}
        for split in ("selection", "evaluation"):
            subset = forecasts[
                (forecasts["estimator"] == estimator) & (forecasts["split"] == split)
            ]
            metrics[estimator][split] = {
                "mean_gaussian_nll": float(subset["gaussian_nll"].mean()),
                "variance_mse": float(subset["squared_error"].mean()),
                "mean_forecast_variance": float(subset["forecast_variance"].mean()),
            }

    selected = min(
        ESTIMATORS, key=lambda name: (metrics[name]["selection"]["mean_gaussian_nll"], name)
    )
    evaluation_scores = {
        name: metrics[name]["evaluation"]["mean_gaussian_nll"] for name in ESTIMATORS
    }
    hindsight_best = min(ESTIMATORS, key=lambda name: (evaluation_scores[name], name))
    regret = evaluation_scores[selected] - evaluation_scores[hindsight_best]
    sample_delta = evaluation_scores[selected] - evaluation_scores["sample"]
    accepted = regret <= config.max_evaluation_regret

    start = len(returns) - config.selection_observations - config.evaluation_observations
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted" if accepted else "rejected",
        "dataset": {
            "assets": list(returns.columns),
            "observations": len(returns),
            "benchmark_start": returns.index[start].isoformat(),
            "benchmark_end": returns.index[-1].isoformat(),
            "input_sha256": _input_digest(returns, normalized_weights),
        },
        "policy": asdict(config),
        "selection": {
            "criterion": "lowest mean one-step Gaussian negative log-likelihood",
            "selected_estimator": selected,
            "evaluation_hindsight_best": hindsight_best,
        },
        "metrics": metrics,
        "gate": {
            "evaluation_regret": regret,
            "max_evaluation_regret": config.max_evaluation_regret,
            "selected_minus_sample_evaluation_nll": sample_delta,
            "passed": accepted,
        },
        "limitations": [
            "The benchmark evaluates one-step portfolio variance, not full multivariate density calibration.",
            "Estimator hyperparameters are fixed by policy and are not optimized by this benchmark.",
            "A passing result does not establish performance under a future regime shift.",
        ],
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    report["evidence_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


def _atomic_write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Leakage-safe covariance estimator benchmark")
    parser.add_argument("--rows", type=int, default=650)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--selection-observations", type=int, default=126)
    parser.add_argument("--evaluation-observations", type=int, default=126)
    parser.add_argument("--max-evaluation-regret", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("artifacts/covariance-benchmark.json"))
    args = parser.parse_args(argv)

    try:
        returns = synthetic_returns(args.rows, args.seed)
        weights = np.array([0.30, 0.20, 0.25, 0.15, 0.10])
        config = BenchmarkConfig(
            window=args.window,
            selection_observations=args.selection_observations,
            evaluation_observations=args.evaluation_observations,
            max_evaluation_regret=args.max_evaluation_regret,
        )
        report = benchmark_covariance_estimators(returns, weights, config)
        _atomic_write_json(args.output, report)
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "malformed", "error": str(exc)}))
        return 2

    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_estimator": report["selection"]["selected_estimator"],
                "evaluation_regret": report["gate"]["evaluation_regret"],
                "evidence_sha256": report["evidence_sha256"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if report["gate"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
