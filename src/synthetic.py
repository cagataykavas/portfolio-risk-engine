from __future__ import annotations

import numpy as np
import pandas as pd

ASSETS = ("global_equity", "tech_equity", "investment_grade", "gold", "usd_index")


def synthetic_returns(rows: int = 900, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    means = np.array([0.00035, 0.00045, 0.00012, 0.00018, 0.00005])
    vols = np.array([0.011, 0.016, 0.0045, 0.009, 0.004])
    corr = np.array(
        [
            [1.00, 0.78, 0.22, -0.12, -0.25],
            [0.78, 1.00, 0.15, -0.10, -0.30],
            [0.22, 0.15, 1.00, 0.05, -0.05],
            [-0.12, -0.10, 0.05, 1.00, -0.18],
            [-0.25, -0.30, -0.05, -0.18, 1.00],
        ]
    )
    covariance = np.outer(vols, vols) * corr
    values = rng.multivariate_normal(means, covariance, size=rows)

    # Insert a short but plausible stress window so drawdown/stress behavior is visible.
    if rows >= 120:
        values[rows // 2 : rows // 2 + 8, 0] -= 0.025
        values[rows // 2 : rows // 2 + 8, 1] -= 0.038
        values[rows // 2 : rows // 2 + 8, 2] -= 0.006
        values[rows // 2 : rows // 2 + 8, 3] += 0.012
        values[rows // 2 : rows // 2 + 8, 4] += 0.007

    index = pd.date_range("2022-01-03", periods=rows, freq="B")
    return pd.DataFrame(values, columns=ASSETS, index=index)


FACTOR_LOADINGS = {
    "global_equity": {"equity": 1.0, "rates": -1.4, "credit": 0.35, "commodity": 0.05, "fx": -0.25},
    "tech_equity": {"equity": 1.2, "rates": -2.2, "credit": 0.25, "commodity": 0.0, "fx": -0.35},
    "investment_grade": {"equity": 0.1, "rates": -3.5, "credit": 0.75, "commodity": 0.0, "fx": -0.05},
    "gold": {"equity": -0.1, "rates": -0.8, "credit": -0.1, "commodity": 0.85, "fx": -0.45},
    "usd_index": {"equity": -0.2, "rates": 0.5, "credit": -0.1, "commodity": -0.15, "fx": 1.0},
}
