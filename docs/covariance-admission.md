# Covariance admission for portfolio simulation

Monte Carlo VaR and component-risk analytics assume that the supplied
covariance matrix is symmetric and positive semidefinite. In production,
missing data, pairwise estimation, rounding, short histories, or serialization
errors can make that assumption false. NumPy may warn and continue, silently
turning a data-quality problem into unstable risk numbers.

This repository now admits covariance evidence before simulation and performs a
bounded repair only when the change is demonstrably small.

## Checks and repair

The gate validates unique asset identities, matrix shape, finite values,
positive variances, observation count, and observation-to-asset ratio. It then:

1. measures asymmetry and uses the symmetric part for spectral analysis;
2. checks implied correlation bounds;
3. eigen-decomposes the matrix and floors small eigenvalues relative to the
   largest eigenvalue;
4. rescales the repaired matrix so every original marginal variance is exactly
   preserved;
5. measures relative Frobenius repair size, maximum correlation shift, and the
   repaired condition number;
6. rejects evidence when any governed budget is exceeded.

The engine passes only the admitted matrix to Monte Carlo simulation and
includes the redacted report in its risk response. A near-singular empirical
matrix can therefore be stabilized, while a materially indefinite or
under-sampled matrix fails closed.

## Standalone evidence

The CLI accepts strict JSON:

```json
{
  "schema_version": "1.0",
  "estimator_id": "sample-covariance-v1",
  "asset_ids": ["equity", "bonds"],
  "observation_count": 300,
  "covariance": [[0.04, 0.006], [0.006, 0.01]]
}
```

```bash
python -m src.covariance_admission covariance.json --output admission.json
```

Exit `0` means admitted, `2` means structurally valid evidence violated policy,
and `3` means malformed evidence. Input bytes and asset count are bounded;
duplicate JSON keys and non-finite constants are rejected. Reports contain
metrics and SHA-256 content identities, not the input or repaired matrix.

## Limitations

Eigenvalue flooring is a numerical safeguard, not a covariance estimator. It
does not solve non-stationarity, asynchronous prices, stale marks, outliers,
missing-not-at-random observations, or regime change. Sample adequacy thresholds
must be calibrated to horizon, frequency, and estimator. Material repair should
trigger data investigation rather than threshold relaxation.

The gate also does not establish that Gaussian Monte Carlo is an adequate tail
model. Heavy tails, volatility clustering, liquidity, concentration, and model
backtesting remain separate controls.

## Next step

Add walk-forward comparison of sample, shrinkage, and EWMA covariance forecasts
against realized portfolio variance, with estimator selection fixed before the
held-out period.
