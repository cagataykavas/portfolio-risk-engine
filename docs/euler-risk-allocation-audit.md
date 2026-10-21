# Euler component-risk allocation audit

A portfolio can satisfy total VaR or volatility limits while one position silently owns most of the
risk budget. Weight diversification is not risk diversification: correlations, volatilities, and
short hedges determine each position's marginal contribution.

`src.risk_allocation_audit` provides a fail-closed admission check for component-volatility evidence
before it is published to a risk report or used for a risk-budget decision.

## Mathematics

For weights $w$, covariance $\Sigma$, and portfolio volatility
$\sigma_p = \sqrt{w^T\Sigma w}$, the audit computes:

$$
\text{MRC}_i = \frac{(\Sigma w)_i}{\sigma_p}, \qquad
\text{CRC}_i = w_i\,\text{MRC}_i.
$$

Because volatility is homogeneous of degree one, valid Euler allocation satisfies
$\sum_i \text{CRC}_i = \sigma_p$. The implementation independently estimates every marginal with a
central finite difference and gates the worst analytic-versus-numeric relative error.

Concentration uses absolute component risk, so a negative hedge contribution is retained rather
than erased:

- maximum absolute component-risk budget share;
- Herfindahl index over absolute risk shares; and
- effective risk positions, $1 / \text{HHI}$.

The gate also checks net exposure, gross leverage, evidence freshness, and future-clock skew.

## Artifact contract

```json
{
  "generated_at": "2026-10-01T11:00:00Z",
  "portfolio_id": "book-alpha",
  "risk_model_digest": "<lowercase SHA-256>",
  "covariance_snapshot_digest": "<lowercase SHA-256>",
  "positions": [
    {"asset_id": "asset-a", "weight": 0.6},
    {"asset_id": "asset-b", "weight": 0.4}
  ],
  "covariance": [[0.04, 0.01], [0.01, 0.02]]
}
```

Position order defines covariance row/column order. Canonical evidence reorders positions and the
matrix together, so semantically identical permutations produce the same digest. Reports hash the
portfolio and asset identifiers and never reproduce weights or covariance entries.

```bash
python -m src.risk_allocation_audit allocation.json \
  --now 2026-10-01T11:00:30Z \
  --output allocation-report.json
```

Exit codes are `0` for accepted evidence, `2` for malformed evidence, and `3` for a well-formed
policy rejection. JSON writes are atomic. Duplicate fields, non-finite numbers, duplicate asset IDs,
matrix shape errors, non-positive portfolio variance, asymmetric covariance, and oversized inputs
fail closed.

## Trust boundary and limitations

This audit does not estimate covariance or certify that the supplied matrix is statistically sound.
Run the repository's covariance admission and leakage-safe estimator benchmark upstream. The caller
must bind truthful immutable digests for the risk model and covariance snapshot; hashes identify
content but do not authenticate the producer.

Volatility allocation is local and linear in the covariance model. It does not capture option
convexity, liquidity, jump/default loss, nonlinear scenario interactions, or Expected Shortfall
allocation. Absolute risk shares are a transparent concentration policy, not a universal regulatory
limit. Production thresholds must be calibrated by book mandate and asset class.
