# Expected Shortfall tail-calibration audit

Expected Shortfall (ES, also called CVaR) estimates the average loss beyond a
Value-at-Risk threshold. A VaR coverage test only checks how often that threshold
is crossed. It can pass while losses inside the tail are systematically more
severe than the reported ES.

`src.es_tail_audit` adds a model-independent release gate for that separate
failure mode. It evaluates an out-of-sample sequence of realized losses and paired
VaR/ES forecasts without loading the risk model.

## Identification moment

Let \(L_t\) be positive loss, \(q_t\) the VaR forecast, \(e_t\) the ES forecast,
and \(\tau = 1-c\) the tail probability for confidence \(c\). The audit computes

\[
V_t = \frac{(L_t-q_t)\mathbf{1}\{L_t>q_t\}}{\tau}-(e_t-q_t).
\]

For a correctly specified VaR/ES pair, the expected value of \(V_t\) is zero.
A positive mean says the realized tail excess is worse than the ES tail buffer;
a negative mean indicates conservative ES forecasts. This is a joint VaR–ES
identification condition: ES is not treated as independently elicitable.

The audit estimates uncertainty in the mean with a Newey–West long-run variance
and Bartlett weights. `hac_lag` must be at least `forecast_horizon - 1`, so an
overlapping multi-period forecast cannot silently use an IID standard error. The
report includes the one-sided normal-reference p-value for underestimation.

Release policy combines statistical and practical evidence:

- a material positive relative tail bias with a one-sided p-value at or below the
  significance threshold;
- an absolute maximum relative tail-bias budget;
- minimum total, expected-tail, and realized-tail evidence;
- a minimum mean ES-minus-VaR tail buffer;
- a non-degenerate HAC variance estimate.

Relative tail bias is the mean identification residual divided by the mean
ES-minus-VaR buffer. Thresholds therefore remain interpretable across portfolios
with different return scales.

## Artifact contract

```json
{
  "schema_version": 1,
  "audit_id": "book-a.2026-09",
  "model_sha256": "<64 lowercase hex characters>",
  "forecast_config_sha256": "<64 lowercase hex characters>",
  "created_at": "2026-09-27T02:59:30Z",
  "confidence": 0.95,
  "forecast_horizon": 5,
  "hac_lag": 4,
  "observations": [
    {
      "timestamp": "2026-09-10T00:00:00Z",
      "loss": 0.0,
      "value_at_risk": 0.02,
      "expected_shortfall": 0.04
    }
  ]
}
```

Production artifacts require substantially more observations than this abbreviated
shape example. Losses use a positive-loss convention. VaR must be non-negative and
ES must be strictly greater than VaR for every observation. Observations must be
out-of-sample, timezone-aware, unique, and strictly chronological.

Run the audit:

```bash
python -m src.es_tail_audit artifacts/es-forecasts.json \
  --output artifacts/es-tail-audit.json
```

| Exit code | Meaning |
| --- | --- |
| `0` | Evidence accepted |
| `2` | Well-formed evidence rejected by policy |
| `3` | Malformed evidence or I/O failure |

The output is written atomically. It exposes aggregate diagnostics, bounded reason
codes, and canonical artifact/policy hashes. Raw audit IDs, timestamps, losses,
VaR forecasts, and ES forecasts are not copied into the report.

## Fail-closed validation

The parser rejects duplicate JSON keys, unknown fields, non-finite or excessive
values, invalid forecast ordering, naive/duplicate/out-of-order timestamps, stale
or future evidence, observations that postdate evidence creation, unsupported
horizons, insufficient HAC lag, and byte or row budget overruns. Equivalent
timezone representations produce the same canonical artifact digest.

## Operational limitations

- The normal-reference HAC test is asymptotic. Small tail samples and heavy
  dependence can produce poor finite-sample calibration.
- The moment jointly assesses VaR and ES. A biased VaR forecast can make the ES
  diagnostic fail even when the reported ES happens to match a sample tail mean.
- The artifact producer remains a trust boundary. SHA-256 binds evidence but does
  not prove that forecasts were generated before outcomes or that omitted periods
  do not exist.
- Structural breaks, changing positions, P&L attribution, liquidity horizons,
  backfilled prices, and multiple portfolios/confidence levels require additional
  controls. A separate multiplicity policy is required when many audits are used
  for one release decision.
- Acceptance is evidence about historical tail calibration, not a guarantee of
  future losses, capital adequacy, or regulatory approval.

The next integration step is to emit this artifact from a rolling, genuinely
out-of-sample risk-forecast adapter and bind the accepted digest into the API risk
report before model promotion.
