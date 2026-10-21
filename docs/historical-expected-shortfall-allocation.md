# Historical Expected Shortfall allocation

Covariance-driven Euler allocation explains local volatility, but it cannot identify which assets
drive losses in a skewed or fat-tailed scenario set. `src.expected_shortfall_allocation` attributes
historical Expected Shortfall (ES) directly from observed or simulated joint return scenarios.

For scenario $s$, position $i$, return $r_{s,i}$, and portfolio weight $w_i$, component loss is
$L_{s,i}=-w_i r_{s,i}$. The portfolio loss is $L_s=\sum_i L_{s,i}$. At confidence $c$, the module
selects tail mass $(1-c)N$ and computes:

$$
ES_c = \frac{1}{(1-c)N}\sum_s q_s L_s, \qquad
CES_{c,i} = \frac{1}{(1-c)N}\sum_s q_s L_{s,i}.
$$

The tail weight $q_s$ is one above the empirical VaR boundary, zero below it, and fractional at the
boundary. All observations tied at that boundary receive the same fractional weight. This makes the
result independent of scenario input order and guarantees $\sum_i CES_{c,i}=ES_c$ without dropping
part of a tied loss bucket.

```python
from src.expected_shortfall_allocation import allocate_historical_expected_shortfall

report = allocate_historical_expected_shortfall(
    asset_ids=("global-equity", "credit", "rates-hedge"),
    weights=(0.55, 0.35, -0.10),
    scenario_returns=return_matrix,
    confidence=0.975,
    min_effective_tail_scenarios=20,
)
```

The returned report includes signed component ES, absolute risk-budget shares, HHI/effective risk
positions, and an explicit reconciliation error. Negative hedge contributions remain negative.
Asset identifiers are replaced with stable SHA-256 references, and scenario rows are never copied
into the report. Inputs fail closed on duplicate or malformed identifiers, shape mismatches,
non-finite values, unsupported sizes, an inadequate effective tail sample, or non-positive tail ES.

## Trust boundary and limitations

This utility allocates the supplied scenario distribution; it does not certify scenario lineage,
sampling quality, representativeness, independence, liquidity, or executable hedge capacity. ES is
still sensitive to confidence, window choice, regime coverage, and sparse tails. The minimum tail
mass is a deterministic sufficiency gate, not a statistical validation claim. Production callers
should bind the return matrix to immutable, authenticated provenance and calibrate confidence and
sample thresholds by book mandate.
