# Portfolio Risk Engine

A compact quantitative-risk toolkit covering portfolio returns, covariance estimation, historical and parametric VaR/CVaR, Monte Carlo stress scenarios, drawdown and marginal risk contribution.

The public baseline uses synthetic returns so it is fully reproducible and safe to publish.

## Methods

- portfolio return and volatility
- covariance shrinkage baseline
- historical VaR / CVaR
- Gaussian parametric VaR
- Monte Carlo scenario simulation
- maximum drawdown
- component risk contribution
- stress shocks and scenario P&L

## Quick start

```bash
pip install -r requirements.txt
python -m src.demo
```

## Engineering focus

Risk metrics are exposed as small testable functions rather than hidden inside a notebook. This makes the project easy to validate, reuse in APIs and later extend with factor models, options Greeks, Expected Shortfall backtesting or live market feeds.
