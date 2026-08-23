from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import PortfolioRiskEngine, PortfolioSpec
from .report import render_report
from .synthetic import FACTOR_LOADINGS, synthetic_returns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Portfolio risk analytics reference project")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--rows", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mc-scenarios", type=int, default=30_000)
    parser.add_argument("--json-output", type=Path, default=Path("artifacts/risk.json"))
    parser.add_argument("--html-output", type=Path, default=Path("artifacts/risk.html"))
    args = parser.parse_args(argv)

    returns = synthetic_returns(args.rows, args.seed)
    spec = PortfolioSpec(tuple(returns.columns), (0.30, 0.20, 0.25, 0.15, 0.10))
    result = PortfolioRiskEngine().analyze(
        returns,
        spec,
        confidence=args.confidence,
        monte_carlo_scenarios=args.mc_scenarios,
        seed=args.seed,
        factor_loadings=FACTOR_LOADINGS,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    render_report(result, args.html_output)
    print(json.dumps({
        "historical_var": result["risk_measures"][0]["var"],
        "monte_carlo_var": result["risk_measures"][2]["var"],
        "max_drawdown": result["max_drawdown"],
        "json": str(args.json_output),
        "html": str(args.html_output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
