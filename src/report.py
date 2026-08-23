from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def render_report(result: dict[str, Any], output: str | Path) -> Path:
    risk_rows = "".join(
        f"<tr><td>{html.escape(str(item['method']))}</td><td>{item['confidence']:.0%}</td><td>{item['var']:.4%}</td><td>{item['cvar']:.4%}</td></tr>"
        for item in result["risk_measures"]
    )
    contrib_rows = "".join(
        f"<tr><td>{html.escape(asset)}</td><td>{float(value):.4%}</td></tr>"
        for asset, value in sorted(result["historical_var_contributions"].items(), key=lambda kv: abs(kv[1]), reverse=True)
    )
    stress_rows = "".join(
        f"<tr><td>{html.escape(str(item['scenario']))}</td><td>{float(item['portfolio_return']):.2%}</td><td>{html.escape(str(item['description']))}</td></tr>"
        for item in result.get("stress_tests", [])
    ) or '<tr><td colspan="3">No factor stress model supplied</td></tr>'
    body = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Risk Report</title><style>
body{{background:#0a0f1e;color:#eef3ff;font-family:Inter,system-ui,sans-serif;margin:0;padding:32px}}main{{max-width:1100px;margin:auto}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}.card{{background:#121b30;border:1px solid #263554;border-radius:14px;padding:18px}}
.big{{font-size:26px;font-weight:800}}.muted{{color:#9eabc4}}table{{width:100%;border-collapse:collapse;background:#121b30;margin:10px 0 28px}}
th,td{{padding:11px;border-bottom:1px solid #263554;text-align:left}}th{{color:#a8bfff}}@media(max-width:850px){{.grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main><p class="muted">Synthetic/public-data reference analytics</p><h1>Portfolio Risk Engine</h1>
<div class="grid"><div class="card"><div class="big">{result['annualized_return']:.1%}</div><div>Ann. return</div></div>
<div class="card"><div class="big">{result['annualized_volatility']:.1%}</div><div>Ann. volatility</div></div>
<div class="card"><div class="big">{result['sharpe_zero_rf']:.2f}</div><div>Sharpe (rf=0)</div></div>
<div class="card"><div class="big">{result['max_drawdown']:.1%}</div><div>Max drawdown</div></div></div>
<h2>VaR / CVaR comparison</h2><table><thead><tr><th>Method</th><th>Confidence</th><th>VaR</th><th>CVaR</th></tr></thead><tbody>{risk_rows}</tbody></table>
<h2>Historical VaR contributions</h2><table><thead><tr><th>Asset</th><th>Contribution</th></tr></thead><tbody>{contrib_rows}</tbody></table>
<h2>Factor stress scenarios</h2><table><thead><tr><th>Scenario</th><th>Portfolio return</th><th>Description</th></tr></thead><tbody>{stress_rows}</tbody></table>
</main></body></html>"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path
