"""Static HTML track-record page built from a signal log (no external assets)."""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import Any

import pandas as pd

from alpha_gen.live.signal_log import SignalRecord
from alpha_gen.live.track import MIN_PERIODS_FOR_RATIOS, track_record, track_summary

_CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--line:#e7e5e4;--card:#fff;--accent:#2563eb;
--good:#15803d;--bad:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#0c0a09;--fg:#f5f5f4;--muted:#a8a29e;
--line:#292524;--card:#1c1917;--accent:#60a5fa;--good:#4ade80;--bad:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px}h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:32px 0 8px}p.muted,.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;
margin:12px 0}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px}
.stat b{display:block;font-size:20px}.stat span{color:var(--muted);font-size:13px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
td,th{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right}
td:first-child,th:first-child{text-align:left;word-break:break-word}
.pos{color:var(--good)}.neg{color:var(--bad)}.scroll{overflow-x:auto}
svg{width:100%;height:auto}.badge{display:inline-block;padding:2px 8px;border-radius:99px;
border:1px solid var(--line);font-size:12px;color:var(--muted)}code{word-break:break-all}
"""


def _pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:+.2%}"


def _ratio(value: float, periods: int) -> str:
    if pd.isna(value):
        return f"n/a ({periods}/{MIN_PERIODS_FOR_RATIOS} periods)"
    return f"{value:+.2f}"


def _equity_svg(equity: pd.Series, width: int = 900, height: int = 220) -> str:
    if len(equity) < 2:
        return '<p class="muted">The equity curve appears after the second rebalance.</p>'
    values = [1.0, *equity.tolist()]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1e-9
    step = width / (len(values) - 1)
    points = " ".join(
        f"{i * step:.1f},{height - (v - lo) / span * (height - 20) - 10:.1f}"
        for i, v in enumerate(values)
    )
    base = height - (1.0 - lo) / span * (height - 20) - 10
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Equity curve">'
        f'<line x1="0" x2="{width}" y1="{base:.1f}" y2="{base:.1f}" stroke="var(--line)"/>'
        f'<polyline fill="none" stroke="var(--accent)" stroke-width="2" points="{points}"/></svg>'
    )


def _strategy_section(records: Sequence[SignalRecord], name: str) -> str:
    mine = [r for r in records if r.strategy == name]
    last = mine[-1]
    timeframe = str(last.metadata.get("timeframe", "1d"))
    record = track_record(mine, name)
    stats = track_summary(record, timeframe)
    description = html.escape(str(last.metadata.get("strategy", {}).get("description", "")))
    cells = [
        ("Total return", _pct(stats.get("total_return", float("nan")))),
        ("Sharpe (ann.)", _ratio(stats.get("sharpe", float("nan")), stats.get("periods", 0))),
        ("Max drawdown", _pct(stats.get("max_drawdown", float("nan")))),
        ("Costs paid", _pct(-stats.get("total_costs", float("nan")))),
        ("Funding", _pct(stats.get("total_funding", float("nan")))),
        ("Rebalances", str(len(mine))),
    ]
    stat_html = "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for k, v in cells)
    positions = sorted(last.weights.items(), key=lambda kv: -kv[1])
    rows = "".join(
        f'<tr><td>{html.escape(s)}</td><td class="{"pos" if w > 0 else "neg"}">{w:+.2%}</td></tr>'
        for s, w in positions
    )
    recent = record.tail(10).iloc[::-1]
    history = "".join(
        f"<tr><td>{ts:%Y-%m-%d %H:%M}</td><td>{_pct(r.gross)}</td><td>{_pct(-r.costs)}</td>"
        f"<td>{_pct(r.funding)}</td>"
        f'<td class="{"pos" if r.net >= 0 else "neg"}">{_pct(r.net)}</td></tr>'
        for ts, r in recent.iterrows()
    )
    return f"""
<section class="card">
  <h2>{html.escape(name)} <span class="badge">{html.escape(timeframe)} bars</span></h2>
  <p class="muted">{description}</p>
  <div class="stats">{stat_html}</div>
  {_equity_svg(record["equity"]) if not record.empty else _equity_svg(pd.Series(dtype=float))}
  <h2>Latest target weights <span class="badge">decided {html.escape(last.timestamp)}</span></h2>
  <div class="scroll"><table><thead><tr><th>Symbol</th><th>Weight</th></tr></thead>
  <tbody>{rows}</tbody></table></div>
  <h2>Recent realised P&amp;L</h2>
  <div class="scroll"><table><thead><tr><th>Period end</th><th>Gross</th><th>Costs</th>
  <th>Funding</th><th>Net</th></tr></thead>
  <tbody>{history or '<tr><td colspan="5">None yet</td></tr>'}</tbody></table></div>
</section>"""


def render_track_page(records: Sequence[SignalRecord], *, verified: bool, title: str) -> str:
    """Self-contained HTML page for every strategy in the log."""
    names = list(dict.fromkeys(r.strategy for r in records))
    head = records[-1].hash if records else "(empty log)"
    sections = "".join(_strategy_section(records, n) for n in names) or (
        '<p class="muted">No signals logged yet.</p>'
    )
    status = "verified" if verified else "FAILED verification"
    generated = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="muted">Paper trading, not investment advice. Every signal is published before the
returns it earns, in an append-only hash-chained log (hash chain {status}).
Generated {generated}.</p>
<p class="muted">Log head: <code>{html.escape(head)}</code></p>
{sections}
</main></body></html>
"""


def track_payload(records: Sequence[SignalRecord]) -> dict[str, Any]:
    """Machine-readable companion to the page."""
    out: dict[str, Any] = {"head": records[-1].hash if records else None, "strategies": {}}
    for name in dict.fromkeys(r.strategy for r in records):
        mine = [r for r in records if r.strategy == name]
        timeframe = str(mine[-1].metadata.get("timeframe", "1d"))
        record = track_record(mine, name)
        out["strategies"][name] = {
            "summary": track_summary(record, timeframe),
            "latest": {"timestamp": mine[-1].timestamp, "weights": mine[-1].weights},
            "equity": {str(k): float(v) for k, v in record["equity"].items()},
        }
    return out
