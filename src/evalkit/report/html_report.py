"""A single self-contained HTML file: no JavaScript, no external assets."""

from __future__ import annotations

from html import escape

from evalkit.compare import Change, Comparison
from evalkit.results import CaseResult, RunResult

_CSS = """
:root {
  --bg: #f7f7f8; --panel: #ffffff; --text: #1d1d22; --muted: #62626e;
  --border: #e2e2e8; --pass: #1a7f45; --pass-bg: #e3f4ea; --fail: #b42318;
  --fail-bg: #fde8e6; --code: #f1f1f4;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #131316; --panel: #1c1c21; --text: #ececf1; --muted: #a0a0ad;
    --border: #2e2e36; --pass: #5fd394; --pass-bg: #163824; --fail: #ff8a7a;
    --fail-bg: #3d1a17; --code: #25252c;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 16px; background: var(--bg); color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 12px; }
.muted { color: var(--muted); }
.panel {
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 20px;
}
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;
  margin-top: 20px; }
.stat .value { font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; }
.stat .label { color: var(--muted); font-size: 13px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-weight: 600;
  font-size: 13px; }
.pass { color: var(--pass); background: var(--pass-bg); }
.fail { color: var(--fail); background: var(--fail-bg); }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border);
  vertical-align: top; }
th { color: var(--muted); font-weight: 500; font-size: 13px; }
td.num { font-variant-numeric: tabular-nums; }
details.case { border: 1px solid var(--border); border-radius: 10px; background: var(--panel);
  margin-bottom: 8px; }
details.case > summary { cursor: pointer; padding: 10px 16px; display: flex; gap: 12px;
  align-items: center; flex-wrap: wrap; }
details.case > summary .id { font-weight: 600; }
details.case .body { padding: 0 16px 16px; }
pre { background: var(--code); padding: 10px 12px; border-radius: 6px; white-space: pre-wrap;
  word-break: break-word; margin: 4px 0 12px; font-size: 13px; }
.chip { font-size: 12px; padding: 1px 8px; border-radius: 6px; }
.tag { font-size: 12px; color: var(--muted); }
.label { font-size: 13px; color: var(--muted); }
.neutral { color: var(--muted); background: var(--code); }
"""

_CHANGE_STYLE = {
    Change.REGRESSED: ("fail", "REGRESSED"),
    Change.FIXED: ("pass", "FIXED"),
    Change.NEW: ("neutral", "NEW"),
    Change.REMOVED: ("neutral", "REMOVED"),
}


def _badge(passed: bool, yes: str = "PASS", no: str = "FAIL") -> str:
    cls = "pass" if passed else "fail"
    return f'<span class="badge {cls}">{yes if passed else no}</span>'


def _pre(label: str, text: str | None) -> str:
    if text is None:
        return ""
    return f'<div class="label">{escape(label)}</div><pre>{escape(text)}</pre>'


def _case(case: CaseResult) -> str:
    chips = "".join(
        f'<span class="chip {"pass" if s.passed else "fail"}">{escape(s.scorer)} '
        f"{s.score:.2f}</span>"
        for s in case.scores
    )
    tags = " ".join(f'<span class="tag">#{escape(t)}</span>' for t in case.tags)
    rows = "".join(
        f"<tr><td>{escape(s.scorer)}</td><td class='num'>{s.score:.3f}</td>"
        f"<td>{_badge(s.passed)}</td><td>{escape(s.reason)}</td></tr>"
        for s in case.scores
    )
    scores_table = (
        "<div class='table-wrap'><table><thead><tr><th>Scorer</th><th>Score</th>"
        f"<th>Result</th><th>Reason</th></tr></thead><tbody>{rows}</tbody></table></div>"
        if rows
        else ""
    )
    latency = "cached" if case.cached else f"{case.latency_s * 1000:.0f} ms"
    return f"""
<details class="case"{"" if case.passed else " open"}>
  <summary>{_badge(case.passed)}<span class="id">{escape(case.case_id)}</span>{chips}
    {tags}<span class="muted">{latency}</span></summary>
  <div class="body">
    {_pre("Error", case.error)}
    {_pre("Prompt", case.prompt)}
    {_pre("Expected", case.expected)}
    {_pre("Output", case.output)}
    {scores_table}
  </div>
</details>"""


def _score(value: float | None) -> str:
    return "&ndash;" if value is None else f"{value:.2f}"


def _comparison(comparison: Comparison | None) -> str:
    if comparison is None:
        return ""
    delta = comparison.pass_rate_delta * 100
    headline = (
        f"{comparison.baseline_pass_rate:.1%} → {comparison.current_pass_rate:.1%} "
        f"({delta:+.1f} pts) · {comparison.regressions} regressed · {comparison.fixes} fixed"
    )
    rows = "".join(
        f"<tr><td><span class='badge {_CHANGE_STYLE[c.change][0]}'>"
        f"{_CHANGE_STYLE[c.change][1]}</span></td><td>{escape(c.case_id)}</td>"
        f"<td class='num'>{_score(c.baseline_score)} → {_score(c.current_score)}</td>"
        f"<td>{escape(c.detail)}</td></tr>"
        for c in comparison.changes
        if c.change in _CHANGE_STYLE
    )
    table = (
        "<div class='table-wrap'><table><thead><tr><th>Change</th><th>Case</th>"
        f"<th>Mean score</th><th>Detail</th></tr></thead><tbody>{rows}</tbody></table></div>"
        if rows
        else "<div class='muted'>No case changed status.</div>"
    )
    return f"""
  <h2>Compared with baseline</h2>
  <div class="panel"><div style="margin-bottom:8px">{headline}</div>{table}</div>"""


def render_html(result: RunResult, comparison: Comparison | None = None) -> str:
    summary_rows = "".join(
        f"<tr><td>{escape(s.scorer)}</td><td class='num'>{s.passed}/{s.count}</td>"
        f"<td class='num'>{s.mean_score:.3f}</td></tr>"
        for s in result.scorer_summaries()
    )
    model = f" · {escape(result.model)}" if result.model else ""
    stats = [
        (f"{result.pass_rate:.1%}", "pass rate"),
        (f"{result.threshold:.0%}", "threshold"),
        (f"{result.passed_count}/{result.total}", "cases passed"),
        (str(result.error_count), "errors"),
        (f"{result.duration_s:.2f}s", "duration"),
    ]
    stat_html = "".join(
        f'<div class="panel stat"><div class="value">{v}</div><div class="label">{lbl}</div></div>'
        for v, lbl in stats
    )
    cases = "".join(_case(c) for c in result.cases)
    gate = result.passed and not (comparison is not None and comparison.regressions)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>evalkit · {escape(result.suite)}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
  <h1>{escape(result.suite)} {_badge(gate, "GATE PASSED", "GATE FAILED")}</h1>
  <div class="muted">{escape(result.provider)}{model} ·
    {escape(result.started_at.strftime("%Y-%m-%d %H:%M:%S %Z"))}</div>
  <div class="stats">{stat_html}</div>
  <h2>Scorers</h2>
  <div class="panel table-wrap"><table>
    <thead><tr><th>Scorer</th><th>Passed</th><th>Mean score</th></tr></thead>
    <tbody>{summary_rows}</tbody>
  </table></div>{_comparison(comparison)}
  <h2>Cases</h2>
  {cases}
</main>
</body>
</html>
"""
