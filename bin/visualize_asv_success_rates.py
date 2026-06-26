#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
from pathlib import Path

FRAMEWORK_ORDER = ["numpy", "scipy", "sparse", "pytorch"]


@dataclass(frozen=True)
class SuccessRow:
    benchmark: str
    framework: str
    finished: int
    total: int


def _framework_sort_key(framework: str) -> tuple[int, str]:
    try:
        return (FRAMEWORK_ORDER.index(framework), framework)
    except ValueError:
        return (len(FRAMEWORK_ORDER), framework)


def _read_time_rows(path: Path) -> list[SuccessRow]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record["metric"] != "time":
                continue
            rows.append(
                SuccessRow(
                    benchmark=record["benchmark"],
                    framework=record["framework"],
                    finished=int(record["finished_datasets"]),
                    total=int(record["total_datasets"]),
                )
            )
    return rows


def _status(row: SuccessRow | None) -> tuple[str, str, str]:
    if row is None:
        return "missing", "Not run", "No ASV runtime row"
    if row.finished == 0:
        return "failed", "No successful datasets", f"0/{row.total}"
    if row.finished == row.total:
        return "complete", "Ran all datasets", f"{row.finished}/{row.total}"
    return "partial", "Ran partially", f"{row.finished}/{row.total}"


def _cell(row: SuccessRow | None) -> str:
    class_name, label, count = _status(row)
    if row is None:
        return f'<td class="{class_name}"><strong>{escape(label)}</strong></td>'
    return (
        f'<td class="{class_name}">'
        f'<span class="status">{escape(label)}</span>'
        f"<strong>{escape(count)}</strong>"
        "</td>"
    )


def _benchmark_summary(rows: list[SuccessRow], frameworks: list[str]) -> str:
    benchmarks = sorted({row.benchmark for row in rows})
    lookup = {(row.benchmark, row.framework): row for row in rows}
    total = len(benchmarks)
    parts = ['<section class="summary">']
    for framework in frameworks:
        ran_any = 0
        ran_all = 0
        for benchmark in benchmarks:
            row = lookup.get((benchmark, framework))
            if row is None:
                continue
            if row.finished > 0:
                ran_any += 1
            if row.finished == row.total:
                ran_all += 1
        any_percent = 100 * ran_any / total if total else 0
        all_percent = 100 * ran_all / total if total else 0
        parts.append(
            '<div class="summary-item">'
            f'<span class="summary-framework">{escape(framework)}</span>'
            '<div class="summary-metric">'
            "<span>Ran at least one dataset</span>"
            f"<strong>{any_percent:.1f}%</strong>"
            f"<em>{ran_any}/{total} benchmarks</em>"
            "</div>"
            '<div class="summary-metric">'
            "<span>Ran every dataset</span>"
            f"<strong>{all_percent:.1f}%</strong>"
            f"<em>{ran_all}/{total} benchmarks</em>"
            "</div>"
            "</div>"
        )
    parts.append("</section>")
    return "\n".join(parts)


def _table(rows: list[SuccessRow], frameworks: list[str]) -> str:
    benchmarks = sorted({row.benchmark for row in rows})
    lookup = {(row.benchmark, row.framework): row for row in rows}
    header = "".join(f"<th>{escape(framework)}</th>" for framework in frameworks)
    parts = [
        '<section class="table-wrap">',
        "<table>",
        f"<thead><tr><th>Benchmark</th>{header}</tr></thead>",
        "<tbody>",
    ]
    for benchmark in benchmarks:
        cells = "".join(
            _cell(lookup.get((benchmark, framework))) for framework in frameworks
        )
        parts.append(f"<tr><th>{escape(benchmark)}</th>{cells}</tr>")
    parts.extend(["</tbody>", "</table>", "</section>"])
    return "\n".join(parts)


def _write_html(rows: list[SuccessRow], output: Path) -> None:
    frameworks = sorted({row.framework for row in rows}, key=_framework_sort_key)
    total_benchmarks = len({row.benchmark for row in rows})
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SAPS ASV Runtime Success Rates</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18212f;
      --muted: #5f6f84;
      --line: #d8dee8;
      --band: #f6f8fb;
      --complete: #d9eee2;
      --partial: #f5e8c2;
      --failed: #f3d3d0;
      --missing: #eef1f5;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: white;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 24px 52px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 10px;
      margin: 22px 0 24px;
    }}
    .summary-item {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--band);
      padding: 12px;
    }}
    .summary-framework {{
      display: block;
      font-weight: 650;
      margin-bottom: 10px;
    }}
    .summary-metric {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 2px 10px;
      align-items: baseline;
      padding-top: 8px;
      border-top: 1px solid var(--line);
    }}
    .summary-metric + .summary-metric {{
      margin-top: 8px;
    }}
    .summary-metric span,
    .summary-metric em {{
      color: var(--muted);
      font-size: 12px;
      font-style: normal;
    }}
    .summary-metric strong {{
      font-size: 22px;
      grid-row: span 2;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th,
    td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: right;
      vertical-align: top;
      white-space: nowrap;
    }}
    th:first-child {{
      text-align: left;
      min-width: 330px;
      position: sticky;
      left: 0;
      z-index: 1;
      background: white;
    }}
    thead th {{
      background: var(--band);
      font-weight: 650;
    }}
    thead th:first-child {{
      background: var(--band);
      z-index: 2;
    }}
    tbody tr:last-child th,
    tbody tr:last-child td {{
      border-bottom: 0;
    }}
    td.complete {{
      background: var(--complete);
    }}
    td.partial {{
      background: var(--partial);
    }}
    td.failed {{
      background: var(--failed);
    }}
    td.missing {{
      background: var(--missing);
      color: var(--muted);
    }}
    .status {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    td strong {{
      display: block;
      margin: 2px 0;
      font-size: 15px;
      color: var(--ink);
    }}
    @page {{
      size: landscape;
      margin: 0.35in;
    }}
    @media print {{
      * {{
        print-color-adjust: exact;
        -webkit-print-color-adjust: exact;
      }}
      body {{
        background: white;
      }}
      main {{
        max-width: none;
        padding: 0;
      }}
      .summary-item,
      tr {{
        break-inside: avoid;
        page-break-inside: avoid;
      }}
      th:first-child {{
        position: static;
      }}
    }}
  </style>
</head>
<body>
<main>
  <h1>SAPS ASV Runtime Success Rates</h1>
  <p>For each benchmark, this shows whether each framework produced runtime results
   and how many datasets finished.
  The table covers {total_benchmarks} benchmark classes.
  Missing cells mean ASV did not have a runtime row for that framework/benchmark.</p>
  {_benchmark_summary(rows, frameworks)}
  {_table(rows, frameworks)}
</main>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize benchmark runtime success rates by framework."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(".saps/outputs/warmup_full_visualizations/asv_success_rates.csv"),
        help="Success-rate CSV produced from ASV raw results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".saps/outputs/warmup_full_visualizations/asv_success_rates.html"),
        help="HTML report path.",
    )
    args = parser.parse_args()

    rows = _read_time_rows(args.input)
    if not rows:
        raise SystemExit(f"No runtime success-rate rows found in {args.input}")
    _write_html(rows, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
