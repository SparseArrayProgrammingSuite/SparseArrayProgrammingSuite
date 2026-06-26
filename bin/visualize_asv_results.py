#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

FRAMEWORK_ORDER = ["numpy", "scipy", "sparse", "pytorch", "unknown"]
FRAMEWORK_COLORS = {
    "numpy": "#2f6f73",
    "scipy": "#6d5fb5",
    "sparse": "#8c5a2b",
    "pytorch": "#b23b4a",
    "unknown": "#5f6f84",
}
DTYPE_SIZES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float32": 4,
    "float64": 8,
}


@dataclass(frozen=True)
class Row:
    benchmark_key: str
    benchmark: str
    dataset: str
    framework: str
    metric: str
    value: float
    unit: str
    commit: str
    machine: str
    env_name: str
    started_at: int | None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _metric_from_key(key: str) -> tuple[str, str]:
    method = key.rsplit(".", 1)[-1]
    if method.startswith("time_"):
        return "time", "seconds"
    if method.startswith("peakmem_"):
        return "memory", "bytes"
    if method.startswith("mem_"):
        return "object_memory", "bytes"
    return method, "value"


def _short_benchmark_name(key: str) -> str:
    module, cls, _method = key.rsplit(".", 2)
    return f"{module}.{cls}"


def _framework_name(data: dict[str, Any]) -> str:
    path = data.get("env_vars", {}).get("SAPS_FRAMEWORK")
    if not path:
        return "unknown"
    name = Path(path).stem
    return name.removeprefix("saps_")


def _read_results(path: Path) -> list[Row]:
    data = json.loads(path.read_text())
    rows: list[Row] = []
    columns = data.get("result_columns", [])
    started_idx = columns.index("started_at") if "started_at" in columns else None
    machine = data.get("params", {}).get("machine", path.parent.name)
    framework = _framework_name(data)

    for key, payload in data.get("results", {}).items():
        if not isinstance(payload, list) or len(payload) < 2:
            continue
        values = payload[0]
        params = payload[1]
        if (
            not isinstance(values, list)
            or not params
            or not isinstance(params[0], list)
        ):
            continue
        datasets = params[0]
        metric, unit = _metric_from_key(key)
        started_at = None
        if started_idx is not None and len(payload) > started_idx:
            raw_started = payload[started_idx]
            if isinstance(raw_started, int):
                started_at = raw_started
        for dataset, value in zip(datasets, values, strict=False):
            if not _is_number(value):
                continue
            rows.append(
                Row(
                    benchmark_key=key,
                    benchmark=_short_benchmark_name(key),
                    dataset=str(dataset),
                    framework=framework,
                    metric=metric,
                    value=float(value),
                    unit=unit,
                    commit=data.get("commit_hash", "")[:8],
                    machine=machine,
                    env_name=data.get("env_name", ""),
                    started_at=started_at,
                )
            )
    return rows


def _result_files(results_dir: Path) -> list[Path]:
    return [
        path
        for path in results_dir.glob("*/*.json")
        if path.name != "benchmarks_meta.json"
    ]


def _latest_result_file(results_dir: Path) -> Path:
    files = list(_result_files(results_dir))
    if not files:
        raise FileNotFoundError(f"No ASV result JSON files found under {results_dir}")
    return max(files, key=lambda path: path.stat().st_mtime)


def _format_value(row: Row) -> str:
    if row.metric == "time":
        if row.value < 1e-3:
            return f"{row.value * 1e6:.1f} us"
        if row.value < 1:
            return f"{row.value * 1e3:.2f} ms"
        return f"{row.value:.2f} s"
    if row.metric == "memory":
        if row.value == 0:
            return "0 B"
        units = ["B", "KiB", "MiB", "GiB"]
        value = row.value
        for unit in units:
            if abs(value) < 1024 or unit == units[-1]:
                return f"{value:.2f} {unit}"
            value /= 1024
    return f"{row.value:.6g}"


def _hex_array_len(value: Any) -> int | None:
    if (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], str)
    ):
        itemsize = DTYPE_SIZES.get(value[0])
        if itemsize:
            return len(value[1]) // (itemsize * 2)
    return None


def _shape_label(shape: Any) -> str | None:
    if isinstance(shape, (list, tuple)) and shape:
        return "x".join(str(dim) for dim in shape)
    return None


def _product(values: list[int] | tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _format_sparsity(nnz: int, shape: Any) -> str:
    if not isinstance(shape, (list, tuple)) or not shape:
        return ""
    total = _product(shape)
    if total <= 0:
        return ""
    return f"{100 * nnz / total:.4g}%"


def _dataset_info_from_cache(cache_path: Path) -> dict[str, str]:
    try:
        payload = json.loads(cache_path.read_text())
        first_array = json.loads(payload["binsparse"][0])
    except (KeyError, IndexError, json.JSONDecodeError, TypeError):
        return {}

    shape = first_array.get("shape")
    shape_text = _shape_label(shape)
    nnz: int | None = None
    if first_array.get("format") == "COO":
        nnz = _hex_array_len(first_array.get("values"))
    elif first_array.get("format") == "dense" and isinstance(shape, (list, tuple)):
        nnz = _product(shape)

    dataset_info = {}
    if shape_text:
        dataset_info["dims"] = shape_text
    if nnz is not None:
        dataset_info["nnz"] = f"{nnz:,}"
        sparsity = _format_sparsity(nnz, shape)
        if sparsity:
            dataset_info["sparsity"] = sparsity
    return dataset_info


def _load_dataset_info(
    manifest_path: Path, cache_dir: Path
) -> dict[str, dict[str, str]]:
    info: dict[str, dict[str, str]] = {}
    manifest = {}
    if not manifest_path.exists():
        manifest = {}
    else:
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}

    for dataset_key, entry in manifest.items():
        digest = entry.get("digest") if isinstance(entry, dict) else None
        if not digest or "." not in dataset_key:
            continue
        generator, dataset = dataset_key.split(".", 1)
        cache_path = cache_dir / generator / dataset / f"{digest}.json"
        if not cache_path.exists():
            continue
        dataset_info = _dataset_info_from_cache(cache_path)
        if dataset_info:
            info[dataset_key] = dataset_info

    for generator_dir in cache_dir.iterdir() if cache_dir.exists() else []:
        if not generator_dir.is_dir():
            continue
        for dataset_dir in generator_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_key = f"{generator_dir.name}.{dataset_dir.name}"
            if dataset_key in info:
                continue
            cache_files = sorted(
                dataset_dir.glob("*.json"), key=lambda path: path.stat().st_mtime
            )
            if not cache_files:
                continue
            dataset_info = _dataset_info_from_cache(cache_files[-1])
            if dataset_info:
                info[dataset_key] = dataset_info
    return info


def _group_label(row: Row, dataset_info: dict[str, dict[str, str]]) -> str:
    label = f"{row.benchmark.split('.')[-1]} / {row.dataset.split('.')[-1]}"
    info = dataset_info.get(row.dataset)
    if not info:
        return label
    details = []
    if "dims" in info:
        details.append(f"dims={info['dims']}")
    if "nnz" in info:
        details.append(f"nnz={info['nnz']}")
    if "sparsity" in info:
        details.append(f"sparsity={info['sparsity']}")
    if not details:
        return label
    return f"{label} ({', '.join(details)})"


def _framework_sort_key(framework: str) -> tuple[int, str]:
    if framework in FRAMEWORK_ORDER:
        return (FRAMEWORK_ORDER.index(framework), framework)
    return (len(FRAMEWORK_ORDER), framework)


def _legend(frameworks: list[str]) -> str:
    items = []
    for framework in frameworks:
        color = FRAMEWORK_COLORS.get(framework, FRAMEWORK_COLORS["unknown"])
        items.append(
            '<span class="legend-item">'
            f'<span class="legend-swatch" style="background:{color}"></span>'
            f"{escape(framework)}"
            "</span>"
        )
    return '<div class="legend">' + "".join(items) + "</div>"


def _comparison_svg(
    rows: list[Row], title: str, dataset_info: dict[str, dict[str, str]]
) -> str:
    if not rows:
        return ""
    groups: dict[tuple[str, str], list[Row]] = {}
    for row in rows:
        groups.setdefault((row.benchmark, row.dataset), []).append(row)

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (_group_label(item[1][0], dataset_info), item[0]),
    )
    values = [row.value for row in rows]
    positive_values = [value for value in values if value > 0]
    max_value = max(values) if values else 0
    min_positive = min(positive_values) if positive_values else 0
    use_log_scale = min_positive > 0 and max_value / min_positive > 50

    width = 980
    left = 270
    right = 92
    row_h = 27
    group_gap = 18
    top = 44
    group_heights = [
        24
        + row_h
        * len(sorted(group_rows, key=lambda row: _framework_sort_key(row.framework)))
        for _group_key, group_rows in ordered_groups
    ]
    height = top + sum(group_heights) + group_gap * max(0, len(ordered_groups) - 1) + 20
    chart_w = width - left - right
    rendered_title = f"{title} (log scale)" if use_log_scale else title
    parts = [
        (
            f'<svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{escape(rendered_title)}">'
        ),
        f'<text x="0" y="22" class="chart-title">{escape(rendered_title)}</text>',
    ]
    y = top
    for _group_key, group_rows in ordered_groups:
        group_rows = sorted(
            group_rows, key=lambda row: _framework_sort_key(row.framework)
        )
        group_label = escape(_group_label(group_rows[0], dataset_info))
        parts.append(
            f'<text x="0" y="{y + 15}" class="group-label">{group_label}</text>'
        )
        y += 24
        for row in group_rows:
            color = FRAMEWORK_COLORS.get(row.framework, FRAMEWORK_COLORS["unknown"])
            if max_value <= 0:
                scaled = 0
            elif use_log_scale and row.value > 0:
                denominator = math.log10(max_value / min_positive) + 0.2
                scaled = (math.log10(row.value / min_positive) + 0.2) / denominator
            else:
                scaled = row.value / max_value
            bar_w = 2 if row.value == 0 else max(2, chart_w * scaled)
            value_x = left + chart_w + 10
            formatted_value = escape(_format_value(row))
            framework = escape(row.framework)
            parts.extend(
                [
                    (
                        f'<text x="28" y="{y + 16}" '
                        f'class="axis-label">{framework}</text>'
                    ),
                    (
                        f'<rect x="{left}" y="{y}" width="{chart_w}" '
                        'height="18" rx="3" class="bar-track" />'
                    ),
                    (
                        f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" '
                        f'height="18" rx="3" fill="{color}" />'
                    ),
                    (
                        f'<text x="{value_x}" y="{y + 14}" '
                        f'class="value-label">{formatted_value}</text>'
                    ),
                ]
            )
            y += row_h
        y += group_gap
    parts.append("</svg>")
    return "\n".join(parts)


def _write_csv(
    rows: list[Row], path: Path, dataset_info: dict[str, dict[str, str]]
) -> None:
    header = [
        "benchmark_key",
        "benchmark",
        "dataset",
        "framework",
        "dimensions",
        "nnz",
        "sparsity",
        "metric",
        "value",
        "unit",
        "formatted_value",
        "commit",
        "machine",
        "started_at",
    ]
    lines = [",".join(header)]
    for row in rows:
        info = dataset_info.get(row.dataset, {})
        values = [
            row.benchmark_key,
            row.benchmark,
            row.dataset,
            row.framework,
            info.get("dims", ""),
            info.get("nnz", ""),
            info.get("sparsity", ""),
            row.metric,
            repr(row.value),
            row.unit,
            _format_value(row),
            row.commit,
            row.machine,
            str(row.started_at or ""),
        ]
        lines.append(",".join(json.dumps(value) for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(
    rows: list[Row],
    output: Path,
    dataset_info: dict[str, dict[str, str]],
) -> None:
    time_rows = sorted(
        (row for row in rows if row.metric == "time"),
        key=lambda row: (
            row.benchmark,
            row.dataset,
            _framework_sort_key(row.framework),
        ),
    )
    mem_rows = sorted(
        (row for row in rows if row.metric == "memory"),
        key=lambda row: (
            row.benchmark,
            row.dataset,
            _framework_sort_key(row.framework),
        ),
    )
    frameworks = sorted({row.framework for row in rows}, key=_framework_sort_key)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SAPS ASV Visualization Test</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18212f;
      --muted: #5f6f84;
      --line: #d8dee8;
      --band: #f6f8fb;
      --time: #2f6f73;
      --memory: #8c5a2b;
      --track: #edf1f6;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: white;
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    section {{
      border-top: 1px solid var(--line);
      padding-top: 24px;
      margin-top: 24px;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
      overflow: visible;
    }}
    .chart-title {{
      font-size: 18px;
      font-weight: 650;
      fill: var(--ink);
    }}
    .axis-label {{
      font-size: 13px;
      fill: var(--ink);
    }}
    .group-label {{
      font-size: 14px;
      font-weight: 650;
      fill: var(--ink);
    }}
    .value-label {{
      font-size: 12px;
      fill: var(--muted);
    }}
    .bar-track {{
      fill: var(--track);
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-swatch {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 2px;
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
      section {{
        break-inside: avoid;
        page-break-inside: avoid;
      }}
      svg {{
        break-inside: avoid;
        page-break-inside: avoid;
      }}
    }}
  </style>
</head>
<body>
<main>
  <h1>SAPS ASV Framework Comparison</h1>
  {_legend(frameworks)}
  <section>
    {
        _comparison_svg(
            time_rows,
            "Runtime by benchmark, dataset, and framework",
            dataset_info,
        )
    }
  </section>
  <section>
    {
        _comparison_svg(
            mem_rows,
            "Peak memory by benchmark, dataset, and framework",
            dataset_info,
        )
    }
  </section>
</main>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")


def _row_search_text(row: Row) -> str:
    return (
        f"{row.benchmark_key} {row.benchmark} {row.dataset} "
        f"{row.framework} {row.metric}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize SAPS ASV result JSON.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(".saps/outputs/results"),
        help="ASV results directory.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="*",
        default=None,
        help="Specific ASV result JSON file(s). Defaults to the newest result file.",
    )
    parser.add_argument(
        "--all-results",
        action="store_true",
        help="Read every ASV result JSON under --results-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".saps/outputs/visualizations"),
        help="Directory for report artifacts.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifest.json"),
        help="SAPS dataset manifest used to annotate dimensions and nnz.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".saps/outputs/cache"),
        help="SAPS dataset cache used to annotate dimensions and nnz.",
    )
    parser.add_argument(
        "--exclude-row",
        action="append",
        default=[],
        help=(
            "Regex for rows to omit from the generated report. "
            "Matches benchmark key, benchmark, dataset, framework, and metric."
        ),
    )
    args = parser.parse_args()

    if args.input:
        sources = args.input
    elif args.all_results:
        sources = _result_files(args.results_dir)
    else:
        sources = [_latest_result_file(args.results_dir)]

    rows_by_key: dict[tuple[str, str, str, str], Row] = {}
    for source in sources:
        for row in _read_results(source):
            key = (row.benchmark_key, row.dataset, row.framework, row.metric)
            previous = rows_by_key.get(key)
            if previous is None or (row.started_at or 0) >= (previous.started_at or 0):
                rows_by_key[key] = row
    rows = list(rows_by_key.values())
    for pattern in args.exclude_row:
        regex = re.compile(pattern)
        rows = [row for row in rows if not regex.search(_row_search_text(row))]
    if not rows:
        raise SystemExit("No successful ASV result rows found")

    dataset_info = _load_dataset_info(args.manifest, args.cache_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "asv_summary.csv"
    html_path = args.output_dir / "asv_summary.html"
    _write_csv(rows, csv_path, dataset_info)
    _write_html(rows, html_path, dataset_info)
    print(f"Wrote {csv_path}")
    print(f"Wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
