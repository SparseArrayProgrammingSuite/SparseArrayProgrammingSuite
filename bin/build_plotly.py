#!/usr/bin/env python3
"""Build a Dolan–More performance profile from asv_summary.csv."""

import math
from pathlib import Path

import pandas as pd
import plotly.express as px

INPUT = Path(".saps/outputs/warmup_full_visualizations/asv_summary.csv")
OUTPUT = Path("docs/assets/plots/runtime-summary.html")
TIME_LIMIT_S = 30.0


def main() -> None:
    df = pd.read_csv(INPUT)
    df = df[df["metric"] == "time"]
    # Only finished runs within the time limit get a ratio.
    df = df[df["value"] <= TIME_LIMIT_S]

    # Each dataset of an N-dataset benchmark is worth 1/N.
    n_datasets = df.groupby("benchmark")["dataset"].transform("nunique")
    df = df.assign(weight=1.0 / n_datasets)

    # Ratio = t / t_best on the same (benchmark, dataset) problem.
    best = df.groupby(["benchmark", "dataset"])["value"].transform("min")
    df = df.assign(ratio=df["value"] / best)

    # For each framework sort by ratio and accumulate weight.
    total_weight = (
        df[["benchmark", "dataset", "weight"]].drop_duplicates().loc[:, "weight"].sum()
    )
    # Shared right edge so every curve draws to the end of the plot.
    # Round up to the next power of 10
    x_max = max(10.0, float(df["ratio"].max()))
    x_max_tick = 10 ** math.ceil(math.log10(x_max))
    tickvals = [10**k for k in range(int(math.log10(x_max_tick)) + 1)]

    curves = []
    for framework, group in df.groupby("framework"):
        ordered = group.sort_values("ratio")
        xs = ordered["ratio"].to_numpy().tolist()
        ys = (100.0 * ordered["weight"].cumsum() / total_weight).to_numpy().tolist()
        # Extend flat to the shared right edge.
        if not xs or xs[-1] < x_max_tick:
            xs.append(x_max_tick)
            ys.append(ys[-1] if ys else 0.0)
        curves.append(pd.DataFrame({"ratio": xs, "pct": ys, "framework": framework}))
    plot_df = pd.concat(curves, ignore_index=True)

    fig = px.line(
        plot_df,
        x="ratio",
        y="pct",
        color="framework",
        log_x=True,
        line_shape="hv",
        labels={
            "ratio": "Ratio (runtime / best runtime)",
            "pct": "% of suite completed",
            "framework": "Framework",
        },
        title="SAPS framework performances",
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Ratio (runtime / best)",
        yaxis_title="% of suite",
        yaxis_range=[0, 100],
        margin={"l": 40, "r": 20, "t": 60, "b": 60},
    )
    # Log spacing, but tick labels are the real ratios: 1, 10, 100, ...
    fig.update_xaxes(
        type="log",
        range=[0, math.log10(x_max_tick)],
        tickmode="array",
        tickvals=tickvals,
        ticktext=[str(v) for v in tickvals],
    )
    fig.update_traces(line={"width": 5})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        fig.to_html(full_html=True, include_plotlyjs="cdn"),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
