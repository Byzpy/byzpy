#!/usr/bin/env python
"""
Generate a benchmark timing plot from the README table.

For each workload, plots ByzFL (ms), ByzPy direct, and ActorPool x2/x4/x6.
Missing ByzFL entries are left empty. Timeout/unsupported rows are encoded as
NaN (unsupported) or 10_000 ms (timeouts) to keep them visible.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _extract_ms(cell: str) -> float:
    """Grab the first numeric token from a table cell."""
    match = re.search(r"([0-9]*\\.?[0-9]+)", cell)
    if match:
        return float(match.group(1))
    return math.nan


def _parse_readme_table(path: Path) -> List[Dict[str, float]]:
    lines = path.read_text().splitlines()
    # Find the table start.
    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("| Workload | PyTorch / ActorPool Command | ByzFL Command |"):
            start_idx = i + 2  # skip header separator
            break
    if start_idx is None:
        raise RuntimeError("Could not locate benchmark table in README.")

    rows: List[Dict[str, float]] = []
    for line in lines[start_idx:]:
        if not line.startswith("|"):
            break
        parts = [p.strip() for p in line.split("|")]
        # Expect columns: '', workload, cmd1, cmd2, byzpy faster?, byzfl, direct, pool2, pool4, pool6, ''
        if len(parts) < 10 or parts[2].startswith("---"):
            continue
        workload = parts[1]
        byzfl = _extract_ms(parts[5])
        direct = _extract_ms(parts[6])
        pool2 = _extract_ms(parts[7])
        pool4 = _extract_ms(parts[8])
        pool6 = _extract_ms(parts[9]) if len(parts) > 9 else math.nan
        rows.append(
            {
                "workload": workload,
                "byzfl": byzfl,
                "byzpy": direct,
                "pool2": pool2,
                "pool4": pool4,
                "pool6": pool6,
            }
        )
    return rows


def _series() -> list[str]:
    return ["byzfl", "byzpy", "pool2", "pool4", "pool6"]


def _colors() -> dict[str, str]:
    return {
        "byzfl": "#8e44ad",
        "byzpy": "#2980b9",
        "pool2": "#16a085",
        "pool4": "#f39c12",
        "pool6": "#c0392b",
    }


def _label(key: str) -> str:
    return (
        key.replace("pool", "ActorPool x")
        .replace("byzpy", "ByzPy direct")
        .replace("byzfl", "ByzFL")
    )


def plot_all(df: pd.DataFrame, output: Optional[str]) -> None:
    df = df.set_index("workload")
    series = _series()
    colors = _colors()

    x = np.arange(len(df.index))
    width = 0.16
    fig, ax = plt.subplots(figsize=(16, 10))

    for idx, key in enumerate(series):
        offset = (idx - 2) * width
        vals = df[key].to_numpy(dtype=float)
        vals = np.where(vals <= 0, 1e-3, vals)  # avoid log-scale issues
        vals = np.nan_to_num(vals, nan=1e-3, posinf=1e-3, neginf=1e-3)
        ax.bar(x + offset, vals, width, label=_label(key), color=colors[key], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=60, ha="right")
    ax.set_ylabel("Time (ms, lower is better)")
    ax.set_title("Benchmark timings: ByzFL vs ByzPy (ActorPool variants)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_yscale("log", nonpositive="clip")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=200)
        print(f"Saved plot to {output}")
    else:
        plt.show()


def plot_per_workload(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = _colors()
    series = _series()
    for _, row in df.iterrows():
        workload = row["workload"]
        vals = [row[s] for s in series]
        mask = [not math.isnan(v) for v in vals]
        kept_series = [s for s, keep in zip(series, mask) if keep]
        kept_vals = [v for v, keep in zip(vals, mask) if keep]

        width = 0.6
        x = np.arange(len(kept_series))
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(
            x,
            kept_vals,
            width,
            color=[colors[s] for s in kept_series],
            alpha=0.9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([_label(s) for s in kept_series], rotation=25, ha="right")
        ax.set_ylabel("Time (ms, lower is better)")
        ax.set_title(f"{workload} timings")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ymax = max(kept_vals) if kept_vals else 1.0
        ax.set_ylim(0, ymax * 1.1)

        for bar, val in zip(bars, kept_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        fig.tight_layout()
        out_path = (
            output_dir
            / f"{workload.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('-', '_')}.png"
        )
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved per-workload plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark timings from README.md.")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks_plot.png",
        help="Where to save the plot (png).",
    )
    parser.add_argument(
        "--per-workload-dir",
        type=str,
        default="plots/workloads",
        help="Directory to save per-workload plots (set to '' to skip).",
    )
    args = parser.parse_args()
    readme_path = Path(__file__).with_name("README.md")
    df = pd.DataFrame(_parse_readme_table(readme_path))
    plot_all(df, args.output)
    if args.per_workload_dir:
        plot_per_workload(df, Path(args.per_workload_dir))


if __name__ == "__main__":
    main()
