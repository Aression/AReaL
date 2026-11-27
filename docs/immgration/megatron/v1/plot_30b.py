#!/usr/bin/env python3
"""Plot the Qwen3-30B-A3B MoE score-centering comparison.

The paired difference is recomputed from the per-arm ``acc`` table instead of the
``paired`` table, whose ``diff_ma10`` column is empty and whose rolling mean ended
up in the ``rho`` column.

Usage: python3 plot_30b.py [30b-exp-data.csv] [sc_30b.png]
"""

import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = [
    ("T2-off", "SC off · σ=0.12", "#FF5722", "-"),
    ("T2-on", "SC on · σ=0.12", "#2196F3", "-"),
    ("T2-clean-off", "SC off · no noise", "#9E9E9E", "-"),
    ("T2-clean-on", "SC on · no noise", "#4CAF50", "-"),
]

PAIRS = [("T2-off", "T2-on", "σ=0.12", "#2196F3"),
         ("T2-clean-off", "T2-clean-on", "no noise", "#4CAF50")]

WINDOW = 10


def load(path):
    acc = defaultdict(dict)
    with open(path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["table"] != "acc":
                continue
            try:
                acc[row["arm_or_condition"]][int(row["step"])] = float(row["acc"])
            except (TypeError, ValueError):
                continue
    return acc


def rolling_mean(steps, values, window=WINDOW):
    series = np.array([values[s] for s in steps], dtype=float)
    half = window // 2
    return np.array([series[max(0, i - half) : i + half + 1].mean() for i in range(len(series))])


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "30b-exp-data.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "sc_30b.png"
    acc = load(src)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    for arm, label, color, style in ARMS:
        steps = sorted(acc[arm])
        axes[0].plot(steps, rolling_mean(steps, acc[arm]), style, color=color,
                     linewidth=1.8, label=label)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Rollout accuracy (10-step mean)")
    axes[0].set_title("Accuracy: SC on/off vs injected mismatch")
    axes[0].set_ylim(0.0, 0.6)

    for off, on, label, color in PAIRS:
        steps = sorted(set(acc[off]) & set(acc[on]))
        diff = {s: acc[on][s] - acc[off][s] for s in steps}
        axes[1].plot(steps, rolling_mean(steps, diff), color=color, linewidth=1.8,
                     label=f"{label} (mean {np.mean(list(diff.values())):+.3f})")
    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Paired difference (SC on − SC off)")
    axes[1].set_title("Paired effect of score centering")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Qwen3-30B-A3B MoE, Megatron actor + vLLM rollout, k=4", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
