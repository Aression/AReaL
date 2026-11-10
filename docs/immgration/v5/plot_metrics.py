#!/usr/bin/env python3
"""Plot the 30-step score-centering warm-up arms from metrics.csv.

Arms are SC on/off plus the noise level injected to emulate sampler drift:
``n0`` = no noise, ``n002`` = 0.02 Gaussian noise.

Only ``src == live`` rows are plotted: the CSV also carries a
``snapshot-0131`` block for one arm that comes from a different source and has
no entropy column, so mixing it in would fabricate a continuation.

Usage: python3 plot_metrics.py [metrics.csv] [out.png]
"""

import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [
    ("sc-off-n0", "SC off · no noise", "#9E9E9E", "-"),
    ("sc-off-n002", "SC off · 0.02 noise (12 live steps)", "#FF5722", "-"),
    ("sc-on-n0", "SC on · no noise", "#2196F3", "-"),
    ("sc-on-n002", "SC on · 0.02 noise", "#4CAF50", "-"),
]


def load(path):
    arms = defaultdict(list)
    arm = None
    header = None
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("# arm="):
                arm = line.split("arm=")[1].split()[0]
                continue
            if line.startswith("#") or not line.strip():
                continue
            if line.startswith("step,"):
                header = line.split(",")
                continue
            if arm is None:
                continue
            row = dict(zip(header, next(csv.reader([line]))))
            if row.get("src") == "live":
                arms[arm].append(row)
    return arms


def series(rows, key):
    steps, values = [], []
    for row in rows:
        try:
            values.append(float(row[key]))
            steps.append(int(float(row["step"])))
        except (KeyError, TypeError, ValueError):
            continue
    return steps, values


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "metrics.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "reward_curves.png"
    arms = load(src)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)

    for name, label, color, style in ARMS:
        steps, reward = series(arms[name], "reward")
        axes[0].plot(steps, reward, style, color=color, linewidth=1.6, marker="o",
                     markersize=3, alpha=0.9, label=label)
        steps, entropy = series(arms[name], "entropy")
        axes[1].plot(steps, entropy, style, color=color, linewidth=1.6, marker="o",
                     markersize=3, alpha=0.9, label=label)

    axes[0].set_ylabel("Reward")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Reward, live rows only")
    axes[1].set_ylabel("Policy entropy")
    axes[1].set_title("Entropy, live rows only")

    for ax in axes:
        ax.set_xlabel("Training step")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 30)

    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Score centering warm-up (30 steps, single seed)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
