#!/usr/bin/env python3
"""
Redraw all experiment figures for the Score Centering PR.

This script reads the four 1000-step training curves (CSV) and produces
self-contained PNG charts.  It requires only matplotlib + numpy (standard
scientific Python); no torch, no AReaL code, no NPU.

Usage:
    pip install matplotlib numpy
    python3 generate_pr_figures.py [--data-dir DIR] [--out-dir DIR]

Inputs (CSV columns: step,grad_norm,entropy,reward,sc_corr_ratio,sc_rho):
    curve_sc_on_rec.csv      SC-on  + recompute-on
    curve_sc_on_norec.csv    SC-on  + recompute-off
    curve_sc_off_rec.csv     SC-off + recompute-on
    curve_sc_off_norec.csv   SC-off + recompute-off

Outputs (PNG):
    reward_curves.png            4-config reward curves (20-step rolling mean)
    grad_norm.png                4-config grad_norm (clipped at 50)
    entropy.png                  4-config entropy curves
    sc_tail_model_health.png     p_mass/q_mass pre-fix vs post-fix (bar chart)
    sc_drift_tracking.png        s_bar vs correction pre-fix vs post-fix (bar chart)
    sc_effect_by_phase.png       SC reward advantage by phase (bar chart)
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIGS = {
    "sc_on_rec": {
        "file": "curve_sc_on_rec.csv",
        "label": "SC-on + recompute-on",
        "color": "#2196F3",
        "marker": "o",
    },
    "sc_on_norec": {
        "file": "curve_sc_on_norec.csv",
        "label": "SC-on + recompute-off",
        "color": "#4CAF50",
        "marker": "s",
    },
    "sc_off_rec": {
        "file": "curve_sc_off_rec.csv",
        "label": "SC-off + recompute-on",
        "color": "#FF5722",
        "marker": "^",
    },
    "sc_off_norec": {
        "file": "curve_sc_off_norec.csv",
        "label": "SC-off + recompute-off",
        "color": "#9E9E9E",
        "marker": "D",
    },
}

PHASES = [
    ("Step 0-99", 0, 100),
    ("Step 100-199", 100, 200),
    ("Step 200-499", 200, 500),
    ("Step 500-799", 500, 800),
    ("Step 800-957", 800, 958),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(filepath):
    """Load a CSV into a dict of lists keyed by column name.

    Handles UTF-8 BOM (which corrupts the first column header) by
    re-encoding as utf-8-sig on read.
    """
    data = defaultdict(list)
    with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if val is None or val == "":
                    data[key].append(np.nan)
                else:
                    try:
                        data[key].append(float(val))
                    except ValueError:
                        data[key].append(np.nan)
    for key in data:
        data[key] = np.array(data[key], dtype=float)
    return data


def rolling_mean(arr, window):
    """Simple rolling mean with edge padding."""
    if len(arr) == 0:
        return arr
    result = np.copy(arr)
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        result[i] = np.nanmean(arr[lo:hi])
    return result


def phase_mean(arr, lo, hi):
    """Mean of arr[lo:hi], ignoring NaN."""
    segment = arr[lo:hi]
    valid = segment[~np.isnan(segment)]
    if len(valid) == 0:
        return np.nan
    return np.mean(valid)


# ---------------------------------------------------------------------------
# Figure 1: Reward curves
# ---------------------------------------------------------------------------

def plot_reward_curves(all_data, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, cfg in CONFIGS.items():
        d = all_data[key]
        reward = d["reward"]
        rm = rolling_mean(reward, 20)
        ax.plot(d["step"], rm, label=cfg["label"], color=cfg["color"],
                linewidth=2, alpha=0.9)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Reward (20-step rolling mean)", fontsize=12)
    ax.set_title("Reward Curves: SC-on vs SC-off (1000 steps)", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reward_curves.png"), dpi=150)
    plt.close(fig)
    print("  -> reward_curves.png")


# ---------------------------------------------------------------------------
# Figure 2: Gradient norm
# ---------------------------------------------------------------------------

def plot_grad_norm(all_data, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, cfg in CONFIGS.items():
        d = all_data[key]
        gn = np.clip(d["grad_norm"], 0, 50)
        ax.plot(d["step"], gn, label=cfg["label"], color=cfg["color"],
                linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Gradient Norm (clipped at 50)", fontsize=12)
    ax.set_title("Gradient Norm: SC-on stable, SC-off+recompute-on crashes",
                 fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 50)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "grad_norm.png"), dpi=150)
    plt.close(fig)
    print("  -> grad_norm.png")


# ---------------------------------------------------------------------------
# Figure 3: Entropy
# ---------------------------------------------------------------------------

def plot_entropy(all_data, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, cfg in CONFIGS.items():
        d = all_data[key]
        ax.plot(d["step"], d["entropy"], label=cfg["label"],
                color=cfg["color"], linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Policy Entropy", fontsize=12)
    ax.set_title("Policy Entropy: SC-on converges, SC-off+recompute-on diverges",
                 fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 2.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "entropy.png"), dpi=150)
    plt.close(fig)
    print("  -> entropy.png")


# ---------------------------------------------------------------------------
# Figure 4: SC tail model health (p_mass / q_mass)
# ---------------------------------------------------------------------------

# Pre-fix data from the experiment report (SC-on-rec, pre-attention-mask-fix)
PRE_FIX_DATA = {
    "p_mass": [0.400, 0.400, 0.299, 0.299, 0.473],
    "q_mass": [0.987, 0.987, 0.993, 0.993, 0.993],
    "rho":    [13.3,  13.3,  742,   742,   1325],
}

# Post-fix data (SC-on-rec, from 2x2 experiment)
POST_FIX_DATA = {
    "p_mass": [0.990, 0.999, 0.999, 0.998, 0.998],
    "q_mass": [0.991, 1.000, 1.000, 1.000, 1.000],
    "rho":    [0.73,  0.87,  0.97,  1.02,  1.03],
}


def plot_tail_model_health(out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x_labels = [p[0].replace("Step ", "") for p in PHASES]
    x = np.arange(len(x_labels))
    width = 0.35

    # Panel 1: p_mass
    ax = axes[0]
    ax.bar(x - width / 2, PRE_FIX_DATA["p_mass"], width, label="Pre-fix",
           color="#FF5722", alpha=0.8)
    ax.bar(x + width / 2, POST_FIX_DATA["p_mass"], width, label="Post-fix",
           color="#4CAF50", alpha=0.8)
    ax.set_ylabel("p_mass (trainer)", fontsize=11)
    ax.set_title("Trainer top-k mass (p_mass)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.2, axis="y")

    # Panel 2: q_mass
    ax = axes[1]
    ax.bar(x - width / 2, PRE_FIX_DATA["q_mass"], width, label="Pre-fix",
           color="#FF5722", alpha=0.8)
    ax.bar(x + width / 2, POST_FIX_DATA["q_mass"], width, label="Post-fix",
           color="#4CAF50", alpha=0.8)
    ax.set_ylabel("q_mass (sampler)", fontsize=11)
    ax.set_title("Sampler top-k mass (q_mass)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.2, axis="y")

    # Panel 3: rho (log scale)
    ax = axes[2]
    ax.bar(x - width / 2, PRE_FIX_DATA["rho"], width, label="Pre-fix",
           color="#FF5722", alpha=0.8)
    ax.bar(x + width / 2, POST_FIX_DATA["rho"], width, label="Post-fix",
           color="#4CAF50", alpha=0.8)
    ax.set_ylabel("rho (tail mass ratio)", fontsize=11)
    ax.set_title("Tail mass ratio (rho)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.2, axis="y", which="both")

    fig.suptitle("SC Tail Model Health: Pre-fix vs Post-fix", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sc_tail_model_health.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  -> sc_tail_model_health.png")


# ---------------------------------------------------------------------------
# Figure 5: SC drift tracking (s_bar vs correction)
# ---------------------------------------------------------------------------

# Post-fix s_bar and correction (SC-on-rec)
POST_FIX_SBAR = [-0.564, -0.341, -0.249, -0.218, -0.262]
POST_FIX_CORR = [-0.151, -0.104, -0.095, -0.106, -0.165]

# Pre-fix s_bar and correction (SC-on-rec)
PRE_FIX_SBAR = [-14.68, -14.68, -47.60, -47.60, -62.85]
PRE_FIX_CORR = [-11.05, -11.05, 2.57, 2.57, 13.01]


def plot_drift_tracking(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_labels = [p[0].replace("Step ", "") for p in PHASES]
    x = np.arange(len(x_labels))
    width = 0.35

    # Panel 1: Post-fix
    ax = axes[0]
    ax.bar(x - width / 2, POST_FIX_SBAR, width, label="s_bar (drift signal)",
           color="#2196F3", alpha=0.8)
    ax.bar(x + width / 2, POST_FIX_CORR, width, label="correction",
           color="#4CAF50", alpha=0.8)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Post-fix: correction tracks s_bar", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.grid(True, alpha=0.2, axis="y")

    # Panel 2: Pre-fix
    ax = axes[1]
    ax.bar(x - width / 2, PRE_FIX_SBAR, width, label="s_bar (drift signal)",
           color="#2196F3", alpha=0.8)
    ax.bar(x + width / 2, PRE_FIX_CORR, width, label="correction",
           color="#FF5722", alpha=0.8)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Pre-fix: correction decouples (sign flip)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("SC Drift Tracking: s_bar vs correction", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sc_drift_tracking.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  -> sc_drift_tracking.png")


# ---------------------------------------------------------------------------
# Figure 6: SC effect by phase (reward advantage)
# ---------------------------------------------------------------------------

def plot_sc_effect(all_data, out_dir):
    sc_on_rewards = []
    sc_off_rewards = []
    for _, lo, hi in PHASES:
        on_vals = []
        for k in ("sc_on_rec", "sc_on_norec"):
            r = all_data[k]["reward"][lo:hi]
            on_vals.append(np.nanmean(r))
        off_vals = []
        for k in ("sc_off_rec", "sc_off_norec"):
            r = all_data[k]["reward"][lo:hi]
            off_vals.append(np.nanmean(r))
        sc_on_rewards.append(np.mean(on_vals))
        sc_off_rewards.append(np.mean(off_vals))

    advantages = [a - b for a, b in zip(sc_on_rewards, sc_off_rewards)]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = [p[0].replace("Step ", "") for p in PHASES]
    x = np.arange(len(x_labels))

    colors = ["#E3F2FD", "#BBDEFB", "#90CAF9", "#64B5F6", "#2196F3"]
    bars = ax.bar(x, advantages, color=colors, edgecolor="#1565C0",
                  linewidth=1.2)

    for bar, val in zip(bars, advantages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"+{val:.3f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.set_xlabel("Training Phase", fontsize=12)
    ax.set_ylabel("Reward Advantage (SC-on - SC-off)", fontsize=12)
    ax.set_title("SC Reward Advantage Grows Over Training", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=30, ha="right")
    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sc_effect_by_phase.png"), dpi=150)
    plt.close(fig)
    print("  -> sc_effect_by_phase.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate PR figures from 2x2 experiment data."
    )
    parser.add_argument("--data-dir", default=".",
                        help="Directory containing the CSV files (default: .)")
    parser.add_argument("--out-dir", default=".",
                        help="Output directory for PNG files (default: .)")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load all CSVs
    all_data = {}
    for key, cfg in CONFIGS.items():
        filepath = os.path.join(data_dir, cfg["file"])
        if not os.path.isfile(filepath):
            print(f"ERROR: {filepath} not found", file=sys.stderr)
            sys.exit(1)
        all_data[key] = load_csv(filepath)
        print(f"  Loaded {cfg['file']}: {len(all_data[key]['step'])} steps")

    print("\nGenerating figures:")
    plot_reward_curves(all_data, out_dir)
    plot_grad_norm(all_data, out_dir)
    plot_entropy(all_data, out_dir)
    plot_tail_model_health(out_dir)
    plot_drift_tracking(out_dir)
    plot_sc_effect(all_data, out_dir)

    print(f"\nDone. Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
