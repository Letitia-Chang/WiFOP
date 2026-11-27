"""Draw the README headline figure from saved metrics (no retraining).

Reads reports/metrics/rerun_compare.json (final rerun, held-out 2023 test set)
and plots tie-aware top-k recall for the per-cell historical fire-rate baseline
versus weighted XGBoost, the best of the five models.

Usage: python src/plot_readme_figure.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics" / "rerun_compare.json"
OUT = ROOT / "reports" / "figures" / "readme_baseline_vs_model.png"

INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e6e5e0"
SERIES = {
    "hist_rate": ("Per-cell historical fire rate (baseline)", "#2a78d6", "o"),
    "weighted_new": ("Weighted XGBoost (best of 5 models)", "#eb6834", "s"),
}


def main():
    m = json.loads(METRICS.read_text())
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    for key, (label, color, marker) in SERIES.items():
        pts = m[key]["tie_aware"]
        k = [p["top_k_pct"] for p in pts]
        r = [100 * p["expected_recall"] for p in pts]
        ax.plot(k, r, color=color, lw=2, marker=marker, ms=7,
                markeredgecolor="white", markeredgewidth=1.5,
                label=f"{label}  ·  PR-AUC {m[key]['pr_auc']:.3f}, ROC-AUC {m[key]['roc_auc']:.2f}")
        dy = 9 if key == "hist_rate" else -14
        ax.annotate(f"{r[-1]:.0f}%", (k[-1], r[-1]), xytext=(0, dy), ha="center",
                    textcoords="offset points", va="center", color=INK, fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.5, 1, 2, 5, 10])
    ax.minorticks_off()
    ax.set_xticklabels(["0.1%", "0.5%", "1%", "2%", "5%", "10%"])
    ax.set_xlim(0.08, 13)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
    ax.set_xlabel("Riskiest share of grid-cell-days flagged (log scale)", color=INK2)
    ax.set_ylabel("Share of 2023 fires caught", color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    fig.suptitle("A per-cell fire-history baseline outranks the best model",
                 x=0.015, ha="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_title("Held-out 2023 test set, 2.37M grid-cell-days, 0.16% fire rate; "
                 "tie-aware recall at each cutoff",
                 loc="left", fontsize=9.5, color=INK2, pad=4)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT, facecolor="white")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
