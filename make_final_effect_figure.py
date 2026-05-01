"""Single-panel summary of the fine-tuned Geneformer's COVID-vs-Healthy
classification on the held-out test set of 248 classical monocytes.

Every dot is one test cell:
  - x position: true label (Healthy | COVID-19), with horizontal jitter
  - marker shape: donor (HIP043 / S556 / S557 / S558)
  - y position: Geneformer's predicted P(COVID)
  - color: correct (blue) or misclassified (red)

Read as: a well-trained classifier should pile Healthy dots at the bottom
and COVID dots at the top, with the decision line at 0.5 cleanly separating them.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
COMP = ROOT / "data/geneformer/tier2/run_mono_covid/comparison"
CSV = COMP / "per_cell_predictions.csv"
JSON = COMP / "comparison.json"
OUT = COMP / "final_effect.png"

COL_CORRECT = "#2b8cbe"
COL_ERROR = "#e41a1c"
MARKER_BY_DONOR = {
    "HIP043": "o",
    "S556": "^",
    "S557": "s",
    "S558": "D",
}
DONOR_LABEL = {
    "HIP043": "HIP043 (Healthy)",
    "S556": "S556 (COVID)",
    "S557": "S557 (COVID)",
    "S558": "S558 (COVID)",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titleweight": "bold",
    "figure.dpi": 140,
})


def main() -> None:
    df = pd.read_csv(CSV)
    df = df.dropna(subset=["gf_proba_covid", "gf_pred_covid", "gf_true_covid"]).copy()
    df["gf_true_covid"] = df["gf_true_covid"].astype(int)
    df["gf_pred_covid"] = df["gf_pred_covid"].astype(int)
    df["correct"] = df["gf_true_covid"] == df["gf_pred_covid"]

    metrics = json.loads(JSON.read_text())["geneformer"]

    n_healthy = int((df["gf_true_covid"] == 0).sum())
    n_covid = int((df["gf_true_covid"] == 1).sum())
    n_healthy_ok = int(((df["gf_true_covid"] == 0) & df["correct"]).sum())
    n_covid_ok = int(((df["gf_true_covid"] == 1) & df["correct"]).sum())

    rng = np.random.default_rng(42)
    df["_x"] = df["gf_true_covid"].astype(float) + rng.normal(0, 0.09, len(df))

    fig, (ax_main, ax_hist) = plt.subplots(
        1, 2, figsize=(11, 6.4),
        gridspec_kw={"width_ratios": [3.4, 1], "wspace": 0.04},
        sharey=True,
    )
    fig.subplots_adjust(top=0.84)

    for donor, marker in MARKER_BY_DONOR.items():
        sub = df[df["sample_id"] == donor]
        if sub.empty:
            continue
        ok = sub[sub["correct"]]
        err = sub[~sub["correct"]]
        ax_main.scatter(
            ok["_x"], ok["gf_proba_covid"],
            s=34, color=COL_CORRECT, alpha=0.8,
            edgecolor="white", linewidth=0.4,
            marker=marker, zorder=3,
        )
        ax_main.scatter(
            err["_x"], err["gf_proba_covid"],
            s=90, color=COL_ERROR, alpha=0.95,
            edgecolor="black", linewidth=0.8,
            marker=marker, zorder=5,
        )

    ax_main.axhline(0.5, ls="--", lw=1.1, color="#555", alpha=0.8)
    ax_main.text(
        -0.52, 0.52, "decision threshold (0.5)",
        ha="left", va="bottom", fontsize=9, color="#555", style="italic",
    )

    ax_main.set_xticks([0, 1])
    ax_main.set_xticklabels([
        f"True: Healthy  (n={n_healthy})",
        f"True: COVID-19  (n={n_covid})",
    ])
    ax_main.set_xlim(-0.55, 1.55)
    ax_main.set_ylim(-0.05, 1.1)
    ax_main.set_ylabel("Geneformer P(COVID)")

    for xc, n_ok, n_total in [(0, n_healthy_ok, n_healthy), (1, n_covid_ok, n_covid)]:
        pct = 100 * n_ok / n_total
        ax_main.text(
            xc, 1.04, f"{n_ok} / {n_total} correct  ({pct:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color="#222",
        )

    fig.suptitle(
        "Fine-tuned Geneformer — per-cell predictions on 248 held-out classical monocytes",
        fontsize=13, y=0.975, fontweight="bold",
    )
    fig.text(
        0.5, 0.925,
        f"Accuracy {metrics['test_acc']*100:.1f}%   ·   "
        f"Macro-F1 {metrics['test_macro_f1']:.3f}   ·   "
        f"AUC {metrics['test_auc_covid']:.3f}",
        ha="center", va="top",
        fontsize=10.5, color="#555",
    )

    legend_elements_correctness = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_CORRECT,
                   markersize=8, label="correct", markeredgecolor="white"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_ERROR,
                   markersize=9, label="misclassified", markeredgecolor="black"),
    ]
    legend_elements_donor = [
        plt.Line2D([0], [0], marker=MARKER_BY_DONOR[d], color="w",
                   markerfacecolor="#888", markersize=8, label=DONOR_LABEL[d],
                   markeredgecolor="white")
        for d in MARKER_BY_DONOR
    ]
    leg1 = ax_main.legend(
        handles=legend_elements_correctness, loc="center left",
        bbox_to_anchor=(-0.02, 0.42), frameon=False, title="prediction",
        title_fontsize=9, fontsize=9,
    )
    ax_main.add_artist(leg1)
    ax_main.legend(
        handles=legend_elements_donor, loc="center left",
        bbox_to_anchor=(-0.02, 0.18), frameon=False, title="donor",
        title_fontsize=9, fontsize=9,
    )

    bins = np.linspace(0, 1, 26)
    healthy_p = df.loc[df["gf_true_covid"] == 0, "gf_proba_covid"]
    covid_p = df.loc[df["gf_true_covid"] == 1, "gf_proba_covid"]
    ax_hist.hist(
        healthy_p, bins=bins, orientation="horizontal",
        color="#6baed6", alpha=0.85, label=f"Healthy truth (n={n_healthy})",
    )
    ax_hist.hist(
        covid_p, bins=bins, orientation="horizontal",
        color="#fdae6b", alpha=0.85, label=f"COVID truth (n={n_covid})",
    )
    ax_hist.axhline(0.5, ls="--", lw=1.1, color="#555", alpha=0.8)
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("cells  (log)")
    ax_hist.legend(loc="upper right", bbox_to_anchor=(1.0, 0.95), frameon=False, fontsize=8)
    ax_hist.spines["left"].set_visible(False)
    ax_hist.tick_params(left=False)
    ax_hist.set_title("P(COVID) distribution", fontsize=10, pad=6)

    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
