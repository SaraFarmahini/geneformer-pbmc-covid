"""Visualize what the monocyte COVID-vs-Healthy classifier learned.

Reads the sorted logistic-regression coefficient table from
data/geneformer/tier2/run_mono_covid/baseline/baseline_feature_importance.csv,
takes the top 20 genes by |coef|, and renders a single horizontal bar chart
with bars coloured by direction (COVID-tilting vs Healthy-tilting) and gene
family annotations highlighting the type-I-IFN-stimulated genes, S100
alarmins, AP-1/immediate-early response genes, and plasmablast-leakage
immunoglobulins.

Output: data/geneformer/tier2/run_mono_covid/comparison/feature_importance.png

Note on attribution: these coefficients are from the logistic-regression
baseline. They serve as a proxy for what the fine-tuned Geneformer learned
because the two models agree on 96.4% of test cells (see agreement.png).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/sarafarmahinifarahani/Downloads/single_Cell/Project1")
RUN = ROOT / "data/geneformer/tier2/run_mono_covid"
CSV = RUN / "baseline" / "baseline_feature_importance.csv"
OUT = RUN / "comparison" / "feature_importance.png"

TOP_N = 20

GENE_FAMILIES: dict[str, dict[str, str]] = {
    "ISG": {
        "label": "Type-I IFN-stimulated",
        "color": "#d94801",
        "genes": "IFI27 IFI44L IFI6 IFITM3 MX1 ISG15 OAS1 OAS2 OAS3 IFIT1 IFIT3",
    },
    "S100": {
        "label": "S100 alarmins",
        "color": "#cc4c02",
        "genes": "S100A8 S100A9 S100A12 S100A6 S100A4",
    },
    "AP1": {
        "label": "AP-1 / immediate-early response",
        "color": "#225ea8",
        "genes": "FOS FOSB JUN JUNB EGR1 EGR2 ATF3 NR4A1 NR4A2",
    },
    "IG": {
        "label": "Plasmablast Ig leakage",
        "color": "#88419d",
        "genes": "IGHM IGHG1 IGHG3 IGHA1 IGKC IGLC1 IGLC2 IGLC3 IGLC7 JCHAIN",
    },
}

GENE_TO_FAMILY: dict[str, str] = {}
for fam, info in GENE_FAMILIES.items():
    for g in info["genes"].split():
        GENE_TO_FAMILY[g] = fam


def family_for(gene: str) -> str | None:
    return GENE_TO_FAMILY.get(gene)


def main() -> None:
    df = pd.read_csv(CSV)
    df["abscoef"] = df["coef_covid"].abs()
    top = df.nlargest(TOP_N, "abscoef").sort_values("coef_covid", ascending=True).reset_index(drop=True)

    n = len(top)
    bar_color = ["#d94801" if c > 0 else "#225ea8" for c in top["coef_covid"]]

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.titleweight": "bold",
        "figure.dpi": 140,
    })

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 7.4))
    fig.subplots_adjust(top=0.80, left=0.18, right=0.92, bottom=0.14)

    y = np.arange(n)
    ax.barh(
        y, top["coef_covid"], color=bar_color, alpha=0.92,
        edgecolor="white", linewidth=0.6, height=0.72, zorder=2,
    )
    ax.axvline(0, color="#222", lw=0.9, zorder=1)

    ax.set_yticks(y)
    tick_colors = []
    tick_weights = []
    for g in top["gene"]:
        fam = family_for(g)
        if fam:
            tick_colors.append(GENE_FAMILIES[fam]["color"])
            tick_weights.append("bold")
        else:
            tick_colors.append("#222")
            tick_weights.append("normal")
    ax.set_yticklabels(top["gene"], fontsize=10.5, family="monospace")
    for tick, color, weight in zip(ax.get_yticklabels(), tick_colors, tick_weights):
        tick.set_color(color)
        tick.set_weight(weight)

    for i, c in enumerate(top["coef_covid"]):
        offset = 0.025 if c > 0 else -0.025
        ha = "left" if c > 0 else "right"
        ax.text(
            c + offset, i, f"{c:+.2f}", va="center", ha=ha,
            fontsize=9, color="#444",
        )

    xmax = float(top["coef_covid"].abs().max()) * 1.22
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-2.0, n + 0.6)
    ax.set_xlabel("logistic-regression coefficient", labelpad=6)

    ax.text(
        -xmax * 0.5, n + 0.2, "← tilts toward Healthy",
        ha="center", va="center",
        fontsize=11, color="#08519c", weight="bold", style="italic",
    )
    ax.text(
        xmax * 0.5, n + 0.2, "tilts toward COVID-19 →",
        ha="center", va="center",
        fontsize=11, color="#a63603", weight="bold", style="italic",
    )

    fig.suptitle(
        "What the monocyte COVID-vs-Healthy classifier learned",
        fontsize=14, y=0.965, fontweight="bold",
    )
    fig.text(
        0.5, 0.905,
        f"Top {TOP_N} genes by |coefficient| — logistic regression on 2,488 classical monocytes (c5 + c13)",
        ha="center", va="top", fontsize=10.5, color="#555",
    )
    fig.text(
        0.5, 0.875,
        "Geneformer ↔ baseline per-cell agreement: 96.4% — these features serve as an interpretable proxy for what Geneformer is using",
        ha="center", va="top", fontsize=9.5, color="#777", style="italic",
    )

    legend_handles = [
        mpatches.Patch(facecolor=info["color"], edgecolor="none", label=info["label"])
        for info in GENE_FAMILIES.values()
    ]
    ax.legend(
        handles=legend_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=4,
        title="gene families (gene name colour)", title_fontsize=9, fontsize=9,
        handlelength=1.2,
    )

    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
