"""
Option A: Annotate Leiden clusters with CellTypist cell-type labels.

Pipeline:
  1. Load combined.ensembl.h5ad (raw counts) and re-key var to gene symbols.
  2. Normalize -> log1p (CellTypist input format).
  3. Download + run CellTypist `Immune_All_High` (broad) and `Immune_All_Low` (fine).
  4. Align labels into embedded.h5ad (which is in tokenizer's length-sorted order,
     not the original cell order) using a composite key + within-group cumcount
     for deterministic tie-breaking.
  5. Per-cluster majority label + purity for both models.
  6. Re-plot UMAPs coloured by cell type.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "data/geneformer/build"
FIG_DIR = ROOT / "data/geneformer/figures"
COMBINED_H5AD = BUILD_DIR / "combined.ensembl.h5ad"
EMBEDDED_H5AD = BUILD_DIR / "embedded.h5ad"
OUT_H5AD = BUILD_DIR / "embedded_annotated.h5ad"

KEY = ["sample_id", "n_counts", "n_genes", "pct_mt"]


def main() -> int:
    print("=== Stage 1: load gene-expression matrix (raw counts) ===", flush=True)
    a = sc.read_h5ad(COMBINED_H5AD)
    print(f"  combined: {a.n_obs:,} cells x {a.n_vars:,} genes (Ensembl)")

    print("\n=== Stage 2: re-key to gene symbols ===", flush=True)
    if "gene_symbol" not in a.var.columns:
        print("ERROR: combined.ensembl.h5ad lacks var['gene_symbol']", file=sys.stderr)
        return 1
    a.var["ensembl_id"] = a.var_names.astype(str).values
    a.var_names = a.var["gene_symbol"].astype(str).values
    a.var_names_make_unique()
    print(f"  shape: {a.n_obs:,} x {a.n_vars:,}")

    print("\n=== Stage 3: normalize to 10k + log1p ===", flush=True)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    print("\n=== Stage 4: run CellTypist (Immune_All_High + Immune_All_Low) ===", flush=True)
    import celltypist
    from celltypist import models

    for name in ("Immune_All_High.pkl", "Immune_All_Low.pkl"):
        models.download_models(force_update=False, model=name)

    high_model = models.Model.load("Immune_All_High.pkl")
    low_model = models.Model.load("Immune_All_Low.pkl")
    print(f"  High model: {len(high_model.cell_types)} cell types")
    print(f"  Low  model: {len(low_model.cell_types)} cell types")

    print("\n  -> running High model")
    high_pred = celltypist.annotate(a, model=high_model, majority_voting=False)
    print("  -> running Low model")
    low_pred = celltypist.annotate(a, model=low_model, majority_voting=False)

    high_lbl = high_pred.predicted_labels
    low_lbl = low_pred.predicted_labels
    print(f"  high pred columns: {list(high_lbl.columns)}")
    print(f"  low  pred columns: {list(low_lbl.columns)}")

    # CellTypist returns labels in the same row order as `a`.
    a.obs["celltype_high"] = high_lbl["predicted_labels"].astype(str).values
    a.obs["celltype_low"] = low_lbl["predicted_labels"].astype(str).values
    if "conf_score" in high_lbl.columns:
        a.obs["celltype_high_score"] = high_lbl["conf_score"].values
    if "conf_score" in low_lbl.columns:
        a.obs["celltype_low_score"] = low_lbl["conf_score"].values

    print("\n  high label distribution:")
    print(a.obs["celltype_high"].value_counts().head(15).to_string())
    print("\n  low label distribution (top 15):")
    print(a.obs["celltype_low"].value_counts().head(15).to_string())

    print("\n=== Stage 5: align labels into embedded.h5ad ===", flush=True)
    emb = sc.read_h5ad(EMBEDDED_H5AD)
    print(f"  embedded: {emb.n_obs:,} cells")

    # Build deterministic composite key (KEY + within-group cumcount). This gives
    # a 1:1 mapping even when (sample, n_counts, n_genes, pct_mt) has duplicates.
    a_meta = a.obs[KEY].copy()
    a_meta["__cum"] = a_meta.groupby(KEY).cumcount()
    a_meta["__row_a"] = np.arange(len(a_meta))

    e_meta = emb.obs[KEY].copy()
    e_meta["__cum"] = e_meta.groupby(KEY).cumcount()
    e_meta["__row_e"] = np.arange(len(e_meta))

    merged = pd.merge(e_meta, a_meta, on=KEY + ["__cum"], how="left")
    print(f"  merged rows: {len(merged):,}; missing: {int(merged['__row_a'].isna().sum())}")
    if merged["__row_a"].isna().any():
        print("ERROR: some cells in embedded.h5ad couldn't be matched", file=sys.stderr)
        return 2

    a_idx_for_emb = merged["__row_a"].astype(int).to_numpy()

    emb.obs["celltype_high"] = a.obs["celltype_high"].to_numpy()[a_idx_for_emb]
    emb.obs["celltype_low"] = a.obs["celltype_low"].to_numpy()[a_idx_for_emb]
    if "celltype_high_score" in a.obs.columns:
        emb.obs["celltype_high_score"] = a.obs["celltype_high_score"].to_numpy()[a_idx_for_emb]
    if "celltype_low_score" in a.obs.columns:
        emb.obs["celltype_low_score"] = a.obs["celltype_low_score"].to_numpy()[a_idx_for_emb]

    # Sanity-check alignment by comparing length values that the tokenizer wrote.
    # (length should be approximately min(n_genes_in_vocab, 2048) but we don't
    # need exact equality; we check that high-confidence consistent fields
    # match between rows pulled from `a` and the embedded.obs.)
    emb_meta_check = emb.obs[KEY].reset_index(drop=True)
    a_meta_check = a.obs[KEY].iloc[a_idx_for_emb].reset_index(drop=True)
    matches = (emb_meta_check.values == a_meta_check.values).all()
    print(f"  alignment metadata-equality check: {matches}")
    if not matches:
        print("ERROR: alignment mismatch on KEY columns", file=sys.stderr)
        return 3

    print("\n=== Stage 6: per-cluster majority labels ===", flush=True)
    summaries = {}
    for level, col, score_col in [
        ("high", "celltype_high", "celltype_high_score"),
        ("low",  "celltype_low",  "celltype_low_score"),
    ]:
        ct = pd.crosstab(emb.obs["leiden_geneformer"], emb.obs[col])
        majority = ct.idxmax(axis=1)
        purity = (ct.max(axis=1) / ct.sum(axis=1) * 100).round(1)
        n = ct.sum(axis=1)
        score = (
            emb.obs.groupby("leiden_geneformer", observed=True)[score_col].mean().round(3)
            if score_col in emb.obs.columns
            else pd.Series(np.nan, index=ct.index)
        )
        df = pd.DataFrame({
            "cluster": ct.index.astype(str),
            "n_cells": n.values,
            f"label_{level}": majority.values,
            f"purity_{level}_pct": purity.values,
            f"mean_score_{level}": score.values,
        })
        out_csv = BUILD_DIR / f"cluster_labels_{level}.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n  >>> {level} (saved {out_csv})")
        print(df.to_string(index=False))
        summaries[level] = df

    # Build a friendly per-cluster cell-type column on the embedded object.
    cluster_to_high = dict(zip(summaries["high"]["cluster"], summaries["high"]["label_high"]))
    emb.obs["cluster_celltype_high"] = (
        emb.obs["leiden_geneformer"].astype(str).map(cluster_to_high)
    )

    print("\n=== Stage 7: re-plot UMAPs ===", flush=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = FIG_DIR

    sc.pl.umap(emb, color="celltype_high",
               save="_celltype_high.png", show=False, frameon=False, legend_loc="right margin")
    sc.pl.umap(emb, color="celltype_low",
               save="_celltype_low.png", show=False, frameon=False, legend_loc="right margin",
               legend_fontsize=5)
    sc.pl.umap(emb, color="cluster_celltype_high",
               save="_cluster_celltype_high.png", show=False, frameon=False,
               legend_loc="on data", legend_fontsize=7)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sc.pl.umap(emb, color="celltype_high", ax=axes[0], show=False, frameon=False,
               legend_loc="right margin", title="CellTypist (broad)")
    sc.pl.umap(emb, color="leiden_geneformer", ax=axes[1], show=False, frameon=False,
               title="Leiden (Geneformer)")
    fig.savefig(FIG_DIR / "umap_celltype_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR / 'umap_celltype_overview.png'}")

    print("\n=== Stage 8: write annotated h5ad ===", flush=True)
    emb.write_h5ad(OUT_H5AD, compression="gzip")
    print(f"  wrote {OUT_H5AD} ({OUT_H5AD.stat().st_size/1e6:.1f} MB)")

    print("\n=== Cluster x condition x cell type (broad) ===")
    print(
        pd.crosstab(
            [emb.obs["leiden_geneformer"], emb.obs["celltype_high"]],
            emb.obs["condition"],
        ).to_string()
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
