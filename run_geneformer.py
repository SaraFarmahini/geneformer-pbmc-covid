"""
Tier-1 Geneformer pipeline (zero-shot, no fine-tuning).

Stages (each idempotent: rerun is cheap if outputs exist):
  1. Tokenize the combined AnnData -> tokenized HF dataset.
  2. Extract per-cell embeddings using the V1 / 30M pretrained checkpoint.
  3. UMAP + plot embeddings coloured by sample_id, condition, %MT, n_counts.

Inputs:
  data/geneformer/build/combined.ensembl.h5ad      (from build_anndata.py)
  Geneformer/Geneformer-V1-10M/                    (pretrained weights)

Outputs:
  data/geneformer/tokenized/pbmc.dataset/          (tokenized HF dataset)
  data/geneformer/embeddings/emb.csv               (cell embeddings, one row per cell)
  data/geneformer/figures/umap_*.png               (visualisations)
  data/geneformer/build/embedded.h5ad              (AnnData with .obsm['X_geneformer'])
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch

ROOT = Path(__file__).resolve().parent
H5AD_INPUT = ROOT / "data/geneformer/build/combined.ensembl.h5ad"
H5AD_INPUT_DIR = ROOT / "data/geneformer/build"  # tokenizer reads from a directory

TOK_OUT_DIR = ROOT / "data/geneformer/tokenized"
EMB_OUT_DIR = ROOT / "data/geneformer/embeddings"
FIG_DIR     = ROOT / "data/geneformer/figures"
EMBEDDED_H5AD = ROOT / "data/geneformer/build/embedded.h5ad"

MODEL_DIR = ROOT / "Geneformer/Geneformer-V1-10M"
TOK_PREFIX = "pbmc"
EMB_PREFIX = "emb"

for d in (TOK_OUT_DIR, EMB_OUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def stage_tokenize(force: bool = False) -> Path:
    out = TOK_OUT_DIR / f"{TOK_PREFIX}.dataset"
    if out.exists() and not force:
        print(f"[tokenize] skip - already exists: {out}")
        return out

    print(f"[tokenize] device for downstream: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
    print(f"[tokenize] reading from         : {H5AD_INPUT_DIR}")
    print(f"[tokenize] writing to           : {out}")

    from geneformer import TranscriptomeTokenizer

    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={
            "sample_id":     "sample_id",
            "manuscript_id": "manuscript_id",
            "condition":     "condition",
            "n_counts":      "n_counts",
            "n_genes":       "n_genes",
            "pct_mt":        "pct_mt",
        },
        nproc=4,
        model_version="V1",
    )

    only_input = H5AD_INPUT_DIR / "_tokenize_input"
    only_input.mkdir(exist_ok=True)
    target = only_input / H5AD_INPUT.name
    if not target.exists():
        target.symlink_to(H5AD_INPUT)

    tk.tokenize_data(
        data_directory=str(only_input),
        output_directory=str(TOK_OUT_DIR),
        output_prefix=TOK_PREFIX,
        file_format="h5ad",
    )
    return out


def stage_embed(tok_dir: Path, force: bool = False) -> Path:
    out_csv = EMB_OUT_DIR / f"{EMB_PREFIX}.csv"
    if out_csv.exists() and not force:
        print(f"[embed] skip - already exists: {out_csv}")
        return out_csv

    print(f"[embed] using model dir         : {MODEL_DIR}")
    print(f"[embed] tokenized input         : {tok_dir}")

    from geneformer import EmbExtractor

    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cell",
        cell_emb_style="mean_pool",
        max_ncells=None,
        nproc=2,
        forward_batch_size=8,
        model_version="V1",
        emb_layer=-1,
    )

    t0 = time.time()
    embex.extract_embs(
        model_directory=str(MODEL_DIR),
        input_data_file=str(tok_dir),
        output_directory=str(EMB_OUT_DIR),
        output_prefix=EMB_PREFIX,
    )
    print(f"[embed] wall time: {(time.time() - t0)/60:.1f} min")
    return out_csv


def stage_visualize(emb_csv: Path):
    print(f"[viz] loading {emb_csv}")
    emb_df = pd.read_csv(emb_csv, index_col=0)
    print(f"[viz] embedding df shape: {emb_df.shape}")
    X_emb = emb_df.values.astype(np.float32)
    print(f"[viz] embedding matrix: {X_emb.shape}")

    # Geneformer sorts cells by length descending before embedding.
    # We re-create that order from the tokenised HF dataset so we can attach metadata.
    print("[viz] loading tokenised dataset to recover cell order + metadata")
    from datasets import load_from_disk
    tok = load_from_disk(str(TOK_OUT_DIR / f"{TOK_PREFIX}.dataset"))
    print(f"  tokenised dataset: {len(tok):,} rows")

    tok_meta = pd.DataFrame({
        "sample_id":     tok["sample_id"],
        "manuscript_id": tok["manuscript_id"],
        "condition":     tok["condition"],
        "n_counts":      tok["n_counts"],
        "n_genes":       tok["n_genes"],
        "pct_mt":        tok["pct_mt"],
        "length":        tok["length"],
    })
    tok_meta_sorted = tok_meta.sort_values("length", ascending=False, kind="stable").reset_index()
    print(f"  sorted tok metadata head:\n{tok_meta_sorted.head(3).to_string()}")

    if X_emb.shape[0] != len(tok_meta_sorted):
        raise RuntimeError(
            f"embedding rows ({X_emb.shape[0]}) != tokenised cells ({len(tok_meta_sorted)})"
        )

    a = ad.AnnData(
        X=np.zeros((X_emb.shape[0], 1), dtype=np.float32),
        obs=tok_meta_sorted.drop(columns="index"),
    )
    a.obs.index = a.obs.index.astype(str)
    a.obsm["X_geneformer"] = X_emb

    print("[viz] running neighbours + UMAP on Geneformer embeddings")
    sc.pp.neighbors(a, use_rep="X_geneformer", n_neighbors=15)
    sc.tl.umap(a, min_dist=0.3)
    sc.tl.leiden(a, resolution=0.5, key_added="leiden_geneformer")

    a.write_h5ad(EMBEDDED_H5AD, compression="gzip")
    print(f"[viz] wrote {EMBEDDED_H5AD}")

    print("[viz] plotting")
    sc.settings.figdir = FIG_DIR
    for col, name in [
        ("sample_id",         "umap_sample_id.png"),
        ("condition",         "umap_condition.png"),
        ("leiden_geneformer", "umap_leiden.png"),
        ("pct_mt",            "umap_pct_mt.png"),
        ("n_counts",          "umap_n_counts.png"),
        ("n_genes",           "umap_n_genes.png"),
    ]:
        sc.pl.umap(a, color=col, save=f"_{col}.png" if col != "leiden_geneformer" else "_leiden.png",
                   show=False, frameon=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    sc.pl.umap(a, color="condition", ax=axes[0], show=False, frameon=False, title="Condition")
    sc.pl.umap(a, color="sample_id", ax=axes[1], show=False, frameon=False, title="Sample")
    fig.savefig(FIG_DIR / "umap_overview.png", dpi=180, bbox_inches="tight")
    print(f"[viz] wrote {FIG_DIR / 'umap_overview.png'}")

    print("\n[viz] cells per (Leiden cluster, condition):")
    print(pd.crosstab(a.obs["leiden_geneformer"], a.obs["condition"]).to_string())

    print("\n[viz] cells per (Leiden cluster, sample_id):")
    print(pd.crosstab(a.obs["leiden_geneformer"], a.obs["sample_id"]).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["tokenize", "embed", "viz", "all"], default="all")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if not H5AD_INPUT.exists():
        print(f"missing input h5ad: {H5AD_INPUT}", file=sys.stderr)
        return 1

    tok = TOK_OUT_DIR / f"{TOK_PREFIX}.dataset"
    emb = EMB_OUT_DIR / f"{EMB_PREFIX}.csv"

    if args.stage in ("tokenize", "all"):
        tok = stage_tokenize(force=args.force)
    if args.stage in ("embed", "all"):
        emb = stage_embed(tok, force=args.force)
    if args.stage in ("viz", "all"):
        stage_visualize(emb)

    return 0


if __name__ == "__main__":
    sys.exit(main())
