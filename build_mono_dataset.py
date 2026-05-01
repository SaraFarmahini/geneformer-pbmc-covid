"""
Tier-2 step 1: build a labelled tokenised dataset of classical monocytes.

Source data:
  - data/geneformer/tokenized/pbmc.dataset    (28,036 cells, rank-value-encoded,
                                                cols incl. `condition`, `sample_id`,
                                                `n_counts`, `n_genes`, `pct_mt`, `length`)
  - data/geneformer/build/embedded_annotated.h5ad  (same 28,036 cells in length-DESC
                                                    order, with `leiden_geneformer`
                                                    and `celltype_curated` per cell)

We sort the tokenised dataset the same way (length DESC) so its row-order matches
embedded.obs row-order, attach the cluster id, then filter to `leiden ∈ {5, 13}`
(curated label = "Classical monocytes"). The resulting dataset has the
`condition` column intact so Geneformer's Classifier can use it directly.

Outputs:
  data/geneformer/tier2/monocytes.dataset
  data/geneformer/tier2/monocytes_meta.csv      (per-cell metadata for baseline)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
from datasets import load_from_disk

ROOT = Path("/Users/sarafarmahinifarahani/Downloads/single_Cell/Project1")
TOK_PATH = ROOT / "data/geneformer/tokenized/pbmc.dataset"
EMB_PATH = ROOT / "data/geneformer/build/embedded_annotated.h5ad"
OUT_DIR = ROOT / "data/geneformer/tier2"
OUT_DATASET = OUT_DIR / "monocytes.dataset"
OUT_META = OUT_DIR / "monocytes_meta.csv"

MONO_CLUSTERS = {"5", "13"}  # curated "Classical monocytes"
KEY = ["sample_id", "n_counts", "n_genes", "pct_mt"]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== load tokenised dataset ===", flush=True)
    tok = load_from_disk(str(TOK_PATH))
    print(f"  rows: {len(tok):,}; cols: {tok.column_names}")

    print("\n=== sort by length DESC (matches embedded.obs order) ===", flush=True)
    tok_sorted = tok.sort("length", reverse=True)
    print(f"  first row length: {tok_sorted[0]['length']}; last row length: {tok_sorted[-1]['length']}")

    print("\n=== load embedded_annotated.h5ad ===", flush=True)
    emb = ad.read_h5ad(EMB_PATH)
    print(f"  rows: {emb.n_obs:,}")
    if emb.n_obs != len(tok_sorted):
        print("ERROR: embedded vs tokenised row count mismatch", file=sys.stderr)
        return 1

    # Check sort alignment by comparing sample_id / n_counts on a few rows.
    tok_keys = pd.DataFrame(
        {k: tok_sorted[k] for k in KEY + ["length"]}
    )
    emb_keys = emb.obs[KEY + ["length"]].reset_index(drop=True)
    eq = (tok_keys.values == emb_keys.values).all(axis=1).mean()
    print(f"  per-row metadata equality: {eq*100:.2f}%")
    if eq < 0.999:
        print("ERROR: row-order mismatch between sorted-tok and embedded.obs",
              file=sys.stderr)
        # Show a small diff for debugging
        bad = (tok_keys.values != emb_keys.values).any(axis=1)
        print("  first mismatched rows:")
        print(pd.concat([tok_keys[bad].head(3), emb_keys[bad].head(3)], axis=1))
        return 2

    print("\n=== add leiden_geneformer + celltype_curated columns ===", flush=True)
    tok_sorted = tok_sorted.add_column(
        "leiden_geneformer",
        emb.obs["leiden_geneformer"].astype(str).tolist(),
    )
    tok_sorted = tok_sorted.add_column(
        "celltype_curated",
        emb.obs["celltype_curated"].astype(str).tolist(),
    )

    print("\n=== filter to classical monocytes (Leiden c5 + c13) ===", flush=True)
    mono = tok_sorted.filter(
        lambda ex: ex["leiden_geneformer"] in MONO_CLUSTERS,
        num_proc=2,
    )
    print(f"  monocytes: {len(mono):,} cells")
    print("  per-condition:")
    print(pd.Series(mono["condition"]).value_counts().to_string())
    print("  per-sample:")
    print(pd.Series(mono["sample_id"]).value_counts().to_string())
    print("  per-leiden:")
    print(pd.Series(mono["leiden_geneformer"]).value_counts().to_string())

    print("\n=== add cell_id column (used for split_id_dict-based splitting) ===", flush=True)
    mono = mono.add_column("cell_id", [f"c{i:06d}" for i in range(len(mono))])

    print("\n=== save ===", flush=True)
    if OUT_DATASET.exists():
        import shutil
        shutil.rmtree(OUT_DATASET)
    mono.save_to_disk(str(OUT_DATASET))
    print(f"  wrote {OUT_DATASET}")

    meta = pd.DataFrame({
        "cell_id":           mono["cell_id"],
        "sample_id":         mono["sample_id"],
        "manuscript_id":     mono["manuscript_id"],
        "condition":         mono["condition"],
        "n_counts":          mono["n_counts"],
        "n_genes":           mono["n_genes"],
        "pct_mt":            mono["pct_mt"],
        "length":            mono["length"],
        "leiden_geneformer": mono["leiden_geneformer"],
    })
    meta.to_csv(OUT_META, index=False)
    print(f"  wrote {OUT_META}  ({len(meta):,} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
