"""
Build a single AnnData from the 4 mtx-exported PBMC samples.

Stages:
  1. Read each sample's mtx + features + barcodes.
  2. Outer-join genes across samples (zero-fill missing).
  3. Filter cells: n_genes >= 200, n_counts >= 500.
  4. Compute mitochondrial percentage (informational; not used by Geneformer).
  5. Map gene symbols -> Ensembl IDs via Geneformer's bundled gc30M dictionary.
  6. Drop genes with no Ensembl ID match.
  7. Save as data/geneformer/build/combined.ensembl.h5ad
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import scipy.sparse as sp

ROOT = Path("/Users/sarafarmahinifarahani/Downloads/single_Cell/Project1")
RAW_DIR = ROOT / "data/geneformer/raw"
BUILD_DIR = ROOT / "data/geneformer/build"
BUILD_DIR.mkdir(parents=True, exist_ok=True)
GENE_NAME_ID_DICT = ROOT / "Geneformer/geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl"

MIN_GENES = 200
MIN_COUNTS = 500


def load_one_sample(sample_dir: Path) -> ad.AnnData:
    """Load matrix.mtx + features.tsv + barcodes.tsv into an AnnData
    (cells in obs, genes in var)."""
    print(f"  loading {sample_dir.name}", flush=True)
    mat = sio.mmread(sample_dir / "matrix.mtx").tocsr()  # genes x cells
    features = pd.read_csv(sample_dir / "features.tsv", header=None, names=["symbol"])
    barcodes = pd.read_csv(sample_dir / "barcodes.tsv", header=None, names=["barcode"])

    a = ad.AnnData(
        X=mat.T.tocsr(),                    # transpose -> cells x genes
        obs=pd.DataFrame(index=barcodes["barcode"].astype(str).values),
        var=pd.DataFrame(index=features["symbol"].astype(str).values),
    )
    return a


def main() -> int:
    print("=== Stage 1: load 4 samples ===", flush=True)
    sample_dirs = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])
    if not sample_dirs:
        print("ERROR: no sample subdirs in", RAW_DIR, file=sys.stderr)
        return 1

    samples = [load_one_sample(d) for d in sample_dirs]

    print("\n=== Stage 2: concatenate (outer join on genes, fill 0) ===", flush=True)
    combined = ad.concat(samples, axis=0, join="outer", fill_value=0,
                         label="from_sample", keys=[d.name for d in sample_dirs],
                         index_unique=None)
    combined.X = sp.csr_matrix(combined.X)
    print(f"  combined: {combined.n_obs:,} cells x {combined.n_vars:,} genes")

    print("\n=== Stage 3: attach per-cell metadata ===", flush=True)
    obs_csv = pd.read_csv(RAW_DIR / "obs.csv").set_index("barcode")
    obs_csv = obs_csv.loc[combined.obs_names]
    for col in ("sample_id", "manuscript_id", "condition", "n_counts", "n_genes"):
        combined.obs[col] = obs_csv[col].values

    print("\n=== Stage 4: compute %MT ===", flush=True)
    mt_mask = combined.var_names.str.upper().str.startswith("MT-")
    print(f"  detected {int(mt_mask.sum())} MT-* genes")
    mt_counts = np.asarray(combined.X[:, mt_mask].sum(axis=1)).ravel()
    combined.obs["pct_mt"] = 100 * mt_counts / np.maximum(combined.obs["n_counts"].values, 1)

    print("\n=== Stage 5: filter cells (n_genes>=200, n_counts>=500) ===", flush=True)
    n_before = combined.n_obs
    keep = (combined.obs["n_genes"] >= MIN_GENES) & (combined.obs["n_counts"] >= MIN_COUNTS)
    combined = combined[keep].copy()
    print(f"  kept {combined.n_obs:,} / {n_before:,} cells "
          f"({100*combined.n_obs/n_before:.1f}%)")

    print("\n  per-sample cell counts after filter:", flush=True)
    print(combined.obs.groupby("sample_id", observed=True).size().to_string())

    print("\n=== Stage 6: map gene symbols -> Ensembl IDs (gc30M dict) ===", flush=True)
    with open(GENE_NAME_ID_DICT, "rb") as fh:
        sym2ens = pickle.load(fh)
    print(f"  dictionary contains {len(sym2ens):,} entries; "
          f"sample: {dict(list(sym2ens.items())[:3])}")

    var_syms = combined.var_names.tolist()
    ens_ids = [sym2ens.get(s) for s in var_syms]
    have_match = [e is not None for e in ens_ids]
    n_match = sum(have_match)
    print(f"  symbols matched to Ensembl: {n_match:,} / {len(var_syms):,} "
          f"({100*n_match/len(var_syms):.1f}%)")

    combined = combined[:, have_match].copy()
    combined.var["ensembl_id"] = [ens_ids[i] for i in range(len(ens_ids)) if have_match[i]]
    combined.var["gene_symbol"] = combined.var_names.values
    combined.var_names = combined.var["ensembl_id"].values

    if combined.var_names.has_duplicates:
        print("  warning: duplicate Ensembl IDs after mapping -- collapsing by sum", flush=True)
        idx = pd.Index(combined.var_names)
        dup_mask = idx.duplicated(keep=False)
        print(f"  {int(dup_mask.sum())} duplicate-rows from {idx[dup_mask].nunique()} ids")

    print(f"\n  final shape: {combined.n_obs:,} cells x {combined.n_vars:,} genes (Ensembl IDs)")

    print("\n=== Stage 7: write h5ad ===", flush=True)
    out_path = BUILD_DIR / "combined.ensembl.h5ad"
    combined.write_h5ad(out_path, compression="gzip")
    size_mb = out_path.stat().st_size / 1e6
    print(f"  wrote {out_path}  ({size_mb:.1f} MB)")

    print("\n=== sample of final obs ===")
    print(combined.obs.head().to_string())
    print("\n=== sample of final var ===")
    print(combined.var.head().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
