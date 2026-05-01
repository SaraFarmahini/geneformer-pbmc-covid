"""Driver: split the tokenised dataset into chunks (already sorted by length DESC,
matching Geneformer's internal sort), embed each in a subprocess, then concatenate.

This is robust to MPS memory leaks because each chunk runs in a fresh process.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
from datasets import load_from_disk

ROOT = Path("/Users/sarafarmahinifarahani/Downloads/single_Cell/Project1")
TOK_DATASET = ROOT / "data/geneformer/tokenized/pbmc.dataset"
CHUNKS_DIR  = ROOT / "data/geneformer/tokenized/chunks"
EMB_OUT_DIR = ROOT / "data/geneformer/embeddings"
WORKER      = ROOT / "embed_one_chunk.py"
MODEL_DIR   = ROOT / "Geneformer/Geneformer-V1-10M"

CHUNK_SIZE = 4000
PYTHON     = os.path.expanduser("~/anaconda3/envs/geneformer/bin/python")


def split_into_chunks() -> list[Path]:
    """Pre-sort by length DESC and split into uniform chunks."""
    if CHUNKS_DIR.exists():
        existing = sorted(CHUNKS_DIR.glob("chunk_*.dataset"))
        if existing:
            print(f"[split] reusing existing {len(existing)} chunks under {CHUNKS_DIR}")
            return existing
        shutil.rmtree(CHUNKS_DIR)

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[split] loading tokenised dataset from {TOK_DATASET}")
    tok = load_from_disk(str(TOK_DATASET))
    print(f"[split] sorting {len(tok):,} cells by length DESC")
    tok = tok.sort("length", reverse=True)

    chunk_paths: list[Path] = []
    n = len(tok)
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(n_chunks):
        start = i * CHUNK_SIZE
        end   = min(start + CHUNK_SIZE, n)
        sub   = tok.select(range(start, end))
        cp    = CHUNKS_DIR / f"chunk_{i:02d}.dataset"
        sub.save_to_disk(str(cp))
        chunk_paths.append(cp)
        print(f"[split] wrote {cp.name}: rows {start:,}..{end-1:,} (length max={sub[0]['length']}, min={sub[-1]['length']})")
    return chunk_paths


def embed_chunk(chunk_path: Path, prefix: str) -> Path:
    out_csv = EMB_OUT_DIR / f"{prefix}.csv"
    if out_csv.exists():
        print(f"[embed] skip {prefix}: {out_csv} already exists")
        return out_csv

    EMB_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[embed] {prefix}: subprocess on {chunk_path.name}")
    env = os.environ.copy()
    env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    env["PYTORCH_ENABLE_MPS_FALLBACK"]      = "1"
    env.pop("PYTHONPATH", None)

    t0 = time.time()
    res = subprocess.run(
        [PYTHON, str(WORKER),
         "--input", str(chunk_path),
         "--output_dir", str(EMB_OUT_DIR),
         "--prefix", prefix,
         "--model_dir", str(MODEL_DIR)],
        env=env,
        cwd=str(ROOT),
    )
    if res.returncode != 0:
        raise RuntimeError(f"chunk {prefix} subprocess failed (returncode={res.returncode})")
    print(f"[embed] {prefix} subprocess time: {(time.time()-t0)/60:.1f} min")
    return out_csv


def concat_chunks(chunk_csvs: list[Path]) -> Path:
    print("[concat] combining chunk CSVs")
    dfs = [pd.read_csv(p, index_col=0) for p in chunk_csvs]
    combined = pd.concat(dfs, axis=0, ignore_index=True)
    out = EMB_OUT_DIR / "emb.csv"
    combined.to_csv(out)
    total_rows = sum(len(d) for d in dfs)
    print(f"[concat] {len(chunk_csvs)} chunks -> {len(combined):,} rows (sum {total_rows:,}) -> {out}")
    return out


def main() -> int:
    chunk_paths = split_into_chunks()
    print(f"[main] {len(chunk_paths)} chunks ready")

    chunk_csvs: list[Path] = []
    for i, cp in enumerate(chunk_paths):
        prefix = f"chunk_{i:02d}"
        try:
            csv_path = embed_chunk(cp, prefix)
        except RuntimeError as e:
            print(f"[main] FAILED on chunk {i}: {e}", file=sys.stderr)
            return 2
        chunk_csvs.append(csv_path)

    concat_chunks(chunk_csvs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
