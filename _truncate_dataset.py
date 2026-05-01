"""Make a copy of monocytes.dataset with input_ids truncated to MAX_LEN tokens.

Geneformer's tokens are rank-value-encoded — index 0 is the most-expressed gene
in the cell. Truncating to the top-1024 keeps the dominant transcriptional
signal and avoids MPS's pathological scaling at seq_len=2048.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from datasets import load_from_disk

SRC = Path("data/geneformer/tier2/monocytes.dataset")
DST = Path("data/geneformer/tier2/monocytes_t1024.dataset")
MAX_LEN = 1024


def truncate(example):
    ids = example["input_ids"]
    if len(ids) > MAX_LEN:
        example["input_ids"] = ids[:MAX_LEN]
        example["length"] = MAX_LEN
    return example


def main() -> int:
    ds = load_from_disk(str(SRC))
    print(f"loaded {len(ds):,} rows; cols={ds.column_names}")
    print(f"length p50/p95/max = {sorted(ds['length'])[len(ds)//2]}/{sorted(ds['length'])[int(len(ds)*0.95)]}/{max(ds['length'])}")

    ds2 = ds.map(truncate, num_proc=2)
    print(f"after truncation to {MAX_LEN}: p50/p95/max = {sorted(ds2['length'])[len(ds2)//2]}/{sorted(ds2['length'])[int(len(ds2)*0.95)]}/{max(ds2['length'])}")

    if DST.exists():
        shutil.rmtree(DST)
    ds2.save_to_disk(str(DST))
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
