"""Run Geneformer EmbExtractor on a single tokenised chunk dataset.
Designed to be run as a subprocess so MPS memory is fully reclaimed between chunks.
"""
from __future__ import annotations
import argparse
import sys
import time
import warnings

warnings.filterwarnings("ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to tokenised .dataset directory")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    import torch
    print(f"[chunk {args.prefix}] device       : {'mps' if torch.backends.mps.is_available() else 'cpu'}")

    from geneformer import EmbExtractor

    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cell",
        cell_emb_style="mean_pool",
        max_ncells=None,
        nproc=1,
        forward_batch_size=args.batch_size,
        model_version="V1",
        emb_layer=-1,
    )

    t0 = time.time()
    embex.extract_embs(
        model_directory=args.model_dir,
        input_data_file=args.input,
        output_directory=args.output_dir,
        output_prefix=args.prefix,
    )
    print(f"[chunk {args.prefix}] wall time    : {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
