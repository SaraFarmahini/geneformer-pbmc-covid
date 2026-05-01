"""
Tier-2 step 2-4: COVID-vs-healthy classifier within classical monocytes.

Pipeline:
  Step A. Prepare train/test split via Geneformer's Classifier (90/10, stratified
          by `condition`, seed=42). Inside the train set we'll later carve a
          further 80/10 train/val split (so effective 81/9/10 train/val/test).
  Step B. Logistic-regression baseline on log-normalized counts (HVG features),
          using the SAME train/test cells identified by Geneformer's prepare_data.
  Step C. Fine-tune Geneformer V1/30M with a 2-class classification head.
  Step D. Evaluate the fine-tuned model on the held-out test set.
  Step E. Compare baseline vs Geneformer; save metrics + plots.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings

# Must be set before importing geneformer (which reads it at module import time).
os.environ.setdefault("GF_FORCE_DEVICE", "cpu")

from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TIER2_DIR = ROOT / "data/geneformer/tier2"
# Truncated to top-1024 ranked tokens per cell; sequences > 1024 cause MPS to
# scale pathologically (78s/step at seq=2048 vs 467ms/step at seq=1024).
INPUT_DATASET = TIER2_DIR / "monocytes_t1024.dataset"
RUN_DIR = TIER2_DIR / "run_mono_covid"
PREP_DIR = RUN_DIR / "prepared"
FT_DIR = RUN_DIR / "ft"
EVAL_DIR = RUN_DIR / "eval"
BASELINE_DIR = RUN_DIR / "baseline"
COMBINED_H5AD = ROOT / "data/geneformer/build/combined.ensembl.h5ad"
MODEL_DIR = ROOT / "Geneformer/Geneformer-V1-10M"

PREFIX = "mono"
KEY = ["sample_id", "n_counts", "n_genes", "pct_mt"]
SPLIT_PATH = TIER2_DIR / "split_indices.json"


def make_split():
    """Stratified 80/10/10 split (by condition) over monocyte cell_ids.

    Geneformer's `prepare_data` won't auto-stratify because newer `datasets`
    requires a ClassLabel column for stratification, which Geneformer doesn't
    produce. We do the split ourselves and pass it via `split_id_dict`.
    """
    if SPLIT_PATH.exists():
        return json.loads(SPLIT_PATH.read_text())
    from datasets import load_from_disk
    from sklearn.model_selection import train_test_split as sk_split

    ds = load_from_disk(str(INPUT_DATASET))
    cell_ids = list(ds["cell_id"])
    conds = list(ds["condition"])
    train_ids, tmp_ids, train_y, tmp_y = sk_split(
        cell_ids, conds, test_size=0.20, stratify=conds, random_state=42,
    )
    val_ids, test_ids, val_y, test_y = sk_split(
        tmp_ids, tmp_y, test_size=0.50, stratify=tmp_y, random_state=42,
    )
    split = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(split))
    print(f"[split] wrote {SPLIT_PATH}  (train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])})")

    def counts(ids):
        s = pd.Series([conds[cell_ids.index(i)] for i in ids]).value_counts()
        return s.to_dict()
    print(f"  train: {counts(train_ids)}")
    print(f"    val: {counts(val_ids)}")
    print(f"   test: {counts(test_ids)}")
    return split


def stage_prepare():
    """Step A: split monocytes.dataset (90/10 train/test) using our pre-computed split."""
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PREP_DIR / f"{PREFIX}_labeled_train.dataset"
    test_path = PREP_DIR / f"{PREFIX}_labeled_test.dataset"
    id_class_path = PREP_DIR / f"{PREFIX}_id_class_dict.pkl"
    if train_path.exists() and test_path.exists() and id_class_path.exists():
        print(f"[prep] already exists at {PREP_DIR}, skipping.")
        return train_path, test_path, id_class_path

    split = make_split()
    from geneformer import Classifier

    cc = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": "condition", "states": ["COVID-19", "Healthy"]},
        training_args=None,
        freeze_layers=2,
        num_crossval_splits=1,
        split_sizes={"train": 0.8, "valid": 0.1, "test": 0.1},
        stratify_splits_col=None,
        forward_batch_size=8,
        model_version="V1",
        nproc=2,
        ngpu=1,
    )
    cc.prepare_data(
        input_data_file=str(INPUT_DATASET),
        output_directory=str(PREP_DIR),
        output_prefix=PREFIX,
        split_id_dict={
            "attr_key": "cell_id",
            "train": split["train"] + split["val"],
            "test": split["test"],
        },
    )

    print(f"[prep] wrote {train_path}")
    print(f"[prep] wrote {test_path}")
    return train_path, test_path, id_class_path


def stage_baseline(train_path: Path, test_path: Path, id_class_path: Path):
    """Step B: log-reg baseline on the same cells Geneformer will see."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = BASELINE_DIR / "baseline_metrics.json"
    if metrics_path.exists():
        print(f"[baseline] already exists at {metrics_path}, skipping.")
        return json.loads(metrics_path.read_text())

    import scanpy as sc
    from datasets import load_from_disk
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, roc_auc_score, classification_report,
    )

    print("[baseline] loading prepared HF train/test (for cell identities)")
    train_ds = load_from_disk(str(train_path))
    test_ds = load_from_disk(str(test_path))
    print(f"  train cells: {len(train_ds):,}; test cells: {len(test_ds):,}")

    with open(id_class_path, "rb") as fh:
        id_class_dict = pickle.load(fh)
    print(f"  id->class: {id_class_dict}")
    class_id = {v: k for k, v in id_class_dict.items()}
    label_to_int = {"COVID-19": class_id["COVID-19"], "Healthy": class_id["Healthy"]}

    train_meta = pd.DataFrame({k: train_ds[k] for k in KEY})
    test_meta = pd.DataFrame({k: test_ds[k] for k in KEY})
    train_y = np.array(train_ds["label"], dtype=np.int64)
    test_y = np.array(test_ds["label"], dtype=np.int64)

    print("\n[baseline] loading combined.ensembl.h5ad")
    a = sc.read_h5ad(COMBINED_H5AD)
    print(f"  full anndata: {a.n_obs:,} cells x {a.n_vars:,} genes")
    a.var_names = a.var["gene_symbol"].astype(str).values
    a.var_names_make_unique()

    full_meta = a.obs[KEY].copy().reset_index(drop=True)
    full_meta["__row"] = np.arange(len(full_meta))

    def select_rows(meta_subset: pd.DataFrame, label: str) -> np.ndarray:
        m = meta_subset.copy()
        m["__cum"] = m.groupby(KEY).cumcount()
        f = full_meta.copy()
        f["__cum"] = f.groupby(KEY).cumcount()
        merged = pd.merge(m, f, on=KEY + ["__cum"], how="left")
        if merged["__row"].isna().any():
            raise RuntimeError(
                f"baseline {label}: {int(merged['__row'].isna().sum())} cells could not be aligned"
            )
        return merged["__row"].astype(int).to_numpy()

    train_rows = select_rows(train_meta, "train")
    test_rows = select_rows(test_meta, "test")
    print(f"  aligned train rows: {len(train_rows):,}; test rows: {len(test_rows):,}")
    print(f"  no-overlap check: {len(set(train_rows) & set(test_rows))} shared cells (should be 0)")

    a_train = a[train_rows].copy()
    a_test = a[test_rows].copy()

    print("\n[baseline] normalize → log1p")
    sc.pp.normalize_total(a_train, target_sum=1e4)
    sc.pp.log1p(a_train)
    sc.pp.normalize_total(a_test, target_sum=1e4)
    sc.pp.log1p(a_test)

    print("\n[baseline] selecting HVGs on train (top 2000)")
    sc.pp.highly_variable_genes(a_train, n_top_genes=2000, flavor="seurat", subset=False)
    hvg_mask = a_train.var["highly_variable"].values
    print(f"  HVGs: {int(hvg_mask.sum()):,}")

    X_train = a_train.X[:, hvg_mask].toarray()
    X_test = a_test.X[:, hvg_mask].toarray()
    print(f"  X_train: {X_train.shape}; X_test: {X_test.shape}")

    print("\n[baseline] fitting LogisticRegression(class_weight='balanced')")
    clf = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        n_jobs=-1,
    )
    t0 = time.time()
    clf.fit(X_train, train_y)
    print(f"  fit in {time.time()-t0:.1f}s")

    test_proba = clf.predict_proba(X_test)
    test_pred = clf.predict(X_test)

    covid_id = label_to_int["COVID-19"]
    healthy_id = label_to_int["Healthy"]
    is_covid_y_test = (test_y == covid_id).astype(int)
    auc = roc_auc_score(is_covid_y_test, test_proba[:, covid_id])
    acc = accuracy_score(test_y, test_pred)
    macro_f1 = f1_score(test_y, test_pred, average="macro")
    cm = confusion_matrix(test_y, test_pred)

    print(f"\n[baseline] test acc={acc:.4f}  macro_f1={macro_f1:.4f}  AUC(COVID)={auc:.4f}")
    print(f"  confusion matrix (rows=true, cols=pred):\n{cm}")
    print(f"\n  classification_report:\n{classification_report(test_y, test_pred, target_names=[id_class_dict[0], id_class_dict[1]])}")

    feat_imp = pd.DataFrame({
        "gene": a_train.var_names[hvg_mask],
        "coef_covid": clf.coef_[0] if covid_id == 1 else -clf.coef_[0],
    })
    feat_imp = feat_imp.reindex(feat_imp["coef_covid"].abs().sort_values(ascending=False).index)
    top_up = feat_imp.head(20).copy()
    top_dn = feat_imp.tail(20)[::-1].copy()
    print("\n[baseline] top genes UP in COVID (by abs coef):")
    print(top_up.to_string(index=False))
    print("\n[baseline] top genes DOWN in COVID:")
    print(top_dn.to_string(index=False))

    metrics = {
        "test_acc": float(acc),
        "test_macro_f1": float(macro_f1),
        "test_auc_covid": float(auc),
        "confusion_matrix": cm.tolist(),
        "id_class_dict": {int(k): str(v) for k, v in id_class_dict.items()},
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "train_class_counts": {int(k): int(v) for k, v in zip(*np.unique(train_y, return_counts=True))},
        "test_class_counts": {int(k): int(v) for k, v in zip(*np.unique(test_y, return_counts=True))},
        "n_features": int(hvg_mask.sum()),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    feat_imp.to_csv(BASELINE_DIR / "baseline_feature_importance.csv", index=False)
    pd.DataFrame(test_proba, columns=[id_class_dict[0], id_class_dict[1]]).assign(
        true_label=[id_class_dict[i] for i in test_y]
    ).to_csv(BASELINE_DIR / "baseline_test_predictions.csv", index=False)
    print(f"\n[baseline] wrote {metrics_path}")
    return metrics


def stage_finetune(train_path: Path, id_class_path: Path):
    """Step C: fine-tune Geneformer with a 2-class head."""
    FT_DIR.mkdir(parents=True, exist_ok=True)
    if list(FT_DIR.glob("*_geneformer_cellClassifier_*/ksplit1/pytorch_model.bin")):
        print(f"[ft] fine-tuned model already exists under {FT_DIR}, skipping.")
        return

    # Force CPU. MPS unified memory on this machine is ~9 GiB and is being
    # squeezed below 1 GiB by other running apps (saw repeated OOMs at
    # "other allocations: 8 GiB"). The Apple Silicon CPU is fast enough for a
    # 6-layer 256-dim BERT (~1.5 s/step at bs=4 seq=1024 in our benchmark) and
    # avoids the MPS-pool race entirely.
    os.environ["GF_FORCE_DEVICE"] = "cpu"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    from geneformer import Classifier

    training_args = {
        "num_train_epochs": 4,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 8,
        "learning_rate": 5e-5,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "logging_steps": 50,
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "group_by_length": True,
        "length_column_name": "length",
        "report_to": "none",
        "dataloader_num_workers": 0,
        "use_cpu": True,
    }

    cc = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": "condition", "states": ["COVID-19", "Healthy"]},
        training_args=training_args,
        freeze_layers=2,
        num_crossval_splits=1,
        split_sizes={"train": 0.8, "valid": 0.1, "test": 0.1},
        stratify_splits_col=None,
        forward_batch_size=8,
        model_version="V1",
        nproc=2,
        ngpu=1,
    )

    split = make_split()
    print("[ft] starting Classifier.validate (this is the actual training)")
    t0 = time.time()
    metrics = cc.validate(
        model_directory=str(MODEL_DIR),
        prepared_input_data_file=str(train_path),
        id_class_dict_file=str(id_class_path),
        output_directory=str(FT_DIR),
        output_prefix=PREFIX,
        split_id_dict={
            "attr_key": "cell_id",
            "train": split["train"],
            "eval": split["val"],
        },
        predict_eval=True,
    )
    print(f"[ft] training+val time: {(time.time()-t0)/60:.1f} min")
    print(f"[ft] training metrics: {{macro_f1: {metrics['macro_f1']}, acc: {metrics['acc']}}}")
    if metrics.get("all_roc_metrics"):
        print(f"  ROC AUC: {metrics['all_roc_metrics']['roc_auc']:.4f}")
    return metrics


def stage_test_eval(test_path: Path, id_class_path: Path):
    """Step D: evaluate the fine-tuned model on held-out test set."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = EVAL_DIR / "test_metrics.json"
    if metrics_path.exists():
        print(f"[test] already exists at {metrics_path}, skipping.")
        return json.loads(metrics_path.read_text())

    saved_models = sorted(FT_DIR.glob("*_geneformer_cellClassifier_*/ksplit1"))
    if not saved_models:
        raise FileNotFoundError(f"no fine-tuned model found under {FT_DIR}")
    saved_model_dir = saved_models[-1]
    print(f"[test] using fine-tuned model: {saved_model_dir}")

    from geneformer import Classifier

    cc = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": "condition", "states": ["COVID-19", "Healthy"]},
        training_args=None,
        freeze_layers=2,
        num_crossval_splits=1,
        split_sizes={"train": 0.8, "valid": 0.1, "test": 0.1},
        stratify_splits_col="label",
        forward_batch_size=8,
        model_version="V1",
        nproc=2,
        ngpu=1,
    )

    t0 = time.time()
    result = cc.evaluate_saved_model(
        model_directory=str(saved_model_dir),
        id_class_dict_file=str(id_class_path),
        test_data_file=str(test_path),
        output_directory=str(EVAL_DIR),
        output_prefix=f"{PREFIX}_test",
        predict=True,
        predict_metadata=["sample_id", "n_counts", "n_genes", "pct_mt"],
    )
    print(f"[test] eval time: {(time.time()-t0)/60:.1f} min")

    with open(id_class_path, "rb") as fh:
        id_class_dict = pickle.load(fh)

    import pandas as _pd

    cm = result["conf_matrix"]
    if isinstance(cm, _pd.DataFrame):
        cm_dict = {"index": cm.index.tolist(), "columns": cm.columns.tolist(), "values": cm.values.tolist()}
    elif hasattr(cm, "tolist"):
        cm_dict = {"values": cm.tolist()}
    else:
        cm_dict = {"values": cm}

    auc_val = None
    rm = result.get("all_roc_metrics") or {}
    if rm and "all_roc_auc" in rm:
        v = rm["all_roc_auc"]
        auc_val = float(v) if not hasattr(v, "__iter__") else float(v[0])
    out = {
        "test_acc": float(result["acc"]),
        "test_macro_f1": float(result["macro_f1"]),
        "test_auc_covid": auc_val,
        "confusion_matrix": cm_dict,
        "id_class_dict": {int(k): str(v) for k, v in id_class_dict.items()},
    }
    metrics_path.write_text(json.dumps(out, indent=2))
    print(f"[test] wrote {metrics_path}")
    print(f"[test] test acc={out['test_acc']:.4f}  macro_f1={out['test_macro_f1']:.4f}  AUC={out['test_auc_covid']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="prep,baseline,ft,test", help="comma-separated steps to run")
    args = ap.parse_args()
    steps = [s.strip() for s in args.steps.split(",")]

    train_path, test_path, id_class_path = (
        PREP_DIR / f"{PREFIX}_labeled_train.dataset",
        PREP_DIR / f"{PREFIX}_labeled_test.dataset",
        PREP_DIR / f"{PREFIX}_id_class_dict.pkl",
    )

    if "prep" in steps:
        train_path, test_path, id_class_path = stage_prepare()

    if "baseline" in steps:
        stage_baseline(train_path, test_path, id_class_path)

    if "ft" in steps:
        stage_finetune(train_path, id_class_path)

    if "test" in steps:
        stage_test_eval(test_path, id_class_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
