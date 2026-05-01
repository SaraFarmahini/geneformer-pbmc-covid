"""
Tier-2 step E: compare logistic-regression baseline vs fine-tuned Geneformer.

Reads:
  data/geneformer/tier2/run_mono_covid/baseline/baseline_metrics.json
  data/geneformer/tier2/run_mono_covid/baseline/baseline_test_predictions.csv
  data/geneformer/tier2/run_mono_covid/eval/test_metrics.json
  data/geneformer/tier2/run_mono_covid/eval/<datestamp>_geneformer_cellClassifier_mono_test/mono_test_pred_dict.pkl

Writes:
  data/geneformer/tier2/run_mono_covid/comparison/comparison.json
  data/geneformer/tier2/run_mono_covid/comparison/per_cell_predictions.csv
  data/geneformer/tier2/run_mono_covid/comparison/agreement.png
  data/geneformer/tier2/run_mono_covid/comparison/roc_curves.png
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path("/Users/sarafarmahinifarahani/Downloads/single_Cell/Project1")
RUN_DIR = ROOT / "data/geneformer/tier2/run_mono_covid"
PREP_DIR = RUN_DIR / "prepared"
EVAL_DIR = RUN_DIR / "eval"
BASELINE_DIR = RUN_DIR / "baseline"
OUT_DIR = RUN_DIR / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_geneformer_predictions():
    """Find the most recent mono_test_pred_dict.pkl under EVAL_DIR."""
    pred_files = sorted(EVAL_DIR.glob("**/mono_test_pred_dict.pkl"))
    if not pred_files:
        raise FileNotFoundError(f"no Geneformer test predictions under {EVAL_DIR}")
    pred_path = pred_files[-1]
    with open(pred_path, "rb") as fh:
        d = pickle.load(fh)
    print(f"[load] geneformer predictions: {pred_path}")
    return d, pred_path


def main() -> int:
    print("=== load id_class_dict ===", flush=True)
    with open(PREP_DIR / "mono_id_class_dict.pkl", "rb") as fh:
        id_class_dict = pickle.load(fh)
    class_id = {v: k for k, v in id_class_dict.items()}
    covid_id = class_id["COVID-19"]
    healthy_id = class_id["Healthy"]
    print(f"  id_class_dict: {id_class_dict}")
    print(f"  COVID-19 -> {covid_id}; Healthy -> {healthy_id}")

    print("\n=== load baseline ===", flush=True)
    bl_metrics = json.loads((BASELINE_DIR / "baseline_metrics.json").read_text())
    bl_preds = pd.read_csv(BASELINE_DIR / "baseline_test_predictions.csv")
    bl_proba_covid = bl_preds["COVID-19"].values
    bl_true = (bl_preds["true_label"] == "COVID-19").astype(int).values
    print(f"  baseline AUC(COVID): {bl_metrics['test_auc_covid']:.4f}; macro_f1={bl_metrics['test_macro_f1']:.4f}")

    print("\n=== load geneformer ===", flush=True)
    gf, gf_path = load_geneformer_predictions()
    gf_pred_ids = np.asarray(gf["pred_ids"])
    gf_label_ids = np.asarray(gf["label_ids"])
    gf_logits = np.asarray(gf["predictions"])
    gf_meta = gf.get("prediction_metadata", None)
    print(f"  pred_ids shape: {gf_pred_ids.shape}; logits shape: {gf_logits.shape}")
    if gf_meta is not None:
        print(f"  metadata keys: {list(gf_meta.keys()) if isinstance(gf_meta, dict) else 'list'}")

    # convert logits -> softmax probabilities
    gf_proba = np.exp(gf_logits - gf_logits.max(axis=1, keepdims=True))
    gf_proba = gf_proba / gf_proba.sum(axis=1, keepdims=True)
    gf_proba_covid = gf_proba[:, covid_id]
    gf_true = (gf_label_ids == covid_id).astype(int)
    gf_pred_covid = (gf_pred_ids == covid_id).astype(int)

    gf_acc = accuracy_score(gf_label_ids, gf_pred_ids)
    gf_macro_f1 = f1_score(gf_label_ids, gf_pred_ids, average="macro")
    gf_auc = roc_auc_score(gf_true, gf_proba_covid)
    gf_cm = confusion_matrix(gf_label_ids, gf_pred_ids, labels=[healthy_id, covid_id])
    print(f"\n  geneformer test:")
    print(f"    acc={gf_acc:.4f}  macro_f1={gf_macro_f1:.4f}  AUC(COVID)={gf_auc:.4f}")
    print(f"    confusion (rows=true [Healthy,COVID], cols=pred):")
    print(f"      {gf_cm}")
    print(f"    classification_report:")
    print(classification_report(gf_label_ids, gf_pred_ids,
                                target_names=[id_class_dict[healthy_id], id_class_dict[covid_id]]))

    # Align baseline and Geneformer per-cell predictions by metadata
    print("\n=== align per-cell predictions ===", flush=True)
    if gf_meta is not None and isinstance(gf_meta, dict) and "sample_id" in gf_meta:
        gf_df = pd.DataFrame({
            "sample_id": gf_meta["sample_id"],
            "n_counts": gf_meta["n_counts"],
            "n_genes": gf_meta["n_genes"],
            "pct_mt": gf_meta["pct_mt"],
            "gf_proba_covid": gf_proba_covid,
            "gf_pred_covid": gf_pred_covid,
            "gf_true_covid": gf_true,
        })
    else:
        print("  (no metadata; alignment by row order)")
        gf_df = pd.DataFrame({
            "gf_proba_covid": gf_proba_covid,
            "gf_pred_covid": gf_pred_covid,
            "gf_true_covid": gf_true,
        })

    bl_df = pd.DataFrame({
        "bl_proba_covid": bl_proba_covid,
        "bl_pred_covid": (bl_preds["COVID-19"] > 0.5).astype(int).values,
        "bl_true_covid": bl_true,
    })

    if len(bl_df) == len(gf_df):
        # Both came from the same test set in the same order; cross-check truth matches.
        merged = pd.concat([gf_df.reset_index(drop=True), bl_df.reset_index(drop=True)], axis=1)
        if (merged["gf_true_covid"] != merged["bl_true_covid"]).any():
            print("WARN: per-row true labels differ — order mismatch.")
    else:
        print(f"WARN: baseline N={len(bl_df)}, geneformer N={len(gf_df)} — using outer join may be unsafe")
        merged = pd.concat([gf_df, bl_df], axis=1)

    merged.to_csv(OUT_DIR / "per_cell_predictions.csv", index=False)
    print(f"  wrote {OUT_DIR/'per_cell_predictions.csv'}  ({len(merged)} rows)")

    # Agreement summary
    if "bl_proba_covid" in merged.columns and "gf_proba_covid" in merged.columns:
        agree = (merged["bl_pred_covid"] == merged["gf_pred_covid"]).mean()
        both_correct = ((merged["bl_pred_covid"] == merged["bl_true_covid"]) &
                         (merged["gf_pred_covid"] == merged["gf_true_covid"])).mean()
        only_bl_correct = ((merged["bl_pred_covid"] == merged["bl_true_covid"]) &
                            (merged["gf_pred_covid"] != merged["gf_true_covid"])).sum()
        only_gf_correct = ((merged["bl_pred_covid"] != merged["bl_true_covid"]) &
                            (merged["gf_pred_covid"] == merged["gf_true_covid"])).sum()
        both_wrong = ((merged["bl_pred_covid"] != merged["bl_true_covid"]) &
                       (merged["gf_pred_covid"] != merged["gf_true_covid"])).sum()
        print(f"\n  per-cell agreement: {agree*100:.1f}%")
        print(f"    both correct:   {both_correct*100:.1f}%")
        print(f"    only baseline:  {only_bl_correct} cells")
        print(f"    only geneformer: {only_gf_correct} cells")
        print(f"    both wrong:     {both_wrong} cells")

    # Plots
    print("\n=== ROC curves ===", flush=True)
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
    fpr_b, tpr_b, _ = roc_curve(bl_true, bl_proba_covid)
    fpr_g, tpr_g, _ = roc_curve(gf_true, gf_proba_covid)
    ax.plot(fpr_b, tpr_b, label=f"Logistic regression (AUC={bl_metrics['test_auc_covid']:.4f})", lw=2)
    ax.plot(fpr_g, tpr_g, label=f"Geneformer fine-tuned   (AUC={gf_auc:.4f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("COVID-19 vs Healthy in classical monocytes — held-out test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curves.png", dpi=120)
    plt.close(fig)
    print(f"  wrote {OUT_DIR/'roc_curves.png'}")

    # Probability scatter
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
    color = ["#1f77b4" if t == 1 else "#ff7f0e" for t in merged["gf_true_covid"]]
    ax.scatter(merged["bl_proba_covid"], merged["gf_proba_covid"], c=color, alpha=0.6, s=18)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
    ax.axhline(0.5, color="grey", lw=0.5)
    ax.axvline(0.5, color="grey", lw=0.5)
    ax.set_xlabel("baseline P(COVID)")
    ax.set_ylabel("Geneformer P(COVID)")
    ax.set_title("Per-cell COVID probabilities  (blue=COVID, orange=Healthy)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "agreement.png", dpi=120)
    plt.close(fig)
    print(f"  wrote {OUT_DIR/'agreement.png'}")

    out = {
        "baseline": {
            "test_acc": bl_metrics["test_acc"],
            "test_macro_f1": bl_metrics["test_macro_f1"],
            "test_auc_covid": bl_metrics["test_auc_covid"],
            "confusion_matrix": bl_metrics["confusion_matrix"],
            "n_features": bl_metrics["n_features"],
        },
        "geneformer": {
            "test_acc": float(gf_acc),
            "test_macro_f1": float(gf_macro_f1),
            "test_auc_covid": float(gf_auc),
            "confusion_matrix": gf_cm.tolist(),
        },
        "id_class_dict": {int(k): v for k, v in id_class_dict.items()},
        "n_test": int(len(gf_df)),
    }
    (OUT_DIR / "comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {OUT_DIR/'comparison.json'}")
    print("\n=== summary ===")
    print(f"  baseline  : acc={bl_metrics['test_acc']:.4f}  macro_f1={bl_metrics['test_macro_f1']:.4f}  AUC={bl_metrics['test_auc_covid']:.4f}")
    print(f"  geneformer: acc={gf_acc:.4f}  macro_f1={gf_macro_f1:.4f}  AUC={gf_auc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
