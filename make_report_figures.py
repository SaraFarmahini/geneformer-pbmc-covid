"""Generate plots for the final report.

Outputs (under data/geneformer/tier2/run_mono_covid/comparison/):
  training_curve.png         loss + eval macro-F1 vs epoch
  confusion_baseline.png     LR confusion-matrix heatmap
  confusion_geneformer.png   Geneformer confusion-matrix heatmap
  per_sample_errors.png      stacked bar of correct/error per sample
  metrics_bar.png            grouped bar of acc/F1/AUC for both models
  report_data.json           machine-readable training-curve data for the canvas
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/sarafarmahinifarahani/Downloads/single_Cell/Project1")
RUN = ROOT / "data/geneformer/tier2/run_mono_covid"
COMP = RUN / "comparison"
COMP.mkdir(parents=True, exist_ok=True)

# matplotlib hygiene -- flat, no spurious styling
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titleweight": "bold",
    "figure.dpi": 130,
})

# -----------------------------------------------------------------------------
# 1. Training curve from trainer_state.json
# -----------------------------------------------------------------------------
ts_path = (
    RUN / "ft" / "260428_geneformer_cellClassifier_mono" / "ksplit1"
    / "checkpoint-1992" / "trainer_state.json"
)
ts = json.loads(ts_path.read_text())

train_steps, train_loss, train_lr = [], [], []
eval_steps, eval_loss, eval_acc, eval_f1 = [], [], [], []
for entry in ts["log_history"]:
    if "loss" in entry and "eval_loss" not in entry:
        train_steps.append(entry["step"])
        train_loss.append(entry["loss"])
        train_lr.append(entry.get("learning_rate", float("nan")))
    if "eval_loss" in entry:
        eval_steps.append(entry["step"])
        eval_loss.append(entry["eval_loss"])
        eval_acc.append(entry["eval_accuracy"])
        eval_f1.append(entry["eval_macro_f1"])

# epoch labels: 1992 steps total, 4 epochs => 498 steps/epoch
steps_per_epoch = 498
train_epoch = [s / steps_per_epoch for s in train_steps]
eval_epoch = [s / steps_per_epoch for s in eval_steps]

fig, ax1 = plt.subplots(1, 1, figsize=(7.0, 4.2))
ax1.plot(train_epoch, train_loss, color="#888", lw=1, alpha=0.7, label="train loss (per 50 steps)")
ax1.plot(eval_epoch, eval_loss, marker="o", color="#1f77b4", lw=2, label="val loss (per epoch)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-entropy loss")
ax1.set_ylim(0, max(train_loss + eval_loss) * 1.05)
ax1.set_xticks([0, 1, 2, 3, 4])

ax2 = ax1.twinx()
ax2.spines["top"].set_visible(False)
ax2.plot(eval_epoch, eval_f1, marker="s", color="#2ca02c", lw=2, label="val macro-F1")
ax2.plot(eval_epoch, eval_acc, marker="^", color="#9467bd", lw=2, label="val accuracy")
ax2.set_ylabel("Validation metric")
ax2.set_ylim(0.85, 1.0)

# combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, frameon=False)

ax1.set_title("Geneformer fine-tune training curve")
fig.tight_layout()
fig.savefig(COMP / "training_curve.png", dpi=130)
plt.close(fig)
print(f"wrote {COMP/'training_curve.png'}")

# -----------------------------------------------------------------------------
# 2. Confusion-matrix heatmaps
# -----------------------------------------------------------------------------
def plot_cm(cm, title, out_path, accent):
    """cm is 2x2: [[TN,FP],[FN,TP]] with rows=true (Healthy, COVID), cols=pred."""
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.6))
    cmap = plt.get_cmap("Blues") if accent == "blue" else plt.get_cmap("Greens")
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Healthy", "COVID-19"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Healthy", "COVID-19"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            color = "white" if v > cm.max() * 0.55 else "#222"
            ax.text(j, i, str(v), ha="center", va="center", fontsize=14, color=color, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


bl_cm = np.array([[32, 3], [3, 211]])
gf_cm = np.array([[34, 1], [3, 210]])
plot_cm(bl_cm, "Logistic regression — confusion", COMP / "confusion_baseline.png", "blue")
plot_cm(gf_cm, "Geneformer fine-tuned — confusion", COMP / "confusion_geneformer.png", "green")
print(f"wrote {COMP/'confusion_baseline.png'} and {COMP/'confusion_geneformer.png'}")

# -----------------------------------------------------------------------------
# 3. Per-sample stacked bar of test errors (Geneformer)
# -----------------------------------------------------------------------------
samples = ["HIP043\n(Healthy)", "S556\n(COVID)", "S557\n(COVID)", "S558\n(COVID)"]
n_test  = np.array([35, 16, 113, 84])
n_err   = np.array([1, 0, 2, 1])
n_ok    = n_test - n_err

fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.6))
x = np.arange(len(samples))
ax.bar(x, n_ok,  color="#2ca02c", label="correct")
ax.bar(x, n_err, bottom=n_ok, color="#d62728", label="error")
for i, (ok, err) in enumerate(zip(n_ok, n_err)):
    ax.text(i, ok + err + 1.5, f"{ok}/{ok+err}", ha="center", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(samples)
ax.set_ylabel("Test cells")
ax.set_title("Geneformer test-set predictions per sample")
ax.legend(loc="upper left", frameon=False)
ax.set_ylim(0, max(n_test) * 1.18)
fig.tight_layout()
fig.savefig(COMP / "per_sample_errors.png", dpi=130)
plt.close(fig)
print(f"wrote {COMP/'per_sample_errors.png'}")

# -----------------------------------------------------------------------------
# 4. Grouped bar of metrics for both models
# -----------------------------------------------------------------------------
metrics = ["Accuracy", "Macro F1", "AUC (COVID)"]
bl_vals = [0.9759, 0.9501, 0.9949]
gf_vals = [0.9839, 0.9675, 0.9945]

x = np.arange(len(metrics))
w = 0.35
fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))
ax.bar(x - w/2, bl_vals, w, color="#888", label="Logistic regression")
ax.bar(x + w/2, gf_vals, w, color="#1f77b4", label="Geneformer fine-tuned")
for i, (b, g) in enumerate(zip(bl_vals, gf_vals)):
    ax.text(i - w/2, b + 0.002, f"{b:.4f}", ha="center", fontsize=9)
    ax.text(i + w/2, g + 0.002, f"{g:.4f}", ha="center", fontsize=9, weight="bold")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0.93, 1.005)
ax.set_ylabel("Score")
ax.set_title("Held-out test set — baseline vs Geneformer")
ax.legend(loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig(COMP / "metrics_bar.png", dpi=130)
plt.close(fig)
print(f"wrote {COMP/'metrics_bar.png'}")

# -----------------------------------------------------------------------------
# 5. Dump training-curve JSON for the canvas to embed inline
# -----------------------------------------------------------------------------
report_data = {
    "training_curve": {
        "train_epoch": train_epoch,
        "train_loss":  train_loss,
        "eval_epoch":  eval_epoch,
        "eval_loss":   eval_loss,
        "eval_acc":    eval_acc,
        "eval_f1":     eval_f1,
    },
}
(COMP / "report_data.json").write_text(json.dumps(report_data, indent=2))
print(f"wrote {COMP/'report_data.json'}")
print("\nDONE")
