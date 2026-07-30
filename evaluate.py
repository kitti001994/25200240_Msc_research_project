import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
import pandas as pd

import config


PALETTE = {
    "train": "#2563eb", # blue
    "val": "#dc2626", # red
    "autistic": "#ea580c", # orange
    "nonautistic": "#16a34a", # green
    "grid": "#e5e7eb",
    "bg": "#ffffff",
    "text": "#111827",
    "light_text": "#6b7280",
}

# Axis labels for the 5 task conditions
CONDITION_LABELS = {
    "spontaneous_laughter": "Spontaneous\nlaughter",
    "conversational_laughter": "Conversational\nlaughter",
    "non_emotional_sound": "Non-emotional\nsound",
    "rest": "Rest",
    "beep": "Beep",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor": PALETTE["bg"],
    "axes.edgecolor": "#d1d5db",
    "axes.labelcolor": PALETTE["text"],
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": PALETTE["text"],
    "ytick.color": PALETTE["text"],
    "grid.color": PALETTE["grid"],
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.fontsize": 8,
    "legend.framealpha": 0.85,
    "font.family": "DejaVu Sans",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": PALETTE["bg"],
})


def _save(fig, fig_dir, name):
    for ext in config.FIGURE_FORMATS:
        path = os.path.join(fig_dir, f"{name}.{ext}")
        fig.savefig(path)
    plt.close(fig)


def _epoch_axis(ax, history):
    """x-axis to epoch numbers."""
    epochs = history["epoch"]
    ax.set_xlabel("Epoch")
    ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
    ax.grid(True, axis="both")


def _annotate_best(ax, epochs, values, label, color):
    """Best (maximum) value with a dot and annotation."""
    best_idx = int(np.argmax(values))
    best_ep = epochs[best_idx]
    best_val = values[best_idx]
    ax.scatter(best_ep, best_val, color=color, zorder=5, s=50)
    ax.annotate(
        f" {label} {best_val:.3f} (epoch {best_ep})",
        xy=(best_ep, best_val),
        fontsize=8, color=PALETTE["text"],
        xytext=(4, 2), textcoords="offset points",
    )


def _annotate_best_loss(ax, epochs, values, label, color):
    best_idx = int(np.argmin(values))
    best_ep = epochs[best_idx]
    best_val = values[best_idx]
    ax.scatter(best_ep, best_val, color=color, zorder=5, s=50)
    ax.annotate(
        f" {label} {best_val:.3f} (epoch {best_ep})",
        xy=(best_ep, best_val),
        fontsize=8, color=PALETTE["text"],
        xytext=(4, 2), textcoords="offset points",
    )


# Individual curve plots

def plot_loss(history, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = history["epoch"]
    ax.plot(epochs, history["train_loss"], color=PALETTE["train"], lw=1.8, label="Training")
    ax.plot(epochs, history["val_loss"], color=PALETTE["val"], lw=1.8, label="Validation")
    _annotate_best_loss(ax, epochs, history["val_loss"], "min", PALETTE["val"])
    ax.set_ylabel("Binary cross-entropy loss")
    ax.set_title("Loss")
    ax.legend()
    _epoch_axis(ax, history)
    _save(fig, fig_dir, "01_loss_curve")


def plot_accuracy(history, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = history["epoch"]
    ax.plot(epochs, history["train_accuracy"], color=PALETTE["train"], lw=1.8, label="Train accuracy")
    ax.plot(epochs, history["val_accuracy"], color=PALETTE["val"], lw=1.8, label="Val accuracy", linestyle="--")
    _annotate_best(ax, epochs, history["train_accuracy"], "max", PALETTE["train"])
    _annotate_best(ax, epochs, history["val_accuracy"], "max", PALETTE["val"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    _epoch_axis(ax, history)
    _save(fig, fig_dir, "02_accuracy_curve")


def plot_auc(history, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = history["epoch"]
    ax.plot(epochs, history["train_auc"], color=PALETTE["train"], lw=1.8, label="Training")
    ax.plot(epochs, history["val_auc"], color=PALETTE["val"], lw=1.8, label="Validation")
    _annotate_best(ax, epochs, history["val_auc"], "best", PALETTE["val"])
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC")
    ax.legend()
    _epoch_axis(ax, history)
    _save(fig, fig_dir, "03_auc_curve")


def plot_f1(history, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = history["epoch"]
    ax.plot(epochs, history["train_f1_macro"], color=PALETTE["train"], lw=1.8, label="Train F1 macro")
    ax.plot(epochs, history["val_f1_macro"], color=PALETTE["val"], lw=1.8, label="Val F1 macro", linestyle="--")
    ax.plot(epochs, history["train_f1_autistic"], color=PALETTE["autistic"], lw=1.2, label="Train F1 autistic", linestyle="-", alpha=0.7)
    ax.plot(epochs, history["val_f1_autistic"], color=PALETTE["autistic"], lw=1.2, label="Val F1 autistic", linestyle="--", alpha=0.7)
    _annotate_best(ax, epochs, history["val_f1_macro"], "max val", PALETTE["val"])
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("F1 Score (Macro and autistic class)")
    ax.legend(ncol=2)
    _epoch_axis(ax, history)
    _save(fig, fig_dir, "04_f1_curve")


def plot_sensitivity_specificity(history, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = history["epoch"]
    ax.plot(epochs, history["val_sensitivity"], color=PALETTE["autistic"], lw=1.8, label="Val Sensitivity")
    ax.plot(epochs, history["val_specificity"], color=PALETTE["nonautistic"], lw=1.8, label="Val Specificity", linestyle="--")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation Sensitivity and Specificity")
    ax.legend()
    _epoch_axis(ax, history)
    _save(fig, fig_dir, "05_sensitivity_specificity")


def plot_dice(history, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = history["epoch"]
    ax.plot(epochs, history["train_dice_autistic"], color=PALETTE["train"], lw=1.8, label="Train Dice (autistic)")
    ax.plot(epochs, history["val_dice_autistic"], color=PALETTE["val"], lw=1.8, label="Val Dice (autistic)", linestyle="--")
    _annotate_best(ax, epochs, history["val_dice_autistic"], "max", PALETTE["val"])
    ax.set_ylabel("Dice Coefficient")
    ax.set_ylim(0, 1.05)
    ax.set_title("Dice Coefficient - Autistic Class")
    ax.legend()
    _epoch_axis(ax, history)
    _save(fig, fig_dir, "06_dice_curve")


def plot_lr_schedule(history, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 3))
    epochs = history["epoch"]
    ax.plot(epochs, history["lr"], color="#7c3aed", lw=1.8)
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate")
    ax.set_yscale("log")
    _epoch_axis(ax, history)
    _save(fig, fig_dir, "13_lr")


def plot_all_metrics_grid(history, fig_dir):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("ST-GCN + FiLM - Training Metrics Overview", fontsize=13,
                 fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
    epochs = history["epoch"]

    panels = [
        ("train_loss", "val_loss", "BCE Loss", "Loss", False, "01"),
        ("train_accuracy", "val_accuracy", "Accuracy", "Accuracy", True, "02"),
        ("train_auc", "val_auc", "AUC-ROC", "AUC-ROC", True, "03"),
        ("train_f1_macro", "val_f1_macro", "F1 Macro", "F1", True, "04"),
        ("train_sensitivity", "val_sensitivity", "Sensitivity (autistic)", "Sensitivity", True, "05"),
        ("train_dice_autistic", "val_dice_autistic", "Dice (autistic)", "Dice", True, "06"),
    ]

    for idx, (tr_key, vl_key, title, ylabel, higher_better, _) in enumerate(panels):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        tr_vals = history[tr_key]
        vl_vals = history[vl_key]
        ax.plot(epochs, tr_vals, color=PALETTE["train"], lw=1.6, label="Train")
        ax.plot(epochs, vl_vals, color=PALETTE["val"], lw=1.6, label="Val", linestyle="--")
        ax.set_title(title, fontweight="semibold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Epoch")
        ax.grid(True, axis="both")
        ax.legend(fontsize=7)
        if higher_better:
            ax.set_ylim(0, 1.05)

    fig.tight_layout()
    _save(fig, fig_dir, "07_all_metrics_grid")


# Test-set plots

def plot_confusion_matrix(results_full, fig_dir):
    cm = np.array(results_full["test_metrics"]["confusion_matrix"])
    labels = ["non-autistic (0)", "autistic (1)"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cmap = LinearSegmentedColormap.from_list("blues", ["#eff6ff", "#1d4ed8"])
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix - Test Set")

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > thresh else PALETTE["text"])

    # TN/FP/FN/TP
    labels_annot = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i + 0.35, labels_annot[i][j],
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else PALETTE["light_text"])

    fig.tight_layout()
    _save(fig, fig_dir, "08_confusion_matrix")


def plot_roc_curve(results_full, fig_dir):
    y_true = np.array(results_full["test_labels"])
    y_prob = np.array(results_full["test_probs"])

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=PALETTE["train"], lw=2,
            label=f"ROC curve (AUC = {roc_auc_val:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["train"])
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1, linestyle="--", label="Chance")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.05)
    ax.set_xlabel("False Positive (1 - Specificity)")
    ax.set_ylabel("True Positive (Sensitivity)")
    ax.set_title("ROC Curve — Test Set")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    _save(fig, fig_dir, "09_roc_curve")


def plot_pr_curve(results_full, fig_dir):
    y_true = np.array(results_full["test_labels"])
    y_prob = np.array(results_full["test_probs"])

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision, color=PALETTE["autistic"], lw=2,
            label=f"PR curve (AP = {ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.08, color=PALETTE["autistic"])
    ax.axhline(baseline, color="#9ca3af", lw=1, linestyle="--",
               label=f"Baseline (prevalence = {baseline:.2f})")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.05)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Test Set")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    _save(fig, fig_dir, "10_pr_curve")


def plot_probability_distribution(results_full, fig_dir):
    y_true = np.array(results_full["test_labels"])
    y_prob = np.array(results_full["test_probs"])

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 25)

    ax.hist(y_prob[y_true == 0], bins=bins, color=PALETTE["nonautistic"], alpha=0.65,
            label="non-autistic (true label=0)", density=True, edgecolor="white", linewidth=0.4)
    ax.hist(y_prob[y_true == 1], bins=bins, color=PALETTE["autistic"], alpha=0.65,
            label="autistic (true label=1)", density=True, edgecolor="white", linewidth=0.4)
    ax.axvline(0.5, color="#111827", lw=1.2, linestyle="--", label="Decision threshold (0.5)")
    ax.set_xlabel("Predicted Probability (autistic)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted Probability Distribution by True Class - Test Set")
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()
    _save(fig, fig_dir, "11_prob_distribution")


# Descriptive statistics

def plot_descriptive_stats(results_full, fig_dir):
    desc = results_full["descriptive_stats"]
    splits = ["train", "val", "test"]

    metrics_to_plot = [
        ("n_windows", "N Windows"),
        ("bold_mean", "BOLD Mean"),
        ("bold_std", "BOLD Std"),
        ("bold_median", "BOLD Median"),
        ("bold_iqr", "BOLD IQR"),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 4.5))
    fig.suptitle("BOLD Window Descriptive Statistics by Split", fontsize=12, fontweight="bold")

    colors = [PALETTE["train"], "#f59e0b", PALETTE["val"]]

    for ax, (key, label) in zip(axes, metrics_to_plot):
        values = [desc[s][key] for s in splits]
        bars = ax.bar(splits, values, color=colors, edgecolor="white", linewidth=0.5, width=0.55)
        ax.set_title(label, fontsize=9, fontweight="semibold")
        ax.set_xticks(range(len(splits)))
        ax.set_xticklabels(splits, fontsize=8)
        ax.grid(True, axis="y")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(bar.get_height()) * 0.02,
                    f"{val:.3f}" if abs(val) < 1000 else f"{int(val):,}",
                    ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    _save(fig, fig_dir, "12_descriptive_stats")


# Per-condition metrics

def plot_per_condition_metrics(results_full, fig_dir):
    """
    Bar chart with AUC, accuracy, sensitivity, and specificity
    for each of the 5 task conditions on the test set.
    """
    cond_data = results_full.get("per_condition_metrics", {})
    if not cond_data:
        return

    # Filter to conditions that have full metric dicts
    valid = {k: v for k, v in cond_data.items() if "auc" in v}
    if not valid:
        return

    cond_names = list(valid.keys())
    short_names = [
        CONDITION_LABELS.get(n, n.replace("_", "\n")) for n in cond_names
    ]
    metrics_to_plot = [
        ("auc", "AUC", PALETTE["train"]),
        ("accuracy", "Accuracy", PALETTE["val"]),
        ("sensitivity", "Sensitivity", PALETTE["autistic"]),
        ("specificity", "Specificity", PALETTE["nonautistic"]),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(16, 5))
    fig.suptitle(
        "K-fold metrics per experimental condition",
        fontsize=12, fontweight="bold",
    )

    for ax, (metric_key, metric_label, color) in zip(axes, metrics_to_plot):
        values = [valid[cn].get(metric_key, float("nan")) for cn in cond_names]
        bars = ax.bar(
            range(len(cond_names)), values,
            color=color, alpha=0.8, edgecolor="white", linewidth=0.5, width=0.6,
        )
        ax.set_xticks(range(len(cond_names)))
        ax.set_xticklabels(short_names, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="#9ca3af", lw=0.8, linestyle=":", label="Chance")
        ax.set_title(metric_label, fontsize=9, fontweight="semibold")
        ax.set_ylabel(f"Window-level test {metric_label}")
        ax.grid(True, axis="y")
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )

    fig.tight_layout()
    _save(fig, fig_dir, "14_per_condition_metrics")


# Bootstrap confidence for test metrics
def bootstrap_test_metrics(results_full, n_boot=2000, seed=42):
    """
    Resample test predictions to estimate SD and
    95% CI for AUC, accuracy, sensitivity, and specificity.
    """
    probs = np.array(results_full.get("test_probs", []))
    labels = np.array(results_full.get("test_labels", []))
    conds = np.array(results_full.get("test_conds", []))
    if len(probs) == 0:
        return {}

    rng = np.random.RandomState(seed)

    def _boot(y_true, y_prob):
        n = len(y_true)
        aucs, accs, sens_list, spec_list = [], [], [], []
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            yt, yp = y_true[idx], y_prob[idx]
            ypr = (yp >= 0.5).astype(int)
            if len(np.unique(yt)) < 2:
                continue
            aucs.append(roc_auc_score(yt, yp))
            accs.append(np.mean(ypr == yt))
            tp = np.sum((ypr == 1) & (yt == 1))
            tn = np.sum((ypr == 0) & (yt == 0))
            fn = np.sum((ypr == 0) & (yt == 1))
            fp = np.sum((ypr == 1) & (yt == 0))
            sens_list.append(tp / max(tp + fn, 1))
            spec_list.append(tn / max(tn + fp, 1))
        out = {}
        for name, vals in [("auc", aucs), ("acc", accs),
                           ("sens", sens_list), ("spec", spec_list)]:
            if vals:
                ci = np.percentile(vals, [2.5, 97.5])
                out[name] = {
                    "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals)),
                    "ci_low": float(ci[0]),
                    "ci_high": float(ci[1]),
                }
        return out

    result = {"overall": _boot(labels, probs)}

    # Per-condition bootstrap
    cond_map = {
        0: "spontaneous_laughter", 1: "conversational_laughter",
        2: "non_emotional_sound", 3: "rest", 4: "beep",
    }
    for ci, cname in cond_map.items():
        mask = conds == ci
        if mask.sum() >= 20: # need enough samples
            result[cname] = _boot(labels[mask], probs[mask])

    return result

def save_summary_table(results_full, fig_dir):
    m = results_full["test_metrics"]
    f1_autistic = m.get("f1_autistic", m.get("f1_asd", float("nan")))
    f1_non_autistic = m.get("f1_nonautistic", m.get("f1_nt", float("nan")))
    precision_autistic = m.get("precision_autistic", m.get("precision_asd", float("nan")))
    precision_non_autistic = m.get("precision_nonautistic", m.get("precision_nt", float("nan")))
    dice_autistic = m.get("dice_autistic", m.get("dice_asd", float("nan")))
    dice_non_autistic = m.get("dice_nonautistic", m.get("dice_nt", float("nan")))
    row = {
        "test_accuracy": m["accuracy"],
        "test_auc": m["auc"],
        "test_f1_macro": m["f1_macro"],
        "test_f1_autistic": f1_autistic,
        "test_f1_non_autistic": f1_non_autistic,
        "test_precision_autistic": precision_autistic,
        "test_precision_non_autistic": precision_non_autistic,
        "test_sensitivity": m["sensitivity"],
        "test_specificity": m["specificity"],
        "test_dice_autistic": dice_autistic,
        "test_dice_non_autistic": dice_non_autistic,
        "test_TP": m["tp"],
        "test_TN": m["tn"],
        "test_FP": m["fp"],
        "test_FN": m["fn"],
        "best_val_auc": results_full["best_val_auc"],
        "best_fold": results_full.get("best_fold", results_full.get("best_epoch", "N/A")),
        "total_params": results_full["total_params"],
    }
    # Subject-level metrics (aggregated per-subject mean probability)
    sm = results_full.get("subject_level_metrics", {})
    if sm:
        row["subj_accuracy"] = sm.get("accuracy", float("nan"))
        row["subj_auc"] = sm.get("auc", float("nan"))
        row["subj_sensitivity"] = sm.get("sensitivity", float("nan"))
        row["subj_specificity"] = sm.get("specificity", float("nan"))
        row["subj_f1_macro"] = sm.get("f1_macro", float("nan"))

    # Bootstrap SD and 95% CI
    boot = bootstrap_test_metrics(results_full)
    if "overall" in boot:
        for metric in ["auc", "acc", "sens", "spec"]:
            b = boot["overall"].get(metric, {})
            row[f"test_{metric}_sd"] = b.get("sd", float("nan"))
            row[f"test_{metric}_ci_low"] = b.get("ci_low", float("nan"))
            row[f"test_{metric}_ci_high"] = b.get("ci_high", float("nan"))
    # Per-condition bootstrap SDs
    for cname, cboot in boot.items():
        if cname == "overall":
            continue
        for metric in ["auc", "acc", "sens", "spec"]:
            b = cboot.get(metric, {})
            row[f"{cname}_{metric}_sd"] = b.get("sd", float("nan"))

    df = pd.DataFrame([row])
    csv_path = os.path.join(fig_dir, "summary_table.csv")
    df.to_csv(csv_path, index=False)

    print("\n Test Set Summary")
    for k, v in row.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Bootstrap summary
    if boot:
        print("\n Bootstrap 95% CI")
        for scope, bdict in boot.items():
            label = scope.replace("_", " ").title()
            parts = []
            for metric in ["auc", "acc", "sens", "spec"]:
                b = bdict.get(metric, {})
                if b:
                    parts.append(
                        f"{metric.upper()}={b['mean']:.4f}+/-{b['sd']:.4f} "
                        f"[{b['ci_low']:.4f},{b['ci_high']:.4f}]"
                    )
            print(f"  {label}: {' | '.join(parts)}")


# LOSO evaluation
def plot_loso_roc(loso_results, fig_dir):
    """
    Subject-level ROC curve from LOSO results.
    Each data point is one subject (N=46 total).
    """
    from sklearn.metrics import roc_curve, auc as sk_auc

    labels = [loso_results["subject_labels"][s] for s in loso_results["subject_probs"]]
    probs = [loso_results["subject_probs"][s] for s in loso_results["subject_probs"]]

    if len(set(labels)) < 2:
        return

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = sk_auc(fpr, tpr)
    n_subj = len(labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PALETTE["val"], lw=2.0,
            label=f"LOSO ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=0.8, linestyle=":")
    ax.set_xlabel("False Positive (1 - Specificity)")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"LOSO Subject-Level ROC (N = {n_subj})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend()
    ax.grid(True)
    _save(fig, fig_dir, "loso_01_subject_roc")


def plot_loso_probs(loso_results, fig_dir):
    """
    Per-subject calibrated probability bar chart from LOSO.
    Orange = autistic, green = non-autistic, purple x = misclassified subjects.
    Threshold markers show the fold-specific inner-validation threshold.
    """
    sids = list(loso_results["subject_probs"].keys())
    probs = np.array([loso_results["subject_probs"][s] for s in sids])
    labels = np.array([loso_results["subject_labels"][s] for s in sids])
    thresholds_map = loso_results.get("subject_thresholds", {})
    preds_map = loso_results.get("subject_preds", {})
    thresholds = np.array([thresholds_map.get(s, 0.5) for s in sids])
    preds = np.array([preds_map.get(s, int(loso_results["subject_probs"][s] >= thresholds_map.get(s, 0.5))) for s in sids])

    order = np.argsort(probs) # sort by probability
    sids = [sids[i] for i in order]
    probs = probs[order]
    labels = labels[order]
    thresholds = thresholds[order]
    preds = preds[order]

    colors = [PALETTE["autistic"] if l == 1 else PALETTE["nonautistic"] for l in labels]
    x = np.arange(len(sids))

    fig, ax = plt.subplots(figsize=(max(10, len(sids) * 0.28), 5))
    ax.bar(x, probs, color=colors, alpha=0.85, width=0.8)
    ax.scatter(
        x, thresholds, marker="_", color="#111827",
        s=55, linewidths=1.6, label="Fold threshold",
    )

    for i, (p, l, pred) in enumerate(zip(probs, labels, preds)):
        if int(pred) != int(l):
            ax.scatter(i, p + 0.02, marker="x", color="#7c3aed",
                       s=70, zorder=5, linewidths=2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(sids, rotation=90, fontsize=6)
    ax.set_ylabel("Calibrated P(autistic)")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "LOSO Per-Subject Autistic Probability  "
        "(orange=autistic, green=non-autistic, x=misclassified)"
    )
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()
    _save(fig, fig_dir, "loso_02_subject_probs")


def plot_loso_calibration(loso_results, fig_dir):
    """
    Calibration curve for LOSO subject predictions.
    """
    probs = np.array(list(loso_results["subject_probs"].values()))
    labels = np.array(list(loso_results["subject_labels"].values()), dtype=int)

    n_bins = min(8, len(probs) // 3)
    if n_bins < 2:
        return

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred, frac_pos, bin_sizes = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        n = mask.sum()
        if n == 0:
            continue
        mean_pred.append(float(probs[mask].mean()))
        frac_pos.append(float(labels[mask].mean()))
        bin_sizes.append(n)

    if len(mean_pred) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(mean_pred, frac_pos, "o-", color=PALETTE["val"], lw=2.0, label="Model")
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=0.8, linestyle=":", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction autistic")
    ax.set_title("LOSO Calibration Curve")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True)

    ax2 = axes[1]
    ax2.bar(range(len(bin_sizes)), bin_sizes, color=PALETTE["train"], alpha=0.7)
    ax2.set_xlabel("Bin index")
    ax2.set_ylabel("Subjects per bin")
    ax2.set_title("Bin counts")
    ax2.grid(True, axis="y")

    fig.tight_layout()
    _save(fig, fig_dir, "loso_03_calibration")


def plot_loso_confusion_matrix(loso_results, fig_dir):
    """
    Subject-level confusion matrix for LOSO.
    """
    from sklearn.metrics import confusion_matrix

    sids = list(loso_results.get("subject_probs", {}).keys())
    if not sids:
        return

    labels = np.array([loso_results["subject_labels"][s] for s in sids], dtype=int)
    thresholds = loso_results.get("subject_thresholds", {})
    preds_map = loso_results.get("subject_preds", {})
    preds = np.array([
        preds_map.get(s, int(loso_results["subject_probs"][s] >= thresholds.get(s, 0.5)))
        for s in sids
    ], dtype=int)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    fig, ax = plt.subplots(figsize=(5.5, 4.7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_labels = ["non-autistic", "autistic"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(tick_labels, rotation=15, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        "LOSO Subject-Level Confusion Matrix\n"
        f"Sensitivity={sensitivity:.3f}, Specificity={specificity:.3f}"
    )

    threshold = cm.max() / 2.0 if cm.max() else 0
    labels_annot = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold else PALETTE["text"]
            ax.text(j, i - 0.08, str(cm[i, j]), ha="center", va="center",
                    fontsize=15, fontweight="bold", color=color)
            ax.text(j, i + 0.25, labels_annot[i][j], ha="center", va="center",
                    fontsize=8, color=color)

    fig.tight_layout()
    _save(fig, fig_dir, "loso_04_confusion_matrix")


def plot_loso_learning_curve(fold_histories, fig_dir):
    """
    LOSO learning curve
    """
    if not fold_histories:
        return

    # Folds that carry at least two epochs of history
    valid_folds = [
        h for h in fold_histories.values() if len(h["epochs"]) >= 2
    ]
    if not valid_folds:
        return

    n_folds = len(fold_histories)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # training loss, one blue line per fold
    for h in valid_folds:
        ax1.plot(
            h["epochs"], h["train_loss"],
            color=PALETTE["train"], lw=1.0, alpha=0.35,
        )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax1.set_title(f"Per-fold training loss ({n_folds} LOSO folds)")
    ax1.grid(True, axis="both")

    # inner validation AUC, one green line per fold
    for h in valid_folds:
        ax2.plot(
            h["epochs"], h["val_auc"],
            color=PALETTE["nonautistic"], lw=1.0, alpha=0.35,
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation AUC-ROC")
    ax2.set_title(f"Per-fold validation AUC ({n_folds} LOSO folds)")
    ax2.grid(True, axis="both")

    fig.tight_layout()
    _save(fig, fig_dir, "loso_05_learning_curve")


def plot_loso_metric_histogram(loso_results, fig_dir):
    """
    LOSO metrics histograms.
    fold-level best validation AUC distribution (one bar per fold).
    subject-level P(autistic) distribution split by true class.
    """
    fold_val_aucs = loso_results.get("fold_val_aucs", {})
    subject_probs = loso_results.get("subject_probs", {})
    subject_labels = loso_results.get("subject_labels", {})

    has_fold_aucs = len(fold_val_aucs) > 0
    has_probs = len(subject_probs) > 0

    if not has_fold_aucs and not has_probs:
        return

    n_panels = int(has_fold_aucs) + int(has_probs)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # fold validation AUC
    if has_fold_aucs:
        aucs = np.array(list(fold_val_aucs.values()))
        ax = axes[panel_idx]
        ax.hist(
            aucs, bins="sqrt", facecolor="chartreuse", edgecolor="grey",
            alpha=0.85,
        )
        ax.axvline(
            np.mean(aucs), color="red", lw=1.5, linestyle="--",
            label=f"Mean = {np.mean(aucs):.4f}",
        )
        ax.axvline(
            np.median(aucs), color="#7c3aed", lw=1.5, linestyle=":",
            label=f"Median = {np.median(aucs):.4f}",
        )
        ax.set_xlabel("Best Validation AUC")
        ax.set_ylabel("Number of Folds")
        ax.set_title(
            f"Fold Validation AUC Distribution (N = {len(aucs)})"
        )
        ax.legend()
        ax.grid(True, axis="y")
        panel_idx += 1

    # subject-level P(autistic) histogram by class
    if has_probs:
        sids = list(subject_probs.keys())
        probs = np.array([subject_probs[s] for s in sids])
        labels_arr = np.array([subject_labels[s] for s in sids])

        ax = axes[panel_idx]
        bins = np.linspace(0, 1, 15)
        ax.hist(
            probs[labels_arr == 0], bins=bins,
            facecolor=PALETTE["nonautistic"], alpha=0.65, edgecolor="white",
            linewidth=0.4, label="Non-autistic (true label = 0)",
        )
        ax.hist(
            probs[labels_arr == 1], bins=bins,
            facecolor=PALETTE["autistic"], alpha=0.65, edgecolor="white",
            linewidth=0.4, label="autistic (true label = 1)",
        )
        ax.axvline(
            0.5, color="#111827", lw=1.2, linestyle="--",
            label="Decision threshold (0.5)",
        )
        ax.set_xlabel("Subject-Level P(autistic)")
        ax.set_ylabel("Number of Subjects")
        ax.set_title(
            f"LOSO Subject Probability Distribution (N = {len(probs)})"
        )
        ax.legend()
        ax.grid(True, axis="y")

    fig.tight_layout()
    _save(fig, fig_dir, "loso_06_metric_histogram")


def plot_all_loso(loso_results, fig_dir, fold_histories=None):
    """
    LOSO evaluation figures and summary CSV.
    """
    os.makedirs(fig_dir, exist_ok=True)
    print("\nGenerating LOSO figures.")
    n_figs = 0

    labels = list(loso_results.get("subject_labels", {}).values())
    if len(set(labels)) >= 2:
        plot_loso_roc(loso_results, fig_dir); n_figs += 1

    if loso_results.get("subject_probs"):
        plot_loso_probs(loso_results, fig_dir); n_figs += 1
        plot_loso_calibration(loso_results, fig_dir); n_figs += 1
        plot_loso_confusion_matrix(loso_results, fig_dir); n_figs += 1

    # Learning curve from per-fold epoch
    if fold_histories:
        plot_loso_learning_curve(fold_histories, fig_dir); n_figs += 1

    # Metric histogram (fold val AUCs + subject-level probabilities)
    if loso_results.get("fold_val_aucs") or loso_results.get("subject_probs"):
        plot_loso_metric_histogram(loso_results, fig_dir); n_figs += 1

    # LOSO summary CSV
    sm = loso_results.get("subject_metrics", {})
    if sm:
        summary = {
            "loso_n_completed": loso_results.get("n_completed"),
            "loso_n_total": loso_results.get("n_total"),
            "subj_accuracy": sm.get("accuracy"),
            "subj_auc": sm.get("auc"),
            "subj_f1_macro": sm.get("f1_macro"),
            "subj_sensitivity": sm.get("sensitivity"),
            "subj_specificity": sm.get("specificity"),
            "elapsed_time": loso_results.get("elapsed_time"),
        }
        csv_path = os.path.join(fig_dir, "loso_summary_table.csv")
        pd.DataFrame([summary]).to_csv(csv_path, index=False)
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


def plot_all(history, results_full, fig_dir):
    """Generate and save all figures. Called from train.py."""
    os.makedirs(fig_dir, exist_ok=True)
    print("\nGenerating figures.")

    n_figs = 0

    # Epoch curves
    if history and history.get("epoch"):
        plot_loss(history, fig_dir); n_figs += 1
        plot_accuracy(history, fig_dir); n_figs += 1
        plot_auc(history, fig_dir); n_figs += 1
        plot_f1(history, fig_dir); n_figs += 1
        plot_sensitivity_specificity(history, fig_dir); n_figs += 1
        plot_dice(history, fig_dir); n_figs += 1
        plot_lr_schedule(history, fig_dir); n_figs += 1
        plot_all_metrics_grid(history, fig_dir); n_figs += 1

    # Test-set figures
    plot_confusion_matrix(results_full, fig_dir); n_figs += 1
    plot_roc_curve(results_full, fig_dir); n_figs += 1
    plot_pr_curve(results_full, fig_dir); n_figs += 1
    plot_probability_distribution(results_full, fig_dir); n_figs += 1

    # Descriptive stats
    plot_descriptive_stats(results_full, fig_dir); n_figs += 1

    # Per-condition metrics bar chart
    if results_full.get("per_condition_metrics"):
        plot_per_condition_metrics(results_full, fig_dir); n_figs += 1

    # CSV summary
    save_summary_table(results_full, fig_dir)