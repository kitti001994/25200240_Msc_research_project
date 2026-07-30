"""
Training pipeline with 5-fold cross-validation,
per-condition evaluation, per-condition single-condition models.

5-fold cross-validation:
  Test set (15% - 6 subjects) is held out. K-fold on the remaining 85% of subjects.

Per-condition training (--all-conditions):
  Trained per condition using only windows from that condition.

Metrics:
  loss - BCEWithLogitsLoss
  accuracy - (TP+TN) / N
  auc - area under the ROC curve
  f1_macro - macro-averaged F1
  sensitivity - TP / (TP + FN) for autistic class
  specificity - TN / (TN + FP) for non-autistic class
  dice_asd - Dice coefficient for autistic class
"""

import os
import time
import shutil
import argparse
import random
import json
import warnings
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report,
    precision_score,
)
from sklearn.linear_model import LogisticRegression
from nilearn.maskers import NiftiLabelsMasker


warnings.filterwarnings(
    "ignore",
    message=".*labels were removed.*",
    category=UserWarning,
)

import config
from dataset import fMRIWindowDataset, collate_fn, get_run_timeseries
from graph import build_population_graph
from model import build_model, build_model_no_film
from demographics import (
    load_demographics,
    build_feature_map,
    compute_normalisation_stats,
    normalise_feature_map,
    get_feature_dim,
)

# training log
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.txt")


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 


def build_selected_model(
    edge_index,
    edge_weight,
    n_parcels,
    n_static_features=0,
    use_film=True,
):
    """Build the selected ST-GCN architecture."""
    if use_film:
        return build_model(
            edge_index, edge_weight, n_parcels=n_parcels,
            n_static_features=n_static_features,
        )
    return build_model_no_film(
        edge_index, edge_weight, n_parcels=n_parcels,
        n_static_features=n_static_features,
    )


def load_subject_list(bids_root):
    tsv_path = os.path.join(bids_root, "participants.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            f"participants.tsv not found at {tsv_path}.\n"
            f"Check if BIDS root contains participants.tsv with a '{config.GROUP_COLUMN}' column."
        )
    df = pd.read_csv(tsv_path, sep="\t")
    required = {"participant_id", config.GROUP_COLUMN}
    if not required.issubset(df.columns):
        raise ValueError(f"participants.tsv must contain {required}. Found: {set(df.columns)}")

    subject_list = []
    for _, row in df.iterrows():
        subj_id = str(row["participant_id"]).replace("sub-", "")
        group = row[config.GROUP_COLUMN]
        if group == config.ASD_LABEL:
            label = 1
        elif group == config.NT_LABEL:
            label = 0
        else:
            print(f"  Unknown group '{group}' for {subj_id} - skipped")
            continue
        subject_list.append((subj_id, label))

    n_asd = sum(l for _, l in subject_list)
    n_nt = len(subject_list) - n_asd
    print(f"Found {len(subject_list)} subjects ({n_asd} autistic / {n_nt} non-autistic)")
    return subject_list


def subsample_subjects(subject_list, n_subjects):
    """Balanced subsample of n_subjects."""
    asd = [s for s in subject_list if s[1] == 1]
    nt = [s for s in subject_list if s[1] == 0]
    n_each = n_subjects // 2
    n_asd_take = min(n_each, len(asd))
    n_nt_take = min(n_subjects - n_asd_take, len(nt))
    subset = asd[:n_asd_take] + nt[:n_nt_take]
    print(
        f"[Subsample] Using {len(subset)} subjects "
        f"({n_asd_take} autistic / {n_nt_take} non-autistic) "
        f"out of {len(subject_list)}"
    )
    return subset


# Train / val / test splits
def split_test_subjects(subject_list):
    ids = [s[0] for s in subject_list]
    labels = [s[1] for s in subject_list]
    n = len(subject_list)
    n_cls = len(set(labels))

    test_size = max(n_cls, int(n * config.TEST_SPLIT))
    test_size = min(test_size, n - n_cls * 2)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=config.SEED)
    trainval_idx, test_idx = next(sss.split(ids, labels))

    trainval_list = [subject_list[i] for i in trainval_idx]
    test_list = [subject_list[i] for i in test_idx]

    n_tv_asd = sum(l for _, l in trainval_list)
    n_test_asd = sum(l for _, l in test_list)
    print(
        f"Split 70/15/15 -> "
        f"train+val: {len(trainval_list)} subjects "
        f"({n_tv_asd} autistic / {len(trainval_list) - n_tv_asd} non-autistic) | "
        f"test: {len(test_list)} subjects "
        f"({n_test_asd} autistic / {len(test_list) - n_test_asd} non-autistic)"
    )
    return trainval_list, test_list


def kfold_subjects(trainval_list, n_folds):
    """
    Create K stratified train/val subject splits from trainval_list.
    Each fold uses roughly (K-1)/K subjects for training and 1/K for validation.
    """
    ids = [s[0] for s in trainval_list]
    labels = [s[1] for s in trainval_list]

    # avoid splits with fewer than 1 subject per class
    actual_folds = min(n_folds, len(trainval_list) // 2)
    if actual_folds < 2:
        actual_folds = 2
        print(f"Too few subjects for {n_folds}-fold CV - using {actual_folds} folds.")

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=config.SEED)
    splits = []
    for train_idx, val_idx in skf.split(ids, labels):
        splits.append(
            ([trainval_list[i] for i in train_idx],
             [trainval_list[i] for i in val_idx])
        )
    return splits


# Weighted sampler
def make_weighted_sampler(dataset):
    labels = [int(dataset.samples[i][2].item()) for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


# Metrics
def dice_coefficient(y_true, y_pred, class_idx):
    """Dice = 2*TP / (2*TP + FP + FN) for a single class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == class_idx) & (y_pred == class_idx))
    fp = np.sum((y_true != class_idx) & (y_pred == class_idx))
    fn = np.sum((y_true == class_idx) & (y_pred != class_idx))
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def compute_metrics(y_true, y_pred, y_prob):
    """Return a dict of all classification metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": auc,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_asd": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_nt": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_asd": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_nt": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "dice_asd": dice_coefficient(y_true, y_pred, class_idx=1),
        "dice_nt": dice_coefficient(y_true, y_pred, class_idx=0),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }


def per_condition_metrics(all_labels, all_preds, all_probs, all_conds):
    """
    Classification metrics separately for each condition.
    (0=spontaneous, 1=conversational, 2=non-emotional, 3=rest, 4=beep).
    """
    results = {}
    for cond_key, cond_name in config.CONDITION_MAP.items():
        cond_idx = cond_key - 1
        mask = [i for i, c in enumerate(all_conds) if c == cond_idx]
        if len(mask) < 4:
            results[cond_name] = {"n_samples": len(mask), "note": "too few samples"}
            continue
        y_t = [all_labels[i] for i in mask]
        y_p = [all_preds[i] for i in mask]
        y_prob = [all_probs[i] for i in mask]
        results[cond_name] = compute_metrics(y_t, y_p, y_prob)
    return results


def per_condition_subject_metrics(all_labels, all_probs, all_conds, subject_ids):
    """
    Subject-level metrics for each condition.

    For each condition c, all test windows with onset condition == c are
    grouped by subject. The mean predicted probability is:
      p_subj(c) = (1/M) * sum_{m=1}^{M} p_hat(window_m)
    where M is the number of windows for that subject in condition c.

    Subject prediction: autistic if p_subj(c) >= 0.5, non-autistic otherwise.
    """
    results = {}
    for cond_key, cond_name in config.CONDITION_MAP.items():
        cond_idx = cond_key - 1 # 0-indexed condition

        # Group windows by subject for this condition
        subj_probs = defaultdict(list)
        subj_labels = {}
        for prob, label, cond, sid in zip(all_probs, all_labels, all_conds, subject_ids):
            if int(cond) == cond_idx:
                subj_probs[sid].append(prob)
                subj_labels[sid] = int(label)

        n_subj = len(subj_probs)
        if n_subj < 2:
            results[cond_name] = {"n_subjects": n_subj, "note": "too few subjects"}
            continue

        ordered_sids = list(subj_probs.keys())
        y_true = [subj_labels[s] for s in ordered_sids]
        y_prob = [float(np.mean(subj_probs[s])) for s in ordered_sids]
        y_pred = [int(p >= 0.5) for p in y_prob]

        # AUC needs both classes
        if len(set(y_true)) < 2:
            results[cond_name] = {"n_subjects": n_subj, "note": "single class in test set"}
            continue

        m = compute_metrics(y_true, y_pred, y_prob)
        m["n_subjects"] = n_subj
        m["subject_details"] = [
            {
                "id": sid,
                "mean_prob": float(np.mean(subj_probs[sid])),
                "n_windows": len(subj_probs[sid]),
                "true_label": subj_labels[sid],
            }
            for sid in ordered_sids
        ]
        results[cond_name] = m

    return results


# Statistics
def _skewness(x):
    m = np.mean(x)
    s = np.std(x)
    return float(np.mean(((x - m) / (s + 1e-8)) ** 3))


def _kurtosis(x):
    m = np.mean(x)
    s = np.std(x)
    return float(np.mean(((x - m) / (s + 1e-8)) ** 4) - 3.0)


def dataset_descriptive_stats(dataset, split_name):
    """Statistics over all BOLD windows in a dataset split."""
    if len(dataset.samples) == 0:
        print(f"\n  Descriptive stats [{split_name}]: empty dataset, skipping.")
        return {}

    labels = [int(s[2].item()) for s in dataset.samples]
    n_total = len(labels)
    n_asd = sum(labels)
    n_nt = n_total - n_asd

    all_X = torch.stack([s[0] for s in dataset.samples]).numpy()
    flat = all_X.reshape(-1)

    stats = {
        "split": split_name,
        "n_windows": n_total,
        "n_asd_windows": n_asd,
        "n_nt_windows": n_nt,
        "class_balance": round(n_asd / n_total, 4) if n_total > 0 else 0.0,
        "bold_mean": float(np.mean(flat)),
        "bold_std": float(np.std(flat)),
        "bold_min": float(np.min(flat)),
        "bold_max": float(np.max(flat)),
        "bold_median": float(np.median(flat)),
        "bold_q25": float(np.percentile(flat, 25)),
        "bold_q75": float(np.percentile(flat, 75)),
        "bold_iqr": float(np.percentile(flat, 75) - np.percentile(flat, 25)),
        "bold_skewness": _skewness(flat),
        "bold_kurtosis": _kurtosis(flat),
        "n_parcels": int(all_X.shape[1]),
        "window_trs": int(all_X.shape[2]),
    }

    print(f"\n  Descriptive stats [{split_name}]")
    print(
        f"    Windows: {n_total} "
        f"(autistic={n_asd}, non-autistic={n_nt}, "
        f"balance={stats['class_balance']:.3f})"
    )
    print(
        f"    BOLD mean={stats['bold_mean']:.4f} std={stats['bold_std']:.4f} "
        f"range=[{stats['bold_min']:.3f}, {stats['bold_max']:.3f}]"
    )
    return stats


# Single epoch
def run_epoch(model, loader, optimizer, criterion, device, train=True):
    """
    Run one training or evaluation epoch.
    Returns avg_loss, metrics_dict, all_probs, all_labels, all_conds.
    """
    model.train(train)
    total_loss = 0.0
    all_probs, all_labels, all_conds = [], [], []
    accum_steps = config.GRAD_ACCUM_STEPS if train else 1
    use_mixup = train and config.MIXUP_ALPHA > 0

    with torch.set_grad_enabled(train):
        if train:
            optimizer.zero_grad()

        for step, (X, cond_reg, y_group, y_cond, static_feat) in enumerate(loader):
            X         = X.to(device)
            cond_reg  = cond_reg.to(device)
            y_group   = y_group.to(device)
            # static_feat is (B, F); F=0 when demographics disabled
            static_feat = static_feat.to(device) if static_feat.shape[-1] > 0 else None

            # Mixup: interpolate pairs of samples within the micro-batch
            # x_mix = lam * x_i + (1 - lam) * x_perm[i]
            # y_mix = lam * y_i + (1 - lam) * y_perm[i]
            if use_mixup:
                lam = np.random.beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA)
                perm = torch.randperm(X.size(0), device=device)
                X_mix    = lam * X + (1.0 - lam) * X[perm]
                cond_mix = lam * cond_reg + (1.0 - lam) * cond_reg[perm]
                y_mix    = lam * y_group + (1.0 - lam) * y_group[perm]
                # Static features are subject-level: mixup mixes feature vectors too
                sf_mix = (
                    lam * static_feat + (1.0 - lam) * static_feat[perm]
                    if static_feat is not None else None
                )
                logits = model(X_mix, cond_mix, sf_mix)
            else:
                logits = model(X, cond_reg, static_feat)

            # Determine loss target
            if use_mixup:
                y_target = y_mix
            else:
                y_target = y_group

            # Label smoothing: y_smooth = y*(1 - eps) + 0.5*eps
            # maps 0 to eps/2 and 1 to 1 - eps/2
            if train and config.LABEL_SMOOTHING > 0:
                eps = config.LABEL_SMOOTHING
                y_target = y_target * (1.0 - eps) + 0.5 * eps

            loss = criterion(logits, y_target)

            # Track unscaled loss for logging (before accumulation scaling)
            total_loss += loss.item() * y_group.size(0)
            # Track probabilities and original (un-mixed) labels for metrics
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            all_labels.extend(y_group.cpu().numpy().tolist())
            all_conds.extend(y_cond.cpu().numpy().tolist())

            if train:
                # Scale loss by 1/accum_steps so accumulated gradients
                # average correctly over the effective batch
                scaled_loss = loss / accum_steps
                scaled_loss.backward()

                # Step optimizer after accumulating accum_steps micro-batches
                # or at the final micro-batch of the epoch
                is_accum_boundary = (step + 1) % accum_steps == 0
                is_last_step = (step + 1) == len(loader)
                if is_accum_boundary or is_last_step:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

    avg_loss = total_loss / max(len(all_labels), 1)
    preds = (np.array(all_probs) >= 0.5).astype(int).tolist()
    metrics = compute_metrics(all_labels, preds, all_probs)
    return avg_loss, metrics, all_probs, all_labels, all_conds


# Single fold training
def train_one_fold(
    fold_idx,
    train_list,
    val_list,
    edge_index,
    edge_weight,
    n_parcels,
    cache_dir,
    fold_ckpt_dir,
    device,
    epochs,
    debug,
    masker,
    fresh=False,
    condition_filter=None,
    laughter_mode=False,
    use_film=True,
    static_feat_map=None,
):
    """
    Train one K-fold iteration.
    """
    os.makedirs(fold_ckpt_dir, exist_ok=True)

    train_state_path = os.path.join(fold_ckpt_dir, "train_state.pt")
    if fresh and os.path.exists(train_state_path):
        os.remove(train_state_path)
        print(f"  [Fold {fold_idx+1}] --fresh: deleted {train_state_path}")

    # dataset
    use_sliding = not debug
    train_ds = fMRIWindowDataset(
        train_list, masker,
        use_sliding=use_sliding,
        cache_dir=cache_dir,
        condition_filter=condition_filter,
        augment=True,
        laughter_mode=laughter_mode,
        static_feat_map=static_feat_map,
    )
    val_ds = fMRIWindowDataset(
        val_list, masker,
        use_sliding=False,
        cache_dir=cache_dir,
        condition_filter=condition_filter,
        augment=False,
        laughter_mode=laughter_mode,
        static_feat_map=static_feat_map,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"  Fold {fold_idx+1} Empty dataset after condition filter - skipping.")
        return None, 0.0, None

    n_tr_pos = sum(int(s[2].item()) for s in train_ds.samples)
    n_vl_pos = sum(int(s[2].item()) for s in val_ds.samples)
    if laughter_mode:
        pos_name, neg_name = "conversational", "spontaneous"
    else:
        pos_name, neg_name = "autistic", "non-autistic"
    print(
        f"  Windows - train: {len(train_ds)} "
        f"({pos_name}={n_tr_pos}, {neg_name}={len(train_ds)-n_tr_pos}) | "
        f"val: {len(val_ds)} "
        f"({pos_name}={n_vl_pos}, {neg_name}={len(val_ds)-n_vl_pos})"
    )

    # micro-batch = effective batch / accumulation steps
    micro_bs = max(1, config.BATCH_SIZE // config.GRAD_ACCUM_STEPS)

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=micro_bs, sampler=sampler,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=micro_bs, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    n_static = len(static_feat_map[next(iter(static_feat_map))]) if static_feat_map else 0
    model = build_selected_model(
        edge_index, edge_weight, n_parcels=n_parcels,
        n_static_features=n_static, use_film=use_film,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    # LR schedule: linear warmup from LR*0.1 to LR, then cosine decay to 1e-6
    warmup_epochs = min(config.LR_WARMUP_EPOCHS, epochs // 3)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )
    criterion = nn.BCEWithLogitsLoss()

    _empty_history = {k: [] for k in [
        "epoch",
        "train_loss", "val_loss",
        "train_accuracy", "val_accuracy",
        "train_auc", "val_auc",
        "train_f1_macro", "val_f1_macro",
        "train_f1_asd", "val_f1_asd",
        "train_sensitivity", "val_sensitivity",
        "train_specificity", "val_specificity",
        "train_dice_asd", "val_dice_asd",
        "lr",
    ]}

    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0
    patience_limit = min(config.PATIENCE, epochs)
    start_epoch = 1
    history = _empty_history
    ckpt_every = 2 if debug else config.CHECKPOINT_EVERY

    # Resume from fold-level checkpoint if available
    if config.RESUME_TRAINING and os.path.exists(train_state_path):
        print(f"  [Fold {fold_idx+1}] Resuming from {train_state_path} .")
        state = torch.load(train_state_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        history = state["history"]
        best_val_auc = state["best_val_auc"]
        patience_counter = state["patience_counter"]
        start_epoch = state["epoch"] + 1
        print(
            f"  [Fold {fold_idx+1}] Resumed at epoch {start_epoch} "
            f"(best val AUC: {best_val_auc:.4f})"
        )

    header = (
        f"{'Ep':>4} {'tr_loss':>8} {'tr_acc':>7} {'tr_auc':>7} {'tr_f1':>7} "
        f"{'vl_loss':>8} {'vl_acc':>7} {'vl_auc':>7} {'vl_f1':>7} {'ES':>6}"
    )
    print(header)
    print("-" * len(header))

    for epoch in range(start_epoch, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]

        tr_loss, tr_m, _, _, _ = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )
        vl_loss, vl_m, _, _, _ = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False
        )
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["train_accuracy"].append(tr_m["accuracy"]); history["val_accuracy"].append(vl_m["accuracy"])
        history["train_auc"].append(tr_m["auc"]); history["val_auc"].append(vl_m["auc"])
        history["train_f1_macro"].append(tr_m["f1_macro"]); history["val_f1_macro"].append(vl_m["f1_macro"])
        history["train_f1_asd"].append(tr_m["f1_asd"]); history["val_f1_asd"].append(vl_m["f1_asd"])
        history["train_sensitivity"].append(tr_m["sensitivity"]); history["val_sensitivity"].append(vl_m["sensitivity"])
        history["train_specificity"].append(tr_m["specificity"]); history["val_specificity"].append(vl_m["specificity"])
        history["train_dice_asd"].append(tr_m["dice_asd"]); history["val_dice_asd"].append(vl_m["dice_asd"])
        history["lr"].append(lr)

        es_str = f"{patience_counter}/{patience_limit}"
        print(
            f"{epoch:4d} {tr_loss:8.4f} {tr_m['accuracy']:7.4f} {tr_m['auc']:7.4f} "
            f"{tr_m['f1_macro']:7.4f} {vl_loss:8.4f} {vl_m['accuracy']:7.4f} "
            f"{vl_m['auc']:7.4f} {vl_m['f1_macro']:7.4f} {es_str:>6}"
        )

        if not np.isnan(vl_m["auc"]) and vl_m["auc"] > best_val_auc:
            best_val_auc = vl_m["auc"]
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [Best fold {fold_idx+1}] val AUC={vl_m['auc']:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

        if epoch % ckpt_every == 0:
            ckpt_num = epoch // ckpt_every
            total_ckpts = max(1, epochs // ckpt_every)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
                "best_val_auc": best_val_auc,
                "patience_counter": patience_counter,
                "n_parcels": n_parcels,
                "use_film": use_film,
                "n_static_features": n_static,
            }, train_state_path)
            print(
                f"\n  [Checkpoint {ckpt_num}/{total_ckpts}] "
                f"Epoch {epoch}/{epochs} saved (best val AUC: {best_val_auc:.4f})\n"
            )

    return history, best_val_auc, best_model_state


# Training log
def write_training_log(mode, hyperparams, dataset_info, results, elapsed_seconds):
    """Append one training run entry to training_log.txt."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    h = int(elapsed_seconds // 3600)
    m = int((elapsed_seconds % 3600) // 60)
    s = int(elapsed_seconds % 60)

    lines = [
        "",
        "=" * 30,
        f"Training Run",
        f"Date: {now}",
        f"Mode: {mode}",
        "=" * 30,
        "Hyperparameters:",
        f"  Optimizer: AdamW",
        f"  Learning rate: {hyperparams['lr']}",
        f"  Weight decay: {hyperparams['weight_decay']}",
        f"  Dropout: {hyperparams['dropout']}",
        f"  Window length: {hyperparams['window_trs']} TRs ({hyperparams['window_sec']} s)",
        f"  Batch size: {hyperparams['batch_size']} (micro-batch: {hyperparams['micro_batch']}, accum steps: {hyperparams['grad_accum_steps']})",
        f"  LR warmup: {hyperparams['lr_warmup']} epochs",
        f"  Mixup alpha: {hyperparams['mixup_alpha']}",
        f"  Augment noise std: {hyperparams['augment_noise_std']}",
        f"  Augment parcel drop: {hyperparams['augment_parcel_drop']}",
        f"  Max epochs: {hyperparams['epochs']}",
        f"  Patience: {hyperparams['patience']}",
        f"  K-folds: {hyperparams['k_folds']}",
        "",
        "Dataset split (subjects):",
        f"  Train+val: {dataset_info.get('n_trainval_subjects', 'N/A')}",
        f"  Test: {dataset_info.get('n_test_subjects', 'N/A')}",
        "",
        "Dataset split (windows, fold 1 shown):",
    ]
    for split in ["train", "val", "test"]:
        d = dataset_info.get(split, {})
        if d:
            lines.append(
                f"  {split.capitalize()}: {d.get('n_windows', 0)} windows "
                f"(autistic: {d.get('n_asd_windows', 0)}, "
                f"non-autistic: {d.get('n_nt_windows', 0)}, "
                f"balance: {d.get('class_balance', 0):.3f})"
            )
    # Train-test split subject IDs
    trainval_ids = dataset_info.get("trainval_subject_ids", [])
    test_ids = dataset_info.get("test_subject_ids", [])
    if trainval_ids or test_ids:
        lines.append("")
        lines.append("Subject split (IDs):")
        if test_ids:
            test_asd = [sid for sid, lbl in test_ids if lbl == 1]
            test_nt = [sid for sid, lbl in test_ids if lbl == 0]
            lines.append(f"  Test autistic ({len(test_asd)}): {', '.join(sorted(test_asd))}")
            lines.append(f"  Test non-autistic ({len(test_nt)}): {', '.join(sorted(test_nt))}")
        if trainval_ids:
            tv_asd = [sid for sid, lbl in trainval_ids if lbl == 1]
            tv_nt = [sid for sid, lbl in trainval_ids if lbl == 0]
            lines.append(f"  TrainVal autistic ({len(tv_asd)}): {', '.join(sorted(tv_asd))}")
            lines.append(f"  TrainVal non-autistic ({len(tv_nt)}): {', '.join(sorted(tv_nt))}")

    lines += ["", "Results:"]
    for key, val in results.items():
        if isinstance(val, float):
            lines.append(f"  {key}: {val:.4f}")
        else:
            lines.append(f"  {key}: {val}")
    lines += [
        f"  Training time: {h}h {m}m {s}s",
        "=" * 30,
    ]

    with open(LOG_FILE, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nTraining log updated: {LOG_FILE}")


def append_training_progress(mode, message):
    """Append a progress line to training_log.txt."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{now}] {mode}: {message}\n")


# Main:
def train(
    bids_root=config.BIDS_ROOT,
    atlas_path=config.ATLAS_PATH,
    cache_dir=config.CACHE_DIR,
    checkpoint_dir=config.CHECKPOINT_DIR,
    debug=False,
    n_subjects=None,
    n_epochs=None,
    fresh=False,
    condition_filter=None,
    laughter_mode=False,
    use_film=True,
    use_demographics=False,
):
    """
    Train the ST-GCN (+ optional FiLM) model with K-fold cross-validation.
    """
    run_start = time.time()

    if debug:
        n_subjects = n_subjects or 6
        n_epochs = n_epochs or 5
        checkpoint_dir = checkpoint_dir.rstrip("/") + "_debug"
        cache_dir = cache_dir.rstrip("/") + "_debug"
        print("=" * 30)
        print(f"DEBUG MODE: subjects={n_subjects}, epochs={n_epochs}, no sliding windows")
        print(f"  checkpoint_dir: {checkpoint_dir}")
        print("=" * 30 + "\n")

    # Condition-specific
    if condition_filter is not None:
        cond_name = config.CONDITION_MAP.get(condition_filter + 1, f"cond{condition_filter + 1}")
        checkpoint_dir = checkpoint_dir.rstrip("/") + f"_cond{condition_filter + 1}"
        print(f"\nCONDITION MODE: condition {condition_filter + 1}: {cond_name}\n")

    # Laughter type classification
    if laughter_mode:
        checkpoint_dir = checkpoint_dir.rstrip("/") + "_laughter"
        print("\n[LAUGHTER MODE: Classify spontaneous (0) vs conversational (1) laughter")

    # ST-GCN only
    if not use_film:
        if checkpoint_dir == config.CHECKPOINT_DIR:
            checkpoint_dir = config.STGCN_ONLY_CHECKPOINT_DIR
        elif condition_filter is not None and not checkpoint_dir.endswith("_stgcn_only"):
            checkpoint_dir = checkpoint_dir.rstrip("/") + "_stgcn_only"
        print("\nST-GCN ONLY MODE: FiLM conditioning disabled.")
        print("  Condition regressors are passed to the model but ignored.")
        print(f"  Checkpoint dir: {checkpoint_dir}\n")

    # Demographics
    if use_demographics:
        print("\nDEMOGRAPHICS MODE: Static features enabled:")
        print(f"  Features : {config.STATIC_FEATURES}")
        if getattr(config, "EXCLUDED_STATIC_FEATURES", None):
            print(f"  Excluded : {config.EXCLUDED_STATIC_FEATURES}")
        print(f"  Non-autistic file : {config.DEMOGRAPHIC_NT_PATH}")
        print(f"  Autistic file     : {config.DEMOGRAPHIC_ASD_PATH}\n")

    epochs = n_epochs if n_epochs is not None else config.EPOCHS
    n_folds = 2 if debug else config.K_FOLDS

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    fig_dir = os.path.join(checkpoint_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Subject list and splits
    subject_list = load_subject_list(bids_root)
    if n_subjects is not None:
        subject_list = subsample_subjects(subject_list, n_subjects)

    trainval_list, test_list = split_test_subjects(subject_list)
    fold_splits = kfold_subjects(trainval_list, n_folds=n_folds)
    print(
        f"\nK-fold CV: {len(fold_splits)} folds | "
        f"trainval: {len(trainval_list)} subjects | "
        f"test: {len(test_list)} subjects\n"
    )

    # 2. Demographics
    static_feat_map = None
    if use_demographics and config.STATIC_FEATURES:
        demo_df = load_demographics()
        raw_map = build_feature_map(demo_df, config.STATIC_FEATURES)
        trainval_ids = [sid for sid, _ in trainval_list]
        feat_mean, feat_std = compute_normalisation_stats(raw_map, trainval_ids)
        static_feat_map = normalise_feature_map(raw_map, feat_mean, feat_std)
        n_matched = sum(1 for sid, _ in subject_list if sid in static_feat_map)
        print(
            f"Demographics loaded: {n_matched}/{len(subject_list)} subjects matched. "
            f"Features: {config.STATIC_FEATURES}"
        )
        print(
            f"  Normalisation (trainval) - "
            f"mean: {feat_mean.round(3)} | std: {feat_std.round(3)}\n"
        )

    # 3. Masker
    masker = NiftiLabelsMasker(
        labels_img=atlas_path, standardize=True, detrend=False,
        t_r=config.TR, memory="nilearn_cache", memory_level=1, verbose=0,
    )

    # 4. Population graph - built from all trainval subjects to maximise data
    graph_cache = os.path.join(cache_dir, f"population_graph_t{config.CORR_THRESHOLD}.npz")
    print("Collecting time-series for population graph (uses TS cache if available).")
    train_ts_list = []
    for subj_id, _ in trainval_list:
        for run in range(1, 5):
            ts = get_run_timeseries(subj_id, f"{run:02d}", masker, cache_dir)
            if ts is not None:
                train_ts_list.append(ts)

    edge_index, edge_weight = build_population_graph(train_ts_list, graph_cache)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    n_parcels = train_ts_list[0].shape[1] if train_ts_list else config.N_PARCELS
    print(f"Atlas parcels: {n_parcels}\n")

    # 5. K-fold training loop
    fold_results = []
    best_overall_auc = 0.0
    best_history = None

    for fold_idx, (train_list, val_list) in enumerate(fold_splits):
        print(f"\n{'='*30}")
        n_tr_asd = sum(l for _, l in train_list)
        n_vl_asd = sum(l for _, l in val_list)
        print(
            f"FOLD {fold_idx+1}/{len(fold_splits)} - "
            f"train: {len(train_list)} subjects "
            f"({n_tr_asd} autistic / {len(train_list)-n_tr_asd} non-autistic) | "
            f"val: {len(val_list)} subjects "
            f"({n_vl_asd} autistic / {len(val_list)-n_vl_asd} non-autistic)"
        )
        print(f"{'='*30}\n")

        fold_ckpt_dir = os.path.join(checkpoint_dir, f"fold_{fold_idx+1}")

        history, best_fold_auc, best_model_state = train_one_fold(
            fold_idx=fold_idx,
            train_list=train_list,
            val_list=val_list,
            edge_index=edge_index,
            edge_weight=edge_weight,
            n_parcels=n_parcels,
            cache_dir=cache_dir,
            fold_ckpt_dir=fold_ckpt_dir,
            device=device,
            epochs=epochs,
            debug=debug,
            masker=masker,
            fresh=fresh,
            condition_filter=condition_filter,
            laughter_mode=laughter_mode,
            use_film=use_film,
            static_feat_map=static_feat_map,
        )

        if history is None:
            continue

        fold_results.append({"fold": fold_idx + 1, "best_val_auc": best_fold_auc})

        with open(os.path.join(fold_ckpt_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # Single best model only across all folds
        if best_fold_auc > best_overall_auc and best_model_state is not None:
            best_overall_auc = best_fold_auc
            best_history = history
            n_static = len(static_feat_map[next(iter(static_feat_map))]) if static_feat_map else 0
            model_tmp = build_selected_model(
                edge_index, edge_weight, n_parcels=n_parcels,
                n_static_features=n_static, use_film=use_film,
            ).to(device)
            model_tmp.load_state_dict(
                {k: v.to(device) for k, v in best_model_state.items()}
            )
            torch.save({
                "fold": fold_idx + 1,
                "model_state": {k: v.cpu() for k, v in model_tmp.state_dict().items()},
                "val_auc": best_fold_auc,
                "n_parcels": n_parcels,
                "use_film": use_film,
                "n_static_features": n_static,
            }, os.path.join(checkpoint_dir, "best_model.pt"))
            print(
                f"\n  Best overall: fold {fold_idx+1} "
                f"val AUC={best_fold_auc:.4f} - best_model.pt saved"
            )

        print(f"\nFold {fold_idx+1} complete - best val AUC: {best_fold_auc:.4f}")
        append_training_progress(
            "kfold",
            f"{checkpoint_dir} fold {fold_idx+1}/{len(fold_splits)} best_val_auc={best_fold_auc:.4f}",
        )

    # K-fold summary
    if fold_results:
        mean_auc = np.mean([r["best_val_auc"] for r in fold_results])
        std_auc = np.std([r["best_val_auc"] for r in fold_results])
        print(f"\n{'='*30}")
        print(f"K-FOLD SUMMARY ({len(fold_results)} folds)")
        print(f"  Val AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
        for r in fold_results:
            print(f"  Fold {r['fold']}: val AUC={r['best_val_auc']:.4f}")
        print(f"{'='*30}\n")

    # 6. Test dataset (uses micro-batch)
    micro_bs = max(1, config.BATCH_SIZE // config.GRAD_ACCUM_STEPS)
    test_ds = fMRIWindowDataset(
        test_list, masker, use_sliding=False,
        cache_dir=cache_dir, condition_filter=condition_filter,
        laughter_mode=laughter_mode,
        static_feat_map=static_feat_map,
    )
    test_loader = DataLoader(
        test_ds, batch_size=micro_bs, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    # Descriptive stats (fold 1 train/val)
    desc_stats = {}
    if fold_splits:
        first_train_ds = fMRIWindowDataset(
            fold_splits[0][0], masker, use_sliding=not debug,
            cache_dir=cache_dir, condition_filter=condition_filter,
            laughter_mode=laughter_mode,
            static_feat_map=static_feat_map,
        )
        first_val_ds = fMRIWindowDataset(
            fold_splits[0][1], masker, use_sliding=False,
            cache_dir=cache_dir, condition_filter=condition_filter,
            laughter_mode=laughter_mode,
            static_feat_map=static_feat_map,
        )
        desc_stats = {
            "train": dataset_descriptive_stats(first_train_ds, "train"),
            "val": dataset_descriptive_stats(first_val_ds, "val"),
            "test": dataset_descriptive_stats(test_ds, "test"),
        }

    # 7. Test evaluation with best model
    print("\n" + "=" * 30)
    print("TEST EVALUATION (best checkpoint across all folds)")
    print("=" * 30)

    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    n_static = len(static_feat_map[next(iter(static_feat_map))]) if static_feat_map else 0
    if not os.path.exists(best_model_path):
        print("\n  best_model.pt not found - saving current model as fallback.\n")
        model_fb = build_selected_model(
            edge_index, edge_weight, n_parcels=n_parcels,
            n_static_features=n_static, use_film=use_film,
        ).to(device)
        torch.save({
            "fold": 0, "model_state": model_fb.state_dict(),
            "val_auc": float("nan"), "n_parcels": n_parcels,
            "use_film": use_film,
            "n_static_features": n_static,
        }, best_model_path)

    ckpt = torch.load(best_model_path, map_location=device)
    model = build_selected_model(
        edge_index, edge_weight, n_parcels=n_parcels,
        n_static_features=n_static, use_film=use_film,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    all_probs, all_labels, all_conds = [], [], []
    with torch.no_grad():
        for X, cond_reg, y_group, y_cond, static_feat in test_loader:
            X        = X.to(device)
            cond_reg = cond_reg.to(device)
            sf       = static_feat.to(device) if static_feat.shape[-1] > 0 else None
            probs    = torch.sigmoid(model(X, cond_reg, sf)).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y_group.numpy().tolist())
            all_conds.extend(y_cond.numpy().tolist())

    all_preds = (np.array(all_probs) >= 0.5).astype(int).tolist()
    test_metrics = compute_metrics(all_labels, all_preds, all_probs)

    if laughter_mode:
        pos_name, neg_name = "conversational (1)", "spontaneous (0)"
    else:
        pos_name, neg_name = "autistic (1)", "non-autistic (0)"

    print(f"\n  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  AUC-ROC: {test_metrics['auc']:.4f}")
    print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"  F1 {pos_name}: {test_metrics['f1_asd']:.4f} | F1 {neg_name}: {test_metrics['f1_nt']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  Dice {pos_name}: {test_metrics['dice_asd']:.4f}")
    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    print(f"  Pred {neg_name} / Pred {pos_name}")
    print(f"  True {neg_name}: {test_metrics['tn']} / {test_metrics['fp']}")
    print(f"  True {pos_name}: {test_metrics['fn']} / {test_metrics['tp']}")
    print(f"\n  Classification report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=[neg_name, pos_name],
        zero_division=0,
    ))

    # Per-condition test metrics
    cond_metrics = per_condition_metrics(all_labels, all_preds, all_probs, all_conds)
    print(f"\n  Per-condition test metrics:")
    print(f"  Condition | N | AUC | Acc | F1 | Sens | Spec")
    for cond_name, cm in cond_metrics.items():
        if "note" in cm:
            print(f"  {cond_name}: n={cm['n_samples']} ({cm['note']})")
        else:
            print(
                f"  {cond_name}: n={cm['n_samples']} "
                f"AUC={cm['auc']:.4f} Acc={cm['accuracy']:.4f} "
                f"F1={cm['f1_macro']:.4f} Sens={cm['sensitivity']:.4f} Spec={cm['specificity']:.4f}"
            )

    # Subject-level evaluation: aggregate window predictions per subject.
    # p_subject = mean(p_window) over all windows for that subject, then
    # threshold at 0.5
    subj_metrics = {}
    cond_subj_metrics = {}
    test_subject_ids = test_ds.subject_ids
    if test_subject_ids and not laughter_mode:
        subj_probs_map = defaultdict(list) # {subject_id: [prob1, prob2, ...]}
        subj_label_map = {} # {subject_id: group_label}
        for prob, label, sid in zip(all_probs, all_labels, test_subject_ids):
            subj_probs_map[sid].append(prob)
            subj_label_map[sid] = int(label)

        # Mean probability per subject
        ordered_sids = list(subj_probs_map.keys())
        subj_y_true = [subj_label_map[s] for s in ordered_sids]
        subj_y_prob = [float(np.mean(subj_probs_map[s])) for s in ordered_sids]
        subj_y_pred = [int(p >= 0.5) for p in subj_y_prob]
        subj_metrics = compute_metrics(subj_y_true, subj_y_pred, subj_y_prob)

        print(f"\n{'='*30}")
        print("SUBJECT-LEVEL TEST EVALUATION (mean probability per subject)")
        print(f"{'='*30}")
        print(f"  Subjects tested: {len(ordered_sids)}")
        for sid in ordered_sids:
            n_wins = len(subj_probs_map[sid])
            mean_p = np.mean(subj_probs_map[sid])
            true_lbl = "autistic" if subj_label_map[sid] == 1 else "non-autistic"
            pred_lbl = "autistic" if mean_p >= 0.5 else "non-autistic"
            correct = "OK" if (mean_p >= 0.5) == (subj_label_map[sid] == 1) else "MISS"
            print(
                f"  sub-{sid}: {n_wins} windows, "
                f"mean_prob={mean_p:.4f}, true={true_lbl}, pred={pred_lbl} [{correct}]"
            )
        print(f"\n  Subject Accuracy: {subj_metrics['accuracy']:.4f}")
        print(f"  Subject AUC-ROC: {subj_metrics['auc']:.4f}")
        print(f"  Subject Sensitivity: {subj_metrics['sensitivity']:.4f}")
        print(f"  Subject Specificity: {subj_metrics['specificity']:.4f}")
        print(f"  Subject F1 (macro): {subj_metrics['f1_macro']:.4f}")

        # Per-condition subject-level evaluation from the full model.
        cond_subj_metrics = per_condition_subject_metrics(
            all_labels, all_probs, all_conds, test_subject_ids
        )
        print(f"\n{'='*30}")
        print("PER-CONDITION SUBJECT-LEVEL (full model, per-condition windows)")
        print(f"{'='*30}")
        print(f"  {'Condition':<25} {'Subj':>5} {'AUC':>7} {'Sens':>7} {'Spec':>7}")
        for cond_name, cm in cond_subj_metrics.items():
            if "note" in cm:
                print(f"  {cond_name:<25} {cm['n_subjects']:>5}   ({cm['note']})")
            else:
                print(
                    f"  {cond_name:<25} {cm['n_subjects']:>5} "
                    f"{cm['auc']:>7.4f} {cm['sensitivity']:>7.4f} "
                    f"{cm['specificity']:>7.4f}"
                )
                for det in cm.get("subject_details", []):
                    true_str = "autistic" if det["true_label"] == 1 else "non-autistic"
                    pred_str = "autistic" if det["mean_prob"] >= 0.5 else "non-autistic"
                    match = "OK" if (det["mean_prob"] >= 0.5) == (det["true_label"] == 1) else "MISS"
                    print(
                        f"    sub-{det['id']}: p={det['mean_prob']:.4f} "
                        f"({det['n_windows']}w) true={true_str} "
                        f"pred={pred_str} [{match}]"
                    )
    print("=" * 30)

    # 7. Save results
    results_full = {
        "test_metrics": test_metrics,
        "subject_level_metrics": subj_metrics,
        "per_condition_metrics": {
            k: {ek: ev for ek, ev in v.items() if ek != "confusion_matrix"}
            for k, v in cond_metrics.items()
        },
        "per_condition_subject_metrics": {
            k: {ek: ev for ek, ev in v.items() if ek != "confusion_matrix"}
            for k, v in cond_subj_metrics.items()
        },
        "best_val_auc": best_overall_auc,
        "best_fold": int(ckpt.get("fold", 0)),
        "fold_results": fold_results,
        "total_params": total_params,
        "descriptive_stats": desc_stats,
        "test_probs": all_probs,
        "test_labels": all_labels,
        "test_preds": all_preds,
        "test_conds": all_conds,
        "test_subject_ids": list(test_ds.subject_ids),
    }
    with open(os.path.join(checkpoint_dir, "results_full.json"), "w") as f:
        json.dump(results_full, f, indent=2)

    if best_history:
        with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
            json.dump(best_history, f, indent=2)

    print(f"\nResults saved to {checkpoint_dir}/results_full.json")

    # 8. Plots
    try:
        from evaluate import plot_all
        plot_all(history=best_history or {}, results_full=results_full, fig_dir=fig_dir)
        print(f"Figures saved to {fig_dir}/")
    except Exception as e:
        print(f"Figures skipped: {e}")

    # 9. Training log
    n_test_asd = sum(l for _, l in test_list)
    dataset_info = {
        "n_trainval_subjects": (
            f"{len(trainval_list)} "
            f"({sum(l for _, l in trainval_list)} autistic / "
            f"{sum(1-l for _, l in trainval_list)} non-autistic)"
        ),
        "n_test_subjects": (
            f"{len(test_list)} "
            f"({n_test_asd} autistic / "
            f"{len(test_list)-n_test_asd} non-autistic)"
        ),
        "trainval_subject_ids": trainval_list, # [(id, label), ...]
        "test_subject_ids": test_list, # [(id, label), ...]
    }
    if desc_stats:
        dataset_info.update(desc_stats)

    if laughter_mode:
        mode_str = "laughter_type"
    elif not use_film:
        mode_str = "stgcn_only"
    elif debug:
        mode_str = "debug"
    elif condition_filter is not None:
        mode_str = f"condition-{condition_filter+1}-{config.CONDITION_MAP.get(condition_filter+1,'')}"
    else:
        mode_str = "full"
    log_results = {
        "best_val_auc": best_overall_auc,
        "test_auc": test_metrics["auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_sensitivity": test_metrics["sensitivity"],
        "test_specificity": test_metrics["specificity"],
        "k_folds_completed": len(fold_results),
    }
    if subj_metrics:
        log_results["subj_accuracy"] = subj_metrics["accuracy"]
        log_results["subj_auc"] = subj_metrics["auc"]
        log_results["subj_sensitivity"] = subj_metrics["sensitivity"]
        log_results["subj_specificity"] = subj_metrics["specificity"]
    write_training_log(
        mode=mode_str,
        hyperparams={
            "lr": config.LR,
            "weight_decay": config.WEIGHT_DECAY,
            "dropout": config.DROPOUT,
            "window_trs": config.WINDOW_TRS,
            "window_sec": config.WINDOW_DURATION_SEC,
            "batch_size": config.BATCH_SIZE,
            "micro_batch": max(1, config.BATCH_SIZE // config.GRAD_ACCUM_STEPS),
            "grad_accum_steps": config.GRAD_ACCUM_STEPS,
            "lr_warmup": config.LR_WARMUP_EPOCHS,
            "mixup_alpha": config.MIXUP_ALPHA,
            "augment_noise_std": config.AUGMENT_NOISE_STD,
            "augment_parcel_drop": config.AUGMENT_PARCEL_DROP,
            "epochs": epochs,
            "patience": config.PATIENCE,
            "k_folds": n_folds,
        },
        dataset_info=dataset_info,
        results=log_results,
        elapsed_seconds=time.time() - run_start,
    )

    return model, best_history, results_full


# LOSO cross-validation
def loso_subject_splits(subject_list):
    for i in range(len(subject_list)):
        test_subj = subject_list[i]
        trainval = [s for j, s in enumerate(subject_list) if j != i]
        yield trainval, test_subj


def loso_inner_split(trainval_list, val_frac=None):
    """
    Stratified train / inner-val split of the N-1 LOSO training subjects.
    """
    if val_frac is None:
        val_frac = config.LOSO_INNER_VAL_FRAC

    ids = [s[0] for s in trainval_list]
    labels = [s[1] for s in trainval_list]

    n_val = max(2, int(len(trainval_list) * val_frac)) # at least 2 val subjects
    n_val = min(n_val, len(trainval_list) - 2) # leave at least 2 for train

    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=config.SEED)
    train_idx, val_idx = next(sss.split(ids, labels))

    return (
        [trainval_list[i] for i in train_idx],
        [trainval_list[i] for i in val_idx],
    )


def get_raw_logits(model, loader, device):
    """
    Raw model logits (before sigmoid) and true labels from a DataLoader.
    Used for Platt calibration: calibration
    """
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, cond_reg, y_group, _, static_feat in loader:
            X        = X.to(device)
            cond_reg = cond_reg.to(device)
            sf       = static_feat.to(device) if static_feat.shape[-1] > 0 else None
            logits   = model(X, cond_reg, sf).cpu().numpy() # (B,) raw logits
            all_logits.extend(logits.tolist())
            all_labels.extend(y_group.numpy().tolist())
    return all_logits, all_labels


def get_raw_logits_with_subjects(model, loader, device):
    """
    Collect raw logits, labels, and subject IDs.
    """
    model.eval()
    all_logits, all_labels, all_subjects = [], [], []
    subject_ids = getattr(loader.dataset, "subject_ids", [])
    cursor = 0

    with torch.no_grad():
        for X, cond_reg, y_group, _, static_feat in loader:
            X        = X.to(device)
            cond_reg = cond_reg.to(device)
            sf       = static_feat.to(device) if static_feat.shape[-1] > 0 else None
            logits   = model(X, cond_reg, sf).cpu().numpy()
            labels   = y_group.numpy().tolist()
            n_batch  = len(labels)
            batch_subjects = subject_ids[cursor:cursor + n_batch]
            cursor += n_batch

            all_logits.extend(logits.tolist())
            all_labels.extend(labels)
            all_subjects.extend(batch_subjects)

    return all_logits, all_labels, all_subjects


def aggregate_subject_logits(logits, labels, subject_ids):
    """
    Aggregate window logits to one mean logit per subject.
    z_subject = mean(z_window)
    """
    subj_logits = defaultdict(list)
    subj_labels = {}
    for logit, label, sid in zip(logits, labels, subject_ids):
        subj_logits[sid].append(float(logit))
        subj_labels[sid] = int(label)

    ordered = list(subj_logits.keys())
    mean_logits = [float(np.mean(subj_logits[sid])) for sid in ordered]
    y_subject = [subj_labels[sid] for sid in ordered]
    counts = [len(subj_logits[sid]) for sid in ordered]
    return ordered, mean_logits, y_subject, counts


def platt_calibrate(val_logits, val_labels):
    """
    Fit Platt scaling on inner validation logits.

    Platt scaling fits a logistic regression on the scalar logit:
        p_calibrated = sigma(a * logit + b)
    where a, b are estimated by maximum likelihood on the inner val set.
    """
    val_arr = np.array(val_logits, dtype=np.float32).reshape(-1, 1)
    lbl_arr = np.array(val_labels, dtype=int)

    if len(np.unique(lbl_arr)) < 2:
        return None # calibration requires both classes

    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    lr.fit(val_arr, lbl_arr)
    return lr


def probabilities_from_logits(calibrator, logits):
    """
    Convert subject-level logits to calibrated probabilities.
    """
    logits_arr = np.array(logits, dtype=np.float32).reshape(-1, 1)
    if calibrator is not None:
        return calibrator.predict_proba(logits_arr)[:, 1].astype(float).tolist()
    return (1.0 / (1.0 + np.exp(-logits_arr.ravel()))).astype(float).tolist()


def select_subject_threshold(y_true, y_prob):
    """
    Select an inner-validation threshold by balanced accuracy.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5

    scores = sorted(set(y_prob.tolist()))
    candidates = [0.5]
    candidates += [scores[0] - 1e-6, scores[-1] + 1e-6]
    candidates += [(a + b) / 2.0 for a, b in zip(scores, scores[1:])]

    best_score = (-1.0, -1.0, -1.0)
    best_threshold = 0.5
    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_acc = 0.5 * (sensitivity + specificity)
        accuracy = accuracy_score(y_true, y_pred)
        tie_break = -abs(threshold - 0.5)
        score = (balanced_acc, accuracy, tie_break)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold)


def fit_subject_calibration(val_logits, val_labels, val_subject_ids):
    """
    Fit calibration and threshold selection on inner-validation subjects.
    """
    val_sids, val_mean_logits, val_y, val_counts = aggregate_subject_logits(
        val_logits, val_labels, val_subject_ids
    )
    calibrator = platt_calibrate(val_mean_logits, val_y)
    val_probs = probabilities_from_logits(calibrator, val_mean_logits)
    threshold = select_subject_threshold(val_y, val_probs)
    return {
        "calibrator": calibrator,
        "threshold": threshold,
        "subject_ids": val_sids,
        "subject_logits": val_mean_logits,
        "subject_probs": val_probs,
        "subject_labels": val_y,
        "subject_window_counts": val_counts,
    }


def apply_subject_calibration(calibration, logits):
    """
    Aggregate test-window logits then apply subject-level calibration.
    """
    mean_logit = float(np.mean(logits))
    prob = probabilities_from_logits(calibration["calibrator"], [mean_logit])[0]
    return prob, mean_logit


def apply_calibration(calibrator, logits):
    """
    Apply Platt calibrator to a list of raw logits and return the mean
    calibrated probability across all windows of one subject.

    Subject-level probability aggregation:
        p_subject = (1 / M) * sum_{m=1}^{M} p_calibrated(logit_m)
    where M is the number of windows for this subject.
    if calibrator is None (single class in val):
        sigma(logit) = 1 / (1 + exp(-logit))
    """
    logits_arr = np.array(logits, dtype=np.float32).reshape(-1, 1)
    if calibrator is not None:
        probs = calibrator.predict_proba(logits_arr)[:, 1] # P(autistic)
    else:
        probs = 1.0 / (1.0 + np.exp(-logits_arr.ravel())) # sigmoid fallback
    return float(np.mean(probs))


def train_one_loso_fold(
    fold_label,
    train_list,
    val_list,
    edge_index,
    edge_weight,
    n_parcels,
    cache_dir,
    fold_ckpt_dir,
    device,
    epochs,
    masker,
    condition_filter=None,
    static_feat_map=None,
    use_film=True,
):
    """
    Train one LOSO fold. Returns the best model state and inner val logits
    for Platt calibration.
    """
    os.makedirs(fold_ckpt_dir, exist_ok=True)

    train_ds = fMRIWindowDataset(
        train_list, masker,
        use_sliding=True, # sliding windows as augmentation during training
        cache_dir=cache_dir,
        condition_filter=condition_filter,
        augment=True,
        static_feat_map=static_feat_map,
    )
    val_ds = fMRIWindowDataset(
        val_list, masker,
        use_sliding=False, # no sliding on inner val for clean calibration
        cache_dir=cache_dir,
        condition_filter=condition_filter,
        augment=False,
        static_feat_map=static_feat_map,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"  [LOSO {fold_label}] Empty dataset - skipping fold.")
        return None, [], [], [], 0.0, {}

    micro_bs = max(1, config.BATCH_SIZE // config.GRAD_ACCUM_STEPS)
    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=micro_bs, sampler=sampler,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=micro_bs, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    n_static = len(next(iter(static_feat_map.values()))) if static_feat_map else 0
    model = build_selected_model(
        edge_index, edge_weight, n_parcels=n_parcels,
        n_static_features=n_static, use_film=use_film,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    warmup_epochs = min(config.LR_WARMUP_EPOCHS, epochs // 3)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0
    patience_limit = config.LOSO_PATIENCE

    # Per-epoch history for learning curve visualisation
    fold_history = {"epochs": [], "train_loss": [], "val_auc": [], "best_auc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_m, _, _, _ = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )
        vl_loss, vl_m, _, _, _ = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False
        )
        scheduler.step()

        if not np.isnan(vl_m["auc"]) and vl_m["auc"] > best_val_auc:
            best_val_auc = vl_m["auc"]
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(
                    f"  [{fold_label}] Early stop at epoch {epoch} "
                    f"(best val AUC={best_val_auc:.4f})"
                )
                break

        # Every epoch for learning curve
        fold_history["epochs"].append(epoch)
        fold_history["train_loss"].append(round(float(tr_loss), 6))
        fold_history["val_auc"].append(round(float(vl_m["auc"]), 6))
        fold_history["best_auc"].append(round(float(best_val_auc), 6))

        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"  [{fold_label}] ep={epoch}/{epochs} "
                f"tr_loss={tr_loss:.4f} vl_auc={vl_m['auc']:.4f} "
                f"best={best_val_auc:.4f} ES={patience_counter}/{patience_limit}"
            )

    fold_history["early_stop_epoch"] = len(fold_history["epochs"])
    fold_history["best_val_auc"] = round(float(best_val_auc), 6)

    if best_model_state is None:
        return None, [], [], [], 0.0, fold_history

    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    val_logits, val_labels, val_subject_ids = get_raw_logits_with_subjects(
        model, val_loader, device
    )

    return best_model_state, val_logits, val_labels, val_subject_ids, best_val_auc, fold_history


def train_loso(
    bids_root=config.BIDS_ROOT,
    atlas_path=config.ATLAS_PATH,
    cache_dir=config.CACHE_DIR,
    checkpoint_dir=None,
    n_epochs=None,
    fresh=False,
    condition_filter=None,
    use_demographics=False,
    use_film=True,
    seed=None,
):
    """
    LOSO cross-validation for subject-level
    """
    run_seed = config.SEED if seed is None else seed

    if checkpoint_dir is None:
        checkpoint_dir = (
            config.LOSO_CHECKPOINT_DIR if use_film
            else config.LOSO_STGCN_ONLY_CHECKPOINT_DIR
        )

    if seed is not None and seed != config.SEED:
        checkpoint_dir = checkpoint_dir.rstrip("/") + f"_seed{run_seed}"

    if condition_filter is not None:
        cond_name = config.CONDITION_MAP.get(condition_filter + 1, f"cond{condition_filter + 1}")
        checkpoint_dir = checkpoint_dir.rstrip("/") + f"_cond{condition_filter + 1}"
        print(f"\nLOSO CONDITION MODE: condition {condition_filter + 1}: {cond_name}\n")

    if not use_film:
        print("\nLOSO ST-GCN ONLY MODE: FiLM conditioning disabled.")
        print(f"  Checkpoint dir: {checkpoint_dir}\n")

    epochs = n_epochs if n_epochs is not None else config.LOSO_EPOCHS
    run_start = time.time()
    set_seed(run_seed)
    print(f"LOSO run seed: {run_seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"LOSO CV: {epochs} epochs/fold, patience={config.LOSO_PATIENCE}\n")

    os.makedirs(checkpoint_dir, exist_ok=True)
    fig_dir = os.path.join(checkpoint_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    masker = NiftiLabelsMasker(
        labels_img=atlas_path, standardize=True, detrend=False,
        t_r=config.TR, memory="nilearn_cache", memory_level=1, verbose=0,
    )

    subject_list = load_subject_list(bids_root)
    n_asd = sum(l for _, l in subject_list)
    n_nt = len(subject_list) - n_asd
    print(
        f"LOSO: {len(subject_list)} subjects total "
        f"({n_asd} autistic / {n_nt} non-autistic)"
    )
    print(f"      {len(subject_list)} folds (one held-out subject per fold)\n")

    # Load raw demographics once; per-fold z-scoring is computed inside the loop
    # using only the N-1 trainval subjects
    raw_demo_map = None
    if use_demographics and config.STATIC_FEATURES:
        demo_df = load_demographics()
        raw_demo_map = build_feature_map(demo_df, config.STATIC_FEATURES)
        n_matched = sum(1 for sid, _ in subject_list if sid in raw_demo_map)
        print(
            f"  Demographics: {config.STATIC_FEATURES} | "
            f"{n_matched}/{len(subject_list)} subjects matched\n"
        )

    loso_state_path = os.path.join(checkpoint_dir, "loso_state.json")
    if not fresh and os.path.exists(loso_state_path):
        with open(loso_state_path) as f:
            loso_state = json.load(f)
        loso_state.setdefault("subject_preds", {})
        loso_state.setdefault("subject_thresholds", {})
        loso_state.setdefault("subject_mean_logits", {})
        loso_state.setdefault("fold_calibration", {})
        n_done = len(loso_state["completed_folds"])
        print(f"Resuming LOSO: {n_done}/{len(subject_list)} folds already completed.\n")
    else:
        loso_state = {
            "completed_folds": [],
            "subject_probs": {},
            "subject_labels": {},
            "subject_preds": {},
            "subject_thresholds": {},
            "subject_mean_logits": {},
            "subject_window_counts": {},
            "fold_val_aucs": {},
            "fold_histories": {},
            "fold_calibration": {},
        }

    # Main LOSO loop
    for fold_i, (trainval_list, test_subj) in enumerate(loso_subject_splits(subject_list)):
        test_id, test_label = test_subj

        if test_id in loso_state["completed_folds"]:
            print(
                f"[LOSO {fold_i+1}/{len(subject_list)}] sub-{test_id}: "
                f"already completed, skipping."
            )
            continue

        set_seed(1000 * run_seed + fold_i)

        print(f"\n{'='*30}")
        print(
            f"LOSO FOLD {fold_i+1}/{len(subject_list)}: "
            f"held-out sub-{test_id} "
            f"({'autistic' if test_label == 1 else 'non-autistic'})"
        )
        print(f"{'='*30}")

        train_list, val_list = loso_inner_split(trainval_list)
        n_tr_asd = sum(l for _, l in train_list)
        n_vl_asd = sum(l for _, l in val_list)
        print(
            f"  Inner train: {len(train_list)} subjects "
            f"({n_tr_asd} autistic / {len(train_list)-n_tr_asd} non-autistic) | "
            f"inner val: {len(val_list)} subjects "
            f"({n_vl_asd} autistic / {len(val_list)-n_vl_asd} non-autistic)"
        )

        static_feat_map = None
        if raw_demo_map is not None:
            train_ids = [sid for sid, _ in train_list]
            feat_mean, feat_std = compute_normalisation_stats(raw_demo_map, train_ids)
            static_feat_map = normalise_feature_map(raw_demo_map, feat_mean, feat_std)

        loso_graph_cache = os.path.join(
            cache_dir,
            f"population_graph_loso_excl-{test_id}_innertrain_t{config.CORR_THRESHOLD}.npz",
        )
        print(f"  Collecting TS for LOSO graph from inner train subjects.")
        loso_ts_list = []
        for subj_id, _ in train_list:
            for run in range(1, 5):
                ts = get_run_timeseries(subj_id, f"{run:02d}", masker, cache_dir)
                if ts is not None:
                    loso_ts_list.append(ts)

        edge_index, edge_weight = build_population_graph(loso_ts_list, loso_graph_cache)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        n_parcels = loso_ts_list[0].shape[1] if loso_ts_list else config.N_PARCELS

        fold_ckpt_dir = os.path.join(checkpoint_dir, f"fold_excl-{test_id}")

        best_model_state, val_logits, val_labels, val_subject_ids, fold_val_auc, fold_hist = train_one_loso_fold(
            fold_label=f"excl-{test_id}",
            train_list=train_list,
            val_list=val_list,
            edge_index=edge_index,
            edge_weight=edge_weight,
            n_parcels=n_parcels,
            cache_dir=cache_dir,
            fold_ckpt_dir=fold_ckpt_dir,
            device=device,
            epochs=epochs,
            masker=masker,
            condition_filter=condition_filter,
            static_feat_map=static_feat_map,
            use_film=use_film,
        )

        if best_model_state is None:
            print(f"  [LOSO {fold_i+1}] Training failed for sub-{test_id} - skipping.")
            continue

        calibration = fit_subject_calibration(val_logits, val_labels, val_subject_ids)
        threshold = calibration["threshold"]

        # Inference on held-out test subject
        test_ds = fMRIWindowDataset(
            [test_subj], masker,
            use_sliding=False,
            cache_dir=cache_dir,
            condition_filter=condition_filter,
            augment=False,
            static_feat_map=static_feat_map,
        )
        if len(test_ds) == 0:
            print(f"  [LOSO {fold_i+1}] No windows for sub-{test_id} - skipping.")
            continue

        micro_bs = max(1, config.BATCH_SIZE // config.GRAD_ACCUM_STEPS)
        test_loader = DataLoader(
            test_ds, batch_size=micro_bs, shuffle=False,
            collate_fn=collate_fn, num_workers=4, pin_memory=True,
        )

        n_static = len(next(iter(static_feat_map.values()))) if static_feat_map else 0
        test_model = build_selected_model(
            edge_index, edge_weight, n_parcels=n_parcels,
            n_static_features=n_static, use_film=use_film,
        ).to(device)
        test_model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        test_logits, _ = get_raw_logits(test_model, test_loader, device)

        subj_prob, subj_mean_logit = apply_subject_calibration(calibration, test_logits)
        subj_pred = int(subj_prob >= threshold)
        n_wins = len(test_logits)
        correct = subj_pred == int(test_label)

        print(
            f"  sub-{test_id}: {n_wins} windows, "
            f"P(autistic)={subj_prob:.4f}, "
            f"threshold={threshold:.4f}, "
            f"pred={'autistic' if subj_pred == 1 else 'non-autistic'}, "
            f"true={'autistic' if test_label == 1 else 'non-autistic'} "
            f"[{'OK' if correct else 'MISS'}]"
        )

        # Fold best model saving
        torch.save({
            "test_subject": test_id,
            "test_label": test_label,
            "model_state": best_model_state,
            "val_auc": fold_val_auc,
            "n_parcels": n_parcels,
            "use_film": use_film,
            "n_static_features": n_static,
            "subject_threshold": threshold,
        }, os.path.join(fold_ckpt_dir, "best_model.pt"))

        loso_state["completed_folds"].append(test_id)
        loso_state["subject_probs"][test_id] = subj_prob
        loso_state["subject_labels"][test_id] = test_label
        loso_state["subject_preds"][test_id] = subj_pred
        loso_state["subject_thresholds"][test_id] = threshold
        loso_state["subject_mean_logits"][test_id] = subj_mean_logit
        loso_state["subject_window_counts"][test_id] = n_wins
        loso_state["fold_val_aucs"][test_id] = fold_val_auc
        loso_state["fold_histories"][test_id] = fold_hist
        loso_state["fold_calibration"][test_id] = {
            "threshold": threshold,
            "val_subject_ids": calibration["subject_ids"],
            "val_subject_probs": calibration["subject_probs"],
            "val_subject_labels": calibration["subject_labels"],
            "val_subject_window_counts": calibration["subject_window_counts"],
        }

        with open(loso_state_path, "w") as f:
            json.dump(loso_state, f, indent=2)
        append_training_progress(
            "loso",
            f"{checkpoint_dir} fold {fold_i+1}/{len(subject_list)} sub-{test_id} "
            f"prob={subj_prob:.4f} threshold={threshold:.4f} pred={subj_pred} true={test_label}",
        )
        del test_model, edge_index, edge_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final subject-level evaluation across all completed folds
    completed = loso_state["completed_folds"]
    if not completed:
        print("No LOSO folds completed.")
        return None

    all_probs = [loso_state["subject_probs"][s] for s in completed]
    all_labels = [loso_state["subject_labels"][s] for s in completed]
    all_preds = [
        int(loso_state.get("subject_preds", {}).get(
            s,
            int(loso_state["subject_probs"][s] >= loso_state.get("subject_thresholds", {}).get(s, 0.5)),
        ))
        for s in completed
    ]

    elapsed = time.time() - run_start
    h = int(elapsed // 3600)
    m_el = int((elapsed % 3600) // 60)
    s_el = int(elapsed % 60)

    print(f"\n{'='*30}")
    print(f"LOSO FINAL RESULTS ({len(completed)}/{len(subject_list)} folds completed)")
    print(f"{'='*30}")
    for sid, pred in zip(completed, all_preds):
        prob = loso_state["subject_probs"][sid]
        true_lbl = loso_state["subject_labels"][sid]
        threshold = loso_state.get("subject_thresholds", {}).get(sid, 0.5)
        match_str = "OK" if pred == true_lbl else "MISS"
        print(
            f"  sub-{sid}: P(autistic)={prob:.4f} threshold={threshold:.4f} "
            f"true={'autistic' if true_lbl==1 else 'non-autistic'} "
            f"pred={'autistic' if pred==1 else 'non-autistic'} [{match_str}]"
        )

    subj_metrics = {}
    if len(set(all_labels)) >= 2:
        subj_metrics = compute_metrics(all_labels, all_preds, all_probs)
        print(f"\n  Subject Accuracy : {subj_metrics['accuracy']:.4f}")
        print(f"  Subject AUC-ROC : {subj_metrics['auc']:.4f}")
        print(f"  Subject F1 (macro) : {subj_metrics['f1_macro']:.4f}")
        print(f"  Subject Sensitivity : {subj_metrics['sensitivity']:.4f}")
        print(f"  Subject Specificity : {subj_metrics['specificity']:.4f}")
    else:
        print("  Cannot compute AUC: only one class in completed folds.")

    # Save LOSO results JSON
    loso_results = {
        "n_completed": len(completed),
        "n_total": len(subject_list),
        "subject_probs": loso_state["subject_probs"],
        "subject_labels": loso_state["subject_labels"],
        "subject_preds": {sid: int(pred) for sid, pred in zip(completed, all_preds)},
        "subject_thresholds": loso_state.get("subject_thresholds", {}),
        "subject_mean_logits": loso_state.get("subject_mean_logits", {}),
        "subject_window_counts": loso_state["subject_window_counts"],
        "fold_val_aucs": loso_state["fold_val_aucs"],
        "fold_calibration": loso_state.get("fold_calibration", {}),
        "subject_metrics": subj_metrics,
        "use_film": use_film,
        "elapsed_time": f"{h}h {m_el}m {s_el}s",
    }
    results_path = os.path.join(checkpoint_dir, "loso_results.json")
    with open(results_path, "w") as f:
        json.dump(loso_results, f, indent=2)
    print(f"\nLOSO results saved to {results_path}")

    # Save per-fold epoch
    fold_histories = loso_state.get("fold_histories", {})
    if fold_histories:
        hist_path = os.path.join(checkpoint_dir, "loso_fold_histories.json")
        with open(hist_path, "w") as f:
            json.dump(fold_histories, f, indent=2)
        print(f"Fold histories saved to {hist_path}")

    # Generate LOSO figures
    try:
        from evaluate import plot_all_loso
        plot_all_loso(loso_results, fig_dir, fold_histories=fold_histories or None)
        print(f"LOSO figures saved to {fig_dir}/")
    except Exception as e:
        print(f"LOSO figures skipped: {e}")

    # training log
    if subj_metrics:
        if condition_filter is not None:
            loso_mode = f"loso_condition-{condition_filter+1}-{config.CONDITION_MAP.get(condition_filter+1,'')}"
        elif use_film:
            loso_mode = "loso"
        else:
            loso_mode = "loso_stgcn_only"
        write_training_log(
            mode=loso_mode,
            hyperparams={
                "lr": config.LR,
                "weight_decay": config.WEIGHT_DECAY,
                "dropout": config.DROPOUT,
                "window_trs": config.WINDOW_TRS,
                "window_sec": config.WINDOW_DURATION_SEC,
                "batch_size": config.BATCH_SIZE,
                "micro_batch": max(1, config.BATCH_SIZE // config.GRAD_ACCUM_STEPS),
                "grad_accum_steps": config.GRAD_ACCUM_STEPS,
                "lr_warmup": config.LR_WARMUP_EPOCHS,
                "mixup_alpha": config.MIXUP_ALPHA,
                "augment_noise_std": config.AUGMENT_NOISE_STD,
                "augment_parcel_drop": config.AUGMENT_PARCEL_DROP,
                "epochs": epochs,
                "patience": config.LOSO_PATIENCE,
                "k_folds": f"LOSO (N={len(subject_list)})",
            },
            dataset_info={
                "n_trainval_subjects": "N-1 per fold (inner train+val)",
                "n_test_subjects": "1 per fold (LOSO)",
            },
            results={
                "loso_subj_accuracy": subj_metrics["accuracy"],
                "loso_subj_auc": subj_metrics["auc"],
                "loso_subj_sensitivity": subj_metrics["sensitivity"],
                "loso_subj_specificity": subj_metrics["specificity"],
                "loso_subj_f1_macro": subj_metrics["f1_macro"],
                "loso_folds_completed": len(completed),
                "loso_folds_total": len(subject_list),
                "use_film": use_film,
            },
            elapsed_seconds=elapsed,
        )

    return loso_results


# Per-condition single-condition model training
def train_per_condition(
    bids_root=config.BIDS_ROOT,
    atlas_path=config.ATLAS_PATH,
    cache_dir=config.CACHE_DIR,
    checkpoint_dir=config.CHECKPOINT_DIR,
    debug=False,
    n_subjects=None,
    n_epochs=None,
    fresh=False,
    use_demographics=False,
):
    """
    Train one model per condition using only windows from that condition.
    """
    print("\n" + "=" * 30)
    print("PER-CONDITION TRAINING (one model per condition)")
    print("=" * 30)

    os.makedirs(checkpoint_dir, exist_ok=True)

    for cond_key, cond_name in config.CONDITION_MAP.items():
        cond_idx = cond_key - 1 # 0-based filter index

        print(f"\n{'='*30}")
        print(f"Condition {cond_key}/{len(config.CONDITION_MAP)}: {cond_name}")
        print(f"{'='*30}")

        train(
            bids_root=bids_root,
            atlas_path=atlas_path,
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
            debug=debug,
            n_subjects=n_subjects,
            n_epochs=n_epochs,
            fresh=fresh,
            condition_filter=cond_idx,
            use_film=False,
            use_demographics=use_demographics,
        )

        src = os.path.join(
            checkpoint_dir.rstrip("/") + f"_cond{cond_key}_stgcn_only",
            "best_model.pt",
        )
        dst = os.path.join(checkpoint_dir, f"best_model_cond{cond_key}_{cond_name}_stgcn_only.pt")
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  Best model for '{cond_name}' saved to {dst}")
        else:
            print(f"  No best_model.pt found for condition {cond_key} ({cond_name})")

    print(f"\nPer-condition training complete. Models saved to {checkpoint_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ST-GCN + FiLM fMRI training pipeline."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help=(
            "Small-scale test: 6 subjects, 5 epochs, no sliding windows. "
            "Outputs to checkpoints_debug/."
        ),
    )
    parser.add_argument(
        "--n-subjects", type=int, default=None, metavar="N",
        help="Limit to N subjects.",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=None, metavar="N",
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help=(
            "Start from scratch."
        ),
    )
    parser.add_argument(
        "--all-conditions", action="store_true",
        help=(
            "Run main training (all conditions) "
        ),
    )
    parser.add_argument(
        "--per-condition-only", action="store_true",
        help=(
            "Run only per-condition models. "
            "Use to resume interrupted per-condition training."
        ),
    )
    parser.add_argument(
        "--laughter-type", action="store_true",
        help=(
            "Classify laughter type (spontaneous=0 vs conversational=1)"
        ),
    )
    parser.add_argument(
        "--demographics", action="store_true",
        help=(
            "Include static demographic features (AQ, Age, Gender, FSIQ) after "
            "global pooling. Features are z-scored using training-set statistics."
        ),
    )
    parser.add_argument(
        "--stgcn-only", action="store_true",
        help=(
            "Train ST-GCN without FiLM conditioning."
        ),
    )
    parser.add_argument(
        "--loso", action="store_true",
        help=(
            "LOSO cross-validation. "
        ),
    )
    parser.add_argument(
        "--loso-epochs", type=int, default=None, metavar="N",
        help=(
            "Override epochs per LOSO fold"
        ),
    )
    parser.add_argument(
        "--loso-condition", type=int, default=None, metavar="C",
        choices=[1, 2, 3, 4, 5],
        help=(
            "Run LOSO using only windows from conditions. "
            "1=spontaneous laughter, 2=conversational laughter, 3=non-emotional sound, "
            "4=rest, 5=beep."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None, metavar="N",
        help=(
            "Override the random seed for the LOSO run."
        ),
    )
    parser.add_argument(
        "--loso-all-conditions", action="store_true",
        help=(
            "Run LOSO per-condition for all 5 conditions sequentially. "
        ),
    )
    args = parser.parse_args()

    use_demo = args.demographics

    if args.loso_all_conditions:
        for cond_idx in range(config.N_CONDITIONS):
            cond_name = config.CONDITION_MAP.get(cond_idx + 1, f"cond{cond_idx + 1}")
            print("\n" + "=" * 30)
            print(f"LOSO CONDITION {cond_idx + 1}/{config.N_CONDITIONS}: {cond_name}")
            print("=" * 30)
            train_loso(
                n_epochs=args.loso_epochs,
                fresh=args.fresh,
                condition_filter=cond_idx,
                use_demographics=use_demo,
                use_film=False,
            )
    elif args.loso:
        loso_use_film = (not args.stgcn_only) and args.loso_condition is None
        train_loso(
            n_epochs=args.loso_epochs,
            fresh=args.fresh,
            condition_filter=(args.loso_condition - 1) if args.loso_condition else None,
            use_demographics=use_demo,
            use_film=loso_use_film,
            seed=args.seed,
        )
    elif args.stgcn_only:
        # ST-GCN without FiLM
        train(
            debug=args.debug,
            n_subjects=args.n_subjects,
            n_epochs=args.n_epochs,
            fresh=args.fresh,
            use_film=False,
            use_demographics=use_demo,
        )
    elif args.laughter_type:
        # Laughter type classification: spontaneous vs conversational
        train(
            debug=args.debug,
            n_subjects=args.n_subjects,
            n_epochs=args.n_epochs,
            fresh=args.fresh,
            laughter_mode=True,
            use_demographics=use_demo,
        )
    elif args.per_condition_only:
        train_per_condition(
            debug=args.debug,
            n_subjects=args.n_subjects,
            n_epochs=args.n_epochs,
            fresh=args.fresh,
            use_demographics=use_demo,
        )
    elif args.all_conditions:
        # Step 1: main training on all conditions
        print("\n" + "=" * 30)
        print("STEP 1 OF 2: Main training on all conditions")
        print("=" * 30)
        train(
            debug=args.debug,
            n_subjects=args.n_subjects,
            n_epochs=args.n_epochs,
            fresh=args.fresh,
            use_demographics=use_demo,
        )

        # Step 2: per-condition training
        print("\n" + "=" * 30)
        print("STEP 2 OF 2: Per-condition training")
        print("=" * 30)
        train_per_condition(
            debug=args.debug,
            n_subjects=args.n_subjects,
            n_epochs=args.n_epochs,
            fresh=args.fresh,
            use_demographics=use_demo,
        )
    else:
        train(
            debug=args.debug,
            n_subjects=args.n_subjects,
            n_epochs=args.n_epochs,
            fresh=args.fresh,
            use_demographics=use_demo,
        )
