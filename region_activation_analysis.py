"""
Brainnetome regional activation comparison.
"""

import argparse
import glob
import os
import tempfile
import warnings

_cache_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
_mpl_cache = os.path.join(_cache_root, "matplotlib-cache")
try:
    os.makedirs(_mpl_cache, exist_ok=True)
except PermissionError:
    _mpl_cache = os.path.abspath("./matplotlib-cache")
    os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)
os.environ.setdefault("XDG_CACHE_HOME", _cache_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
from nilearn.maskers import NiftiLabelsMasker
from scipy import stats

import config
from dataset import build_condition_regressors, get_run_timeseries
from region_feature_selection import (
    build_effect_value_table,
    collect_subject_connectivity,
    combine_region_scores,
    compute_edge_connectivity_stats,
    compute_effect_stats,
    plot_top_region_distributions,
    select_top_regions,
)


GROUP_AUTISTIC = 1
GROUP_NON_AUTISTIC = 0
GROUP_NAMES = {
    GROUP_AUTISTIC: "autistic",
    GROUP_NON_AUTISTIC: "non-autistic",
}


def load_subject_groups(bids_root):
    participants_path = os.path.join(bids_root, "participants.tsv")
    if not os.path.exists(participants_path):
        raise FileNotFoundError(f"participants.tsv not found: {participants_path}")

    df = pd.read_csv(participants_path, sep="\t")
    required = {"participant_id", config.GROUP_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"participants.tsv missing columns: {sorted(missing)}")

    subjects = []
    for _, row in df.iterrows():
        sid = str(row["participant_id"]).replace("sub-", "")
        group = row[config.GROUP_COLUMN]
        if group == config.ASD_LABEL:
            subjects.append((sid, GROUP_AUTISTIC))
        elif group == config.NT_LABEL:
            subjects.append((sid, GROUP_NON_AUTISTIC))
    return subjects


def atlas_region_table(atlas_path, parcel_atlas_labels, labels_path=None):
    """
    Build the parcel_index -> atlas_label coordinate table.
    """
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(int)

    names = {}
    if labels_path:
        label_df = pd.read_csv(labels_path, sep=None, engine="python")
        lower = {c.lower(): c for c in label_df.columns}
        label_col = lower.get("label") or lower.get("index") or lower.get("id")
        name_col = lower.get("name") or lower.get("region") or lower.get("label_name")
        if label_col and name_col:
            names = {
                int(r[label_col]): str(r[name_col])
                for _, r in label_df.iterrows()
                if pd.notna(r[label_col])
            }

    rows = []
    for parcel_index, atlas_label in enumerate(parcel_atlas_labels):
        vox = np.argwhere(atlas_data == int(atlas_label))
        if vox.size:
            ijk = vox.mean(axis=0)
            xyz = nib.affines.apply_affine(atlas_img.affine, ijk)
        else:
            xyz = np.array([np.nan, np.nan, np.nan])
        rows.append({
            "parcel_index": parcel_index,
            "atlas_label": int(atlas_label),
            "region_name": names.get(int(atlas_label), f"Brainnetome_{int(atlas_label):03d}"),
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
        })

    native_labels = sorted(int(L) for L in (set(np.unique(atlas_data)) - {0}))
    dropped = sorted(set(native_labels) - set(int(L) for L in parcel_atlas_labels))
    if dropped:
        print(
            f"Native atlas has {len(native_labels)} non-zero labels but the "
            f"masker only retained {len(parcel_atlas_labels)} after resampling. "
            f"Dropped labels (excluded from analysis): {dropped}."
        )

    return pd.DataFrame(rows), atlas_img, atlas_data


def fdr_bh(p_values):
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    q_valid = np.empty_like(pv)
    q_valid[order] = adjusted
    q[valid] = q_valid
    return q


def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled_var = (
        ((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1))
        / max(len(x) + len(y) - 2, 1)
    )
    pooled_sd = np.sqrt(max(pooled_var, 1e-12))
    return float((np.mean(x) - np.mean(y)) / pooled_sd)


def estimate_run_betas(ts, events_path):
    X = build_condition_regressors(events_path, ts.shape[0], config.TR)
    if X.size == 0 or np.allclose(X, 0):
        return None

    # Include intercept
    design = np.column_stack([X, np.ones(ts.shape[0], dtype=np.float32)])

    # Identify condition columns whose regressor is effectively zero variance
    col_std = X.std(axis=0)
    valid_cond = col_std > 1e-8

    if not np.any(valid_cond):
        return None

    keep_design_cols = np.concatenate([valid_cond, [True]])  # always keep intercept
    fit_design = design[:, keep_design_cols]

    # near-singular components are truncated
    betas_fit, *_ = np.linalg.lstsq(fit_design, ts, rcond=None)

    betas = np.full((config.N_CONDITIONS, ts.shape[1]), np.nan, dtype=np.float64)
    fit_idx = 0
    for ci in range(config.N_CONDITIONS):
        if valid_cond[ci]:
            betas[ci, :] = betas_fit[fit_idx, :]
            fit_idx += 1

    return betas


def _find_reference_bold(subjects, fmriprep_dir):
    """Return any preprocessed BOLD file path, or None if none can be found."""
    for sid, _ in subjects:
        for run in range(1, 7):
            run_str = f"{run:02d}"
            candidates = [
                os.path.join(
                    fmriprep_dir, f"sub-{sid}", "func",
                    f"sub-{sid}_task-laughter_run-{run_str}"
                    "_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
                ),
                os.path.join(
                    fmriprep_dir, f"sub-{sid}", "func",
                    f"sub-{sid}_task-laughter_run-{run_str}"
                    "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                ),
            ]
            for p in candidates:
                hits = glob.glob(p)
                if hits:
                    return hits[0]
    return None


def collect_betas(subjects, bids_root, cache_dir, atlas_path):
    masker = NiftiLabelsMasker(
        labels_img=atlas_path,
        standardize=True,
        detrend=False,
        t_r=config.TR,
        memory="nilearn_cache",
        memory_level=1,
        verbose=0,
    )
    ref_bold = _find_reference_bold(subjects, config.FMRIPREP_DIR)
    parcel_atlas_labels = None
    if ref_bold is not None:
        try:
            _ = masker.fit_transform(ref_bold)
            parcel_atlas_labels = [int(L) for L in masker.labels_ if int(L) != 0]
            print(
                f"Masker probed with reference BOLD: {os.path.basename(ref_bold)} "
                f"(retained {len(parcel_atlas_labels)} parcels after resampling)."
            )
        except Exception as e:
            print(f"Could not probe masker on reference BOLD: {e}")

    rows = []
    n_regions = None
    for sid, group in subjects:
        subject_condition_betas = {ci: [] for ci in range(config.N_CONDITIONS)}
        for run in range(1, 5):
            run_str = f"{run:02d}"
            events_path = os.path.join(
                bids_root,
                f"sub-{sid}",
                "func",
                f"sub-{sid}_task-laughter_run-{run_str}_events.tsv",
            )
            if not os.path.exists(events_path):
                continue

            ts = get_run_timeseries(sid, run_str, masker, cache_dir)
            if ts is None:
                continue
            ts = np.nan_to_num(ts, nan=0.0)
            n_regions = ts.shape[1] if n_regions is None else n_regions

            betas = estimate_run_betas(ts, events_path)
            if betas is None:
                continue

            for ci in range(config.N_CONDITIONS):
                if np.any(np.isfinite(betas[ci])) and not np.allclose(betas[ci], 0):
                    subject_condition_betas[ci].append(betas[ci])

        for ci, beta_list in subject_condition_betas.items():
            if not beta_list:
                continue
            subj_beta = np.mean(beta_list, axis=0)
            for parcel_index, beta in enumerate(subj_beta):
                rows.append({
                    "subject_id": sid,
                    "group": GROUP_NAMES[group],
                    "group_code": group,
                    "condition_index": ci,
                    "condition": config.CONDITION_MAP[ci + 1],
                    "parcel_index": parcel_index,
                    "beta": float(beta),
                })

        print(f"Processed sub-{sid}: {sum(len(v) for v in subject_condition_betas.values())} condition-run beta vectors")

    if not rows:
        raise RuntimeError("No condition betas were estimated. Check BIDS paths, cache, and atlas.")

    if parcel_atlas_labels is None and getattr(masker, "labels_", None) is not None:
        parcel_atlas_labels = [int(L) for L in masker.labels_ if int(L) != 0]

    if parcel_atlas_labels is not None and len(parcel_atlas_labels) != int(n_regions):
        print(
            f"masker labels list length ({len(parcel_atlas_labels)}) "
            f"does not match time-series parcel count ({n_regions}). "
            "Falling back to native-atlas inference is no longer used; "
            "verify the BOLD reference and atlas paths."
        )

    return pd.DataFrame(rows), int(n_regions), parcel_atlas_labels


def save_outlier_report(beta_df, out_dir, prefix, top_n=20):
    """
    Save and print subject-level max absolute beta rankings.
    """
    report_df = beta_df.copy()
    report_df["abs_beta"] = report_df["beta"].abs()

    max_df = (
        report_df
        .groupby(["condition", "subject_id", "group"], as_index=False)["abs_beta"]
        .max()
        .sort_values(["condition", "abs_beta"], ascending=[True, False])
    )
    out_path = os.path.join(out_dir, f"{prefix}_condition_subject_abs_beta_max.csv")
    max_df.to_csv(out_path, index=False)

    beep_top = (
        max_df[max_df["condition"] == "beep"]
        .sort_values("abs_beta", ascending=False)
        .head(top_n)
    )
    print(f"\nTop {top_n} beep abs(beta) outlier check [{prefix}]:")
    if beep_top.empty:
        print("  No beep rows found.")
    else:
        print(beep_top[["subject_id", "group", "abs_beta"]].to_string(index=False))
    print(f"Saved outlier report: {out_path}")
    return max_df


def exclude_subjects(beta_df, subject_ids):
    subject_ids = [sid for sid in subject_ids if sid]
    if not subject_ids:
        return beta_df
    before = len(beta_df)
    filtered = beta_df[~beta_df["subject_id"].isin(subject_ids)].copy()
    removed = before - len(filtered)
    print(
        f"\nExcluded subjects from analysis: {', '.join(subject_ids)} "
        f"({removed} parcel-condition rows removed)."
    )
    return filtered


def winsorize_abs_beta(beta_df, percentile):
    """
    For each condition and Brainnetome parcel, cap beta to +/- the requested
    percentile of abs(beta) across participants.
    """
    if percentile is None:
        return beta_df
    if percentile <= 0 or percentile > 100:
        raise ValueError("--winsorize-abs-beta-pct must be in (0, 100].")

    out = beta_df.copy()
    capped = 0
    for _, idx in out.groupby(["condition_index", "parcel_index"]).groups.items():
        vals = out.loc[idx, "beta"].to_numpy(dtype=float)
        cap = np.nanpercentile(np.abs(vals), percentile)
        if not np.isfinite(cap):
            continue
        new_vals = np.clip(vals, -cap, cap)
        capped += int(np.sum(vals != new_vals))
        out.loc[idx, "beta"] = new_vals

    print(
        f"\nWinsorised beta values at abs(beta) percentile {percentile:g} "
        f"within each condition x parcel ({capped} values capped)."
    )
    return out


def compute_region_stats(beta_df, region_df):
    stats_df = compute_effect_stats(beta_df, region_df=region_df)

    # Backward-compatible aliases for existing map/plot functions.
    stats_df["condition"] = np.where(
        stats_df["effect_type"] == "condition",
        stats_df["effect_name"],
        np.nan,
    )
    cond_to_idx = {name: idx - 1 for idx, name in config.CONDITION_MAP.items()}
    stats_df["condition_index"] = stats_df["condition"].map(cond_to_idx)
    stats_df["t"] = stats_df["t_welch"]
    stats_df["p_uncorrected"] = stats_df["p_welch"]
    stats_df["q_fdr_condition"] = stats_df["q_fdr_effect"]
    return stats_df


def parcel_values_to_img(values, atlas_img, atlas_data, region_df):
    stat_data = np.zeros(atlas_data.shape, dtype=np.float32)
    for _, row in region_df.iterrows():
        parcel_index = int(row["parcel_index"])
        if parcel_index >= len(values):
            continue
        atlas_label = int(row["atlas_label"])
        stat_data[atlas_data == atlas_label] = float(values[parcel_index])
    return nib.Nifti1Image(stat_data, atlas_img.affine, atlas_img.header)


def save_maps_and_glass_brains(stats_df, region_df, atlas_img, atlas_data, out_dir):
    maps_dir = os.path.join(out_dir, "maps")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    value_cols = [
        ("mean_autistic", "autistic_mean", "Autistic mean activation"),
        ("mean_non_autistic", "non_autistic_mean", "Non-autistic mean activation"),
        ("diff_autistic_minus_non_autistic", "autistic_minus_non_autistic", "Autistic minus non-autistic"),
        ("cohens_d", "cohend", "Cohen d"),
    ]

    for _, cond_name in sorted(config.CONDITION_MAP.items()):
        cdf = stats_df[stats_df["condition"] == cond_name].sort_values("parcel_index")
        for col, suffix, title_prefix in value_cols:
            values = np.full(len(region_df), np.nan, dtype=np.float32)
            for _, row in cdf.iterrows():
                pi = int(row["parcel_index"])
                if pi < len(values):
                    values[pi] = row[col]
            img = parcel_values_to_img(values, atlas_img, atlas_data, region_df)
            base = f"{cond_name}_{suffix}"
            nii_path = os.path.join(maps_dir, f"{base}.nii.gz")
            nib.save(img, nii_path)

            fig = plt.figure(figsize=(12, 3.2))
            display = plotting.plot_glass_brain(
                img,
                display_mode="lyrz",
                figure=fig,
                title=f"{title_prefix}: {cond_name.replace('_', ' ')}",
                colorbar=True,
                plot_abs=False,
                threshold="auto",
            )
            fig.savefig(os.path.join(fig_dir, f"{base}.png"), dpi=300, bbox_inches="tight")
            display.close()
            plt.close(fig)


def plot_top_regions(stats_df, out_dir, top_n):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    top_rows = []
    for _, cond_name in sorted(config.CONDITION_MAP.items()):
        cdf = stats_df[stats_df["condition"] == cond_name].copy()
        cdf = cdf.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False).head(top_n)
        top_rows.append(cdf)

        labels = [
            f"{r.region_name}\n#{int(r.atlas_label)}"
            for r in cdf.itertuples()
        ]
        values = cdf["diff_autistic_minus_non_autistic"].to_numpy()
        colors = ["#ea580c" if v >= 0 else "#16a34a" for v in values]

        fig, ax = plt.subplots(figsize=(max(8, top_n * 0.9), 5))
        ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.6)
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("Mean beta difference")
        ax.set_title(
            f"Top Brainnetome regions: {cond_name.replace('_', ' ')}\n"
            "Positive = higher activation in autistic participants"
        )
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            os.path.join(fig_dir, f"top_regions_condition_{cond_name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    top_df = pd.concat(top_rows, ignore_index=True)
    top_df.to_csv(os.path.join(out_dir, "top_regions_per_condition.csv"), index=False)

    union = top_df["parcel_index"].drop_duplicates().tolist()
    heat = []
    y_labels = []
    for pi in union:
        region = stats_df[stats_df["parcel_index"] == pi].iloc[0]
        y_labels.append(f"{region['region_name']} #{int(region['atlas_label'])}")
        heat.append([
            stats_df[
                (stats_df["parcel_index"] == pi)
                & (stats_df["condition"] == cond_name)
            ]["cohens_d"].iloc[0]
            for _, cond_name in sorted(config.CONDITION_MAP.items())
        ])

    fig_h = max(5, 0.28 * len(y_labels))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    im = ax.imshow(np.asarray(heat), aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_xticks(range(config.N_CONDITIONS))
    ax.set_xticklabels(
        [name.replace("_", "\n") for _, name in sorted(config.CONDITION_MAP.items())],
        fontsize=8,
    )
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.set_title("Cohen d by condition for top Brainnetome regions")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Cohen d: autistic minus non-autistic")
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "cohend_heatmap_top_regions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Brainnetome activation between autistic and non-autistic participants."
    )
    parser.add_argument("--bids-root", default=config.BIDS_ROOT)
    parser.add_argument("--cache-dir", default=config.CACHE_DIR)
    parser.add_argument("--atlas-path", default=config.ATLAS_PATH)
    parser.add_argument("--labels-path", default=None, help="Optional Brainnetome label CSV/TSV.")
    parser.add_argument("--out-dir", default="./region_activation_analysis")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument(
        "--top-k-per-effect",
        type=int,
        default=20,
        help=(
            "Number of parcels selected for each condition/contrast in "
            "top_regions_per_effect.csv."
        ),
    )
    parser.add_argument(
        "--condition-connectivity",
        action="store_true",
        help=(
            "Condition-specific connectivity from concatenated "
            "trial windows."
        ),
    )
    parser.add_argument(
        "--exclude-subject",
        action="append",
        default=["220530WH"],
        help=(
            "Exclude -> extreme beta outlier."
        ),
    )
    parser.add_argument(
        "--include-220530WH",
        action="store_true",
        help="Disable the 220530WH exclusion.",
    )
    parser.add_argument(
        "--winsorize-abs-beta-pct",
        type=float,
        default=None,
        help=(
            "Computed within each condition x parcel."
        ),
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(args.out_dir, exist_ok=True)

    subjects = load_subject_groups(args.bids_root)
    n_autistic = sum(label == GROUP_AUTISTIC for _, label in subjects)
    n_non_autistic = sum(label == GROUP_NON_AUTISTIC for _, label in subjects)
    print(
        f"Found {len(subjects)} participants "
        f"({n_autistic} autistic / {n_non_autistic} non-autistic)."
    )

    beta_df, n_regions, parcel_atlas_labels = collect_betas(
        subjects=subjects,
        bids_root=args.bids_root,
        cache_dir=args.cache_dir,
        atlas_path=args.atlas_path,
    )
    beta_df.to_csv(os.path.join(args.out_dir, "raw_subject_condition_betas.csv"), index=False)
    save_outlier_report(beta_df, args.out_dir, prefix="raw")

    exclude_ids = list(args.exclude_subject or [])
    if args.include_220530WH:
        exclude_ids = [sid for sid in exclude_ids if sid != "220530WH"]
    beta_df = exclude_subjects(beta_df, exclude_ids)
    subjects = [(sid, group) for sid, group in subjects if sid not in set(exclude_ids)]
    pre_winsor_beta_df = beta_df.copy()
    beta_df = winsorize_abs_beta(beta_df, args.winsorize_abs_beta_pct)
    save_outlier_report(beta_df, args.out_dir, prefix="analysis")

    beta_df.to_csv(os.path.join(args.out_dir, "subject_condition_betas.csv"), index=False)

    if parcel_atlas_labels is None:
        raise RuntimeError(
            "Could not determine which atlas labels the masker retained. "
            "No reference BOLD was found to fit the masker. "
            "FMRIPREP_DIR and that at least one preprocessed BOLD exists."
        )
    if len(parcel_atlas_labels) != n_regions:
        raise RuntimeError(
            f"masker labels list length ({len(parcel_atlas_labels)}) does not "
            f"match time-series parcel count ({n_regions}); refusing to "
            "produce a misaligned region table."
        )

    region_df, atlas_img, atlas_data = atlas_region_table(
        args.atlas_path,
        parcel_atlas_labels=parcel_atlas_labels,
        labels_path=args.labels_path,
    )
    region_df.to_csv(os.path.join(args.out_dir, "brainnetome_region_table.csv"), index=False)

    stats_df = compute_region_stats(beta_df, region_df)
    stats_df.to_csv(os.path.join(args.out_dir, "region_effect_stats.csv"), index=False)

    condition_stats_df = stats_df[stats_df["effect_type"] == "condition"].copy()
    condition_stats_df.to_csv(
        os.path.join(args.out_dir, "region_condition_stats.csv"),
        index=False,
    )

    print("\nComputing full-run group connectivity and graph summaries.")
    masker = NiftiLabelsMasker(
        labels_img=args.atlas_path,
        standardize=True,
        detrend=False,
        t_r=config.TR,
        memory="nilearn_cache",
        memory_level=1,
        verbose=0,
    )
    subject_mats, condition_mats, subject_labels = collect_subject_connectivity(
        subjects=subjects,
        masker=masker,
        bids_root=args.bids_root,
        cache_dir=args.cache_dir,
        condition_specific=args.condition_connectivity,
    )
    edge_df, matrices, node_df = compute_edge_connectivity_stats(
        subject_mats, subject_labels, effect_name="full_run"
    )
    if len(edge_df):
        edge_df.to_csv(
            os.path.join(args.out_dir, "region_connectivity_edges.csv"),
            index=False,
        )
    if len(node_df):
        node_df = node_df.merge(region_df, on="parcel_index", how="left")
        node_df.to_csv(
            os.path.join(args.out_dir, "region_connectivity_node_summaries.csv"),
            index=False,
        )
    if matrices:
        np.savez_compressed(
            os.path.join(args.out_dir, "group_connectivity_matrices.npz"),
            **matrices,
        )

    if args.condition_connectivity and condition_mats:
        cond_edge_rows = []
        cond_node_rows = []
        conn_dir = os.path.join(args.out_dir, "condition_connectivity_matrices")
        os.makedirs(conn_dir, exist_ok=True)
        for cond_idx, mats in sorted(condition_mats.items()):
            cond_name = config.CONDITION_MAP.get(cond_idx + 1, f"cond{cond_idx + 1}")
            c_edge, c_matrices, c_node = compute_edge_connectivity_stats(
                mats, subject_labels, effect_name=cond_name
            )
            if len(c_edge):
                cond_edge_rows.append(c_edge)
            if len(c_node):
                c_node = c_node.merge(region_df, on="parcel_index", how="left")
                cond_node_rows.append(c_node)
            if c_matrices:
                np.savez_compressed(
                    os.path.join(conn_dir, f"{cond_name}_connectivity_matrices.npz"),
                    **c_matrices,
                )
        if cond_edge_rows:
            pd.concat(cond_edge_rows, ignore_index=True).to_csv(
                os.path.join(args.out_dir, "condition_connectivity_edges.csv"),
                index=False,
            )
        if cond_node_rows:
            pd.concat(cond_node_rows, ignore_index=True).to_csv(
                os.path.join(args.out_dir, "condition_connectivity_node_summaries.csv"),
                index=False,
            )

    scored_df = combine_region_scores(stats_df, node_df=node_df if len(node_df) else None)
    scored_df.to_csv(os.path.join(args.out_dir, "region_scores.csv"), index=False)

    if args.winsorize_abs_beta_pct:
        raw_stats_df = compute_region_stats(pre_winsor_beta_df, region_df)
        raw_scored_df = combine_region_scores(
            raw_stats_df,
            node_df=node_df if len(node_df) else None,
        )
        compare_cols = [
            "effect_type", "effect_name", "parcel_index",
            "hedges_g", "q_fdr_effect", "region_score",
            "outlier_penalty",
        ]
        raw_compare = raw_scored_df[compare_cols].rename(columns={
            "hedges_g": "raw_hedges_g",
            "q_fdr_effect": "raw_q_fdr_effect",
            "region_score": "raw_region_score",
            "outlier_penalty": "raw_outlier_penalty",
        })
        win_compare = scored_df[compare_cols].rename(columns={
            "hedges_g": "winsorized_hedges_g",
            "q_fdr_effect": "winsorized_q_fdr_effect",
            "region_score": "winsorized_region_score",
            "outlier_penalty": "winsorized_outlier_penalty",
        })
        raw_compare.merge(
            win_compare,
            on=["effect_type", "effect_name", "parcel_index"],
            how="inner",
        ).to_csv(
            os.path.join(args.out_dir, "raw_vs_winsorized_region_scores.csv"),
            index=False,
        )

    _, top_effect_df = select_top_regions(
        scored_df,
        top_k_per_effect=args.top_k_per_effect,
        include_contrasts=True,
    )
    top_effect_df.to_csv(
        os.path.join(args.out_dir, "top_regions_per_effect.csv"),
        index=False,
    )

    effect_values_df = build_effect_value_table(beta_df)
    plot_top_region_distributions(
        effect_values_df,
        top_effect_df,
        args.out_dir,
        max_plots=min(30, args.top_k_per_effect * 2),
        winsorized_label=(
            f"winsorized {args.winsorize_abs_beta_pct:g}%"
            if args.winsorize_abs_beta_pct else None
        ),
    )

    plot_top_regions(condition_stats_df, args.out_dir, top_n=args.top_n)
    save_maps_and_glass_brains(condition_stats_df, region_df, atlas_img, atlas_data, args.out_dir)

    print(f"Region activation analysis saved to {args.out_dir}")
    print("Primary tables: region_effect_stats.csv, region_scores.csv")


if __name__ == "__main__":
    main()
