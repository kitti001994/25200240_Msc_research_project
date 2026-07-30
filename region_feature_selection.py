"""
  1. region_activation_analysis.py -> analysis tables.
  2. train.py selects top regions inside each training fold only.
"""

import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

import config
from dataset import (
    build_condition_regressors,
    get_run_timeseries,
    load_events,
    trial_locked_windows,
)


GROUP_AUTISTIC = 1
GROUP_NON_AUTISTIC = 0
GROUP_NAMES = {
    GROUP_AUTISTIC: "autistic",
    GROUP_NON_AUTISTIC: "non-autistic",
}

CONTRASTS = {
    "conversational_minus_spontaneous": (1, 0),
    "conversational_minus_non_emotional_sound": (1, 2),
    "spontaneous_minus_non_emotional_sound": (0, 2),
    "laughter_average_minus_non_emotional_sound": ((0, 1), 2),
    "conversational_minus_rest": (1, 3),
    "spontaneous_minus_rest": (0, 3),
}


def fdr_bh(p_values):
    """Benjamini-Hochberg FDR correction."""
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
    """Cohen's d for two independent groups: mean(x)-mean(y) over pooled SD."""
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


def hedges_g(x, y):
    """Hedges' g is Cohen's d corrected for small sample bias."""
    d = cohens_d(x, y)
    if not np.isfinite(d):
        return np.nan
    n = np.isfinite(x).sum() + np.isfinite(y).sum()
    if n <= 3:
        return np.nan
    correction = 1.0 - (3.0 / (4.0 * n - 9.0))
    return float(d * correction)


def bootstrap_mean_diff_ci(x, y, n_boot=2000, seed=42):
    """Bootstrap 95% CI for mean(x)-mean(y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan

    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs.append(np.mean(xb) - np.mean(yb))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def _outlier_penalty(values):
    """Penalty is larger when a row is driven by one very extreme subject."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 4:
        return 0.0
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-8
    robust_z = np.abs((vals - med) / (1.4826 * mad))
    return float(min(np.max(robust_z) / 10.0, 1.0))


def _effect_score(row):
    """Combined score for ranking regions."""
    effect = abs(row.get("hedges_g", np.nan))
    if not np.isfinite(effect):
        effect = abs(row.get("cohens_d", 0.0))
    q = row.get("q_fdr_effect", np.nan)
    q_bonus = 0.25 if np.isfinite(q) and q <= 0.10 else 0.0
    ci_low = row.get("diff_ci_low", np.nan)
    ci_high = row.get("diff_ci_high", np.nan)
    stable = (
        np.isfinite(ci_low)
        and np.isfinite(ci_high)
        and ((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))
    )
    stability_bonus = 0.5 if stable else 0.0
    var_ratio = row.get("variance_ratio_asd_over_nt", np.nan)
    variance_score = min(abs(np.log(var_ratio)), 2.0) * 0.25 if var_ratio > 0 else 0.0
    conn_score = abs(row.get("node_strength_diff_z", 0.0))
    penalty = row.get("outlier_penalty", 0.0)
    return float(effect + q_bonus + stability_bonus + variance_score + conn_score - penalty)


def _contrast_value(row, left, right):
    """Return one contrast value for a subject/parcel row from a condition pivot."""
    if isinstance(left, tuple):
        left_val = np.nanmean([row.get(config.CONDITION_MAP[i + 1], np.nan) for i in left])
    else:
        left_val = row.get(config.CONDITION_MAP[left + 1], np.nan)
    right_val = row.get(config.CONDITION_MAP[right + 1], np.nan)
    return left_val - right_val


def build_effect_value_table(beta_df):
    """
    Convert subject-condition betas into rows for both single conditions and contrasts.
    """
    rows = []
    base_cols = ["subject_id", "group", "group_code", "parcel_index"]

    for _, row in beta_df.iterrows():
        rows.append({
            **{c: row[c] for c in base_cols},
            "effect_type": "condition",
            "effect_name": row["condition"],
            "value": float(row["beta"]),
        })

    pivot = beta_df.pivot_table(
        index=base_cols,
        columns="condition",
        values="beta",
        aggfunc="mean",
    ).reset_index()

    for _, row in pivot.iterrows():
        for name, (left, right) in CONTRASTS.items():
            val = _contrast_value(row, left, right)
            if np.isfinite(val):
                rows.append({
                    **{c: row[c] for c in base_cols},
                    "effect_type": "contrast",
                    "effect_name": name,
                    "value": float(val),
                })

    return pd.DataFrame(rows)


def compute_effect_stats(beta_df, region_df=None, n_boot=2000, seed=42):
    """
    Compute group differences for each parcel and each condition/contrast.
    """
    effect_df = build_effect_value_table(beta_df)
    rows = []

    for (effect_type, effect_name, parcel_index), rdf in effect_df.groupby(
        ["effect_type", "effect_name", "parcel_index"]
    ):
        autistic = rdf[rdf["group_code"] == GROUP_AUTISTIC]["value"].to_numpy()
        non_autistic = rdf[rdf["group_code"] == GROUP_NON_AUTISTIC]["value"].to_numpy()
        if len(autistic) >= 2 and len(non_autistic) >= 2:
            t_stat, p_val = stats.ttest_ind(
                autistic, non_autistic, equal_var=False, nan_policy="omit"
            )
            lev_stat, lev_p = stats.levene(autistic, non_autistic, center="mean")
            bf_stat, bf_p = stats.levene(autistic, non_autistic, center="median")
            ci_low, ci_high = bootstrap_mean_diff_ci(
                autistic, non_autistic, n_boot=n_boot, seed=seed + int(parcel_index)
            )
        else:
            t_stat, p_val = np.nan, np.nan
            lev_stat, lev_p = np.nan, np.nan
            bf_stat, bf_p = np.nan, np.nan
            ci_low, ci_high = np.nan, np.nan

        var_asd = np.nanvar(autistic, ddof=1) if len(autistic) >= 2 else np.nan
        var_nt = np.nanvar(non_autistic, ddof=1) if len(non_autistic) >= 2 else np.nan
        var_ratio = var_asd / var_nt if np.isfinite(var_asd) and var_nt > 0 else np.nan

        rows.append({
            "effect_type": effect_type,
            "effect_name": effect_name,
            "parcel_index": int(parcel_index),
            "n_autistic": int(len(autistic)),
            "n_non_autistic": int(len(non_autistic)),
            "mean_autistic": float(np.nanmean(autistic)) if len(autistic) else np.nan,
            "mean_non_autistic": float(np.nanmean(non_autistic)) if len(non_autistic) else np.nan,
            "diff_autistic_minus_non_autistic": (
                float(np.nanmean(autistic) - np.nanmean(non_autistic))
                if len(autistic) and len(non_autistic) else np.nan
            ),
            "cohens_d": cohens_d(autistic, non_autistic),
            "hedges_g": hedges_g(autistic, non_autistic),
            "t_welch": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p_welch": float(p_val) if np.isfinite(p_val) else np.nan,
            "diff_ci_low": ci_low,
            "diff_ci_high": ci_high,
            "variance_autistic": float(var_asd) if np.isfinite(var_asd) else np.nan,
            "variance_non_autistic": float(var_nt) if np.isfinite(var_nt) else np.nan,
            "variance_ratio_asd_over_nt": float(var_ratio) if np.isfinite(var_ratio) else np.nan,
            "levene_stat": float(lev_stat) if np.isfinite(lev_stat) else np.nan,
            "levene_p": float(lev_p) if np.isfinite(lev_p) else np.nan,
            "brown_forsythe_stat": float(bf_stat) if np.isfinite(bf_stat) else np.nan,
            "brown_forsythe_p": float(bf_p) if np.isfinite(bf_p) else np.nan,
            "outlier_penalty": _outlier_penalty(np.concatenate([autistic, non_autistic])),
        })

    stats_df = pd.DataFrame(rows)
    stats_df["q_fdr_effect"] = np.nan
    stats_df["q_fdr_levene_effect"] = np.nan
    stats_df["q_fdr_brown_forsythe_effect"] = np.nan

    for effect_name in sorted(stats_df["effect_name"].unique()):
        mask = stats_df["effect_name"] == effect_name
        stats_df.loc[mask, "q_fdr_effect"] = fdr_bh(stats_df.loc[mask, "p_welch"])
        stats_df.loc[mask, "q_fdr_levene_effect"] = fdr_bh(stats_df.loc[mask, "levene_p"])
        stats_df.loc[mask, "q_fdr_brown_forsythe_effect"] = fdr_bh(
            stats_df.loc[mask, "brown_forsythe_p"]
        )

    stats_df["abs_effect_size"] = stats_df["hedges_g"].abs()
    stats_df["activation_score"] = stats_df.apply(_effect_score, axis=1)
    stats_df = stats_df.sort_values(
        ["effect_name", "abs_effect_size", "q_fdr_effect"],
        ascending=[True, False, True],
    )

    if region_df is not None:
        stats_df = stats_df.merge(region_df, on="parcel_index", how="left")
    return stats_df


def _corrcoef_safe(ts):
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 2 or ts.shape[0] < 4:
        return None
    ts = np.nan_to_num(ts, nan=0.0)
    corr = np.corrcoef(ts, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def collect_subject_connectivity(
    subjects,
    masker,
    bids_root,
    cache_dir,
    parcel_indices=None,
    condition_specific=False,
):
    """
    Build one correlation matrix per subject.
    """
    parcel_indices = None if parcel_indices is None else np.asarray(parcel_indices, dtype=int)
    full_mats = {}
    cond_mats = defaultdict(dict)
    labels = {}

    for sid, group in subjects:
        labels[sid] = int(group)
        run_corrs = []
        cond_ts = defaultdict(list)

        for run in range(1, 5):
            run_str = f"{run:02d}"
            ts = get_run_timeseries(sid, run_str, masker, cache_dir)
            if ts is None:
                continue
            if parcel_indices is not None:
                ts = ts[:, parcel_indices]
            corr = _corrcoef_safe(ts)
            if corr is not None:
                run_corrs.append(corr)

            if condition_specific:
                events_path = os.path.join(
                    bids_root, f"sub-{sid}", "func",
                    f"sub-{sid}_task-laughter_run-{run_str}_events.tsv",
                )
                if not os.path.exists(events_path):
                    continue
                regressors = build_condition_regressors(events_path, ts.shape[0], config.TR)
                events_df = load_events(events_path)
                wins, _, labs = trial_locked_windows(
                    ts, regressors, events_df,
                    tr=config.TR,
                    window_trs=config.WINDOW_TRS,
                    onset_offset_sec=config.ONSET_OFFSET_SEC,
                )
                for win, lab in zip(wins, labs):
                    # win is (N, W, 1). Convert to (W, N).
                    cond_ts[int(lab)].append(win[:, :, 0].T)

        if run_corrs:
            full_mats[sid] = np.nanmean(np.stack(run_corrs), axis=0)

        if condition_specific:
            for cond_idx, pieces in cond_ts.items():
                cat = np.concatenate(pieces, axis=0) if pieces else None
                corr = _corrcoef_safe(cat) if cat is not None and cat.shape[0] >= 8 else None
                if corr is not None:
                    cond_mats[cond_idx][sid] = corr

    return full_mats, cond_mats, labels


def fisher_z(corr):
    corr = np.clip(np.asarray(corr, dtype=float), -0.999999, 0.999999)
    return np.arctanh(corr)


def compute_edge_connectivity_stats(subject_mats, subject_labels, effect_name="full_run"):
    """
    Edge-wise ASD vs NT connectivity comparison -> Fisher-z correlations.
    """
    if not subject_mats:
        return pd.DataFrame(), {}, pd.DataFrame()

    sids = [sid for sid in subject_mats if sid in subject_labels]
    mats = np.stack([subject_mats[sid] for sid in sids])
    labels = np.array([subject_labels[sid] for sid in sids], dtype=int)
    z_mats = fisher_z(mats)
    n = mats.shape[1]

    asd = z_mats[labels == GROUP_AUTISTIC]
    nt = z_mats[labels == GROUP_NON_AUTISTIC]
    mean_asd_z = np.nanmean(asd, axis=0)
    mean_nt_z = np.nanmean(nt, axis=0)
    mean_asd = np.tanh(mean_asd_z)
    mean_nt = np.tanh(mean_nt_z)
    diff = mean_asd - mean_nt

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if asd.shape[0] >= 2 and nt.shape[0] >= 2:
                t_stat, p_val = stats.ttest_ind(
                    asd[:, i, j], nt[:, i, j],
                    equal_var=False, nan_policy="omit",
                )
            else:
                t_stat, p_val = np.nan, np.nan
            rows.append({
                "effect_name": effect_name,
                "source_parcel": i,
                "target_parcel": j,
                "mean_r_autistic": float(mean_asd[i, j]),
                "mean_r_non_autistic": float(mean_nt[i, j]),
                "diff_r_autistic_minus_non_autistic": float(diff[i, j]),
                "t_welch_fisher_z": float(t_stat) if np.isfinite(t_stat) else np.nan,
                "p_welch_fisher_z": float(p_val) if np.isfinite(p_val) else np.nan,
            })
    edge_df = pd.DataFrame(rows)
    edge_df["q_fdr_edge"] = fdr_bh(edge_df["p_welch_fisher_z"])

    matrices = {
        "autistic_mean_corr": mean_asd,
        "non_autistic_mean_corr": mean_nt,
        "autistic_minus_non_autistic_corr": diff,
    }
    node_df = compute_node_graph_summaries(mean_asd, mean_nt, effect_name=effect_name)
    return edge_df, matrices, node_df


def _graph_from_corr(corr, threshold=None):
    threshold = config.CORR_THRESHOLD if threshold is None else threshold
    adj = np.array(corr, dtype=float).copy()
    np.fill_diagonal(adj, 0.0)
    adj[np.abs(adj) < threshold] = 0.0
    graph = nx.from_numpy_array(np.abs(adj))
    return graph, adj


def _participation_coefficients(adj, communities):
    n = adj.shape[0]
    abs_adj = np.abs(adj)
    strengths = abs_adj.sum(axis=1)
    parts = np.zeros(n, dtype=float)
    community_index = {}
    for ci, comm in enumerate(communities):
        for node in comm:
            community_index[node] = ci
    for node in range(n):
        if strengths[node] <= 0:
            continue
        score = 0.0
        for ci in range(len(communities)):
            members = [m for m in communities[ci] if m != node]
            k_is = abs_adj[node, members].sum() if members else 0.0
            score += (k_is / strengths[node]) ** 2
        parts[node] = 1.0 - score
    return parts


def _single_group_node_summary(corr, group_name):
    graph, adj = _graph_from_corr(corr)
    n = adj.shape[0]
    pos = np.where(adj > 0, adj, 0.0)
    neg = np.where(adj < 0, adj, 0.0)
    degree = (adj != 0).sum(axis=1)
    strength = np.abs(adj).sum(axis=1)
    pos_strength = pos.sum(axis=1)
    neg_strength = np.abs(neg).sum(axis=1)
    clustering = nx.clustering(graph, weight="weight")
    global_eff = nx.global_efficiency(graph) if n > 1 else np.nan

    if graph.number_of_edges() > 0:
        try:
            communities = nx.algorithms.community.louvain_communities(
                graph, weight="weight", seed=config.SEED
            )
        except Exception:
            communities = nx.algorithms.community.greedy_modularity_communities(
                graph, weight="weight"
            )
        modularity = nx.algorithms.community.modularity(
            graph, communities, weight="weight"
        )
    else:
        communities = [set(range(n))]
        modularity = 0.0
    participation = _participation_coefficients(adj, [set(c) for c in communities])

    rows = []
    for node in range(n):
        rows.append({
            "parcel_index": node,
            f"node_strength_{group_name}": float(strength[node]),
            f"positive_strength_{group_name}": float(pos_strength[node]),
            f"negative_strength_{group_name}": float(neg_strength[node]),
            f"degree_{group_name}": int(degree[node]),
            f"clustering_{group_name}": float(clustering.get(node, 0.0)),
            f"participation_{group_name}": float(participation[node]),
            f"global_efficiency_{group_name}": float(global_eff),
            f"modularity_{group_name}": float(modularity),
        })
    return pd.DataFrame(rows)


def compute_node_graph_summaries(mean_asd_corr, mean_nt_corr, effect_name="full_run"):
    """Node strength, degree, clustering, participation, efficiency, and modularity."""
    asd_df = _single_group_node_summary(mean_asd_corr, "autistic")
    nt_df = _single_group_node_summary(mean_nt_corr, "non_autistic")
    out = asd_df.merge(nt_df, on="parcel_index", how="inner")
    out["effect_name"] = effect_name
    out["node_strength_diff"] = (
        out["node_strength_autistic"] - out["node_strength_non_autistic"]
    )
    sd = out["node_strength_diff"].std(ddof=1)
    out["node_strength_diff_z"] = (
        out["node_strength_diff"] / (sd + 1e-8) if np.isfinite(sd) else 0.0
    )
    out["positive_strength_diff"] = (
        out["positive_strength_autistic"] - out["positive_strength_non_autistic"]
    )
    out["negative_strength_diff"] = (
        out["negative_strength_autistic"] - out["negative_strength_non_autistic"]
    )
    out["degree_diff"] = out["degree_autistic"] - out["degree_non_autistic"]
    return out


def combine_region_scores(effect_stats_df, node_df=None):
    """Connectivity node difference to the activation/variance score."""
    out = effect_stats_df.copy()
    if node_df is not None and len(node_df):
        conn = node_df[node_df["effect_name"] == "full_run"] if "effect_name" in node_df else node_df
        conn = conn[["parcel_index", "node_strength_diff", "node_strength_diff_z"]]
        out = out.merge(conn, on="parcel_index", how="left")
    out["node_strength_diff"] = out.get("node_strength_diff", 0.0)
    out["node_strength_diff_z"] = out.get("node_strength_diff_z", 0.0)
    out[["node_strength_diff", "node_strength_diff_z"]] = out[
        ["node_strength_diff", "node_strength_diff_z"]
    ].fillna(0.0)
    out["region_score"] = out.apply(_effect_score, axis=1)
    return out.sort_values(["effect_name", "region_score"], ascending=[True, False])


def select_top_regions(effect_stats_df, top_k_per_effect=20, include_contrasts=True):
    """Top parcels per condition/contrast by region_score."""
    df = effect_stats_df.copy()
    if not include_contrasts:
        df = df[df["effect_type"] == "condition"]
    if "region_score" not in df:
        df["region_score"] = df.apply(_effect_score, axis=1)

    top_rows = []
    for effect_name, sub in df.groupby("effect_name"):
        ranked = sub.sort_values("region_score", ascending=False).head(top_k_per_effect)
        top_rows.append(ranked)
    top_df = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    parcels = sorted(top_df["parcel_index"].dropna().astype(int).unique().tolist())
    return parcels, top_df


def plot_top_region_distributions(
    effect_values_df,
    top_df,
    out_dir,
    max_plots=30,
    winsorized_label=None,
):
    """
    Top regions -> violin/box, histogram, and subject scatter plots.
    """
    fig_dir = os.path.join(out_dir, "figures", "top_region_distributions")
    os.makedirs(fig_dir, exist_ok=True)
    plot_rows = top_df.sort_values("region_score", ascending=False).head(max_plots)

    for row in plot_rows.itertuples():
        effect_name = row.effect_name
        parcel_index = int(row.parcel_index)
        sub = effect_values_df[
            (effect_values_df["effect_name"] == effect_name)
            & (effect_values_df["parcel_index"] == parcel_index)
        ].copy()
        if sub.empty:
            continue
        asd = sub[sub["group_code"] == GROUP_AUTISTIC]["value"].to_numpy()
        nt = sub[sub["group_code"] == GROUP_NON_AUTISTIC]["value"].to_numpy()
        if len(asd) < 2 or len(nt) < 2:
            continue

        title_bits = [effect_name.replace("_", " "), f"parcel {parcel_index}"]
        if winsorized_label:
            title_bits.append(winsorized_label)
        title = " | ".join(title_bits)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].violinplot([nt, asd], showmeans=True)
        axes[0].boxplot([nt, asd], widths=0.18)
        axes[0].set_xticks([1, 2])
        axes[0].set_xticklabels(["non-autistic", "autistic"], rotation=15)
        axes[0].set_ylabel("Beta / contrast value")
        axes[0].set_title("Violin + box")
        axes[0].grid(True, axis="y", alpha=0.3)

        axes[1].hist(nt, bins=12, alpha=0.45, label="non-autistic", density=True)
        axes[1].hist(asd, bins=12, alpha=0.45, label="autistic", density=True)
        x_min = float(np.nanmin(np.concatenate([nt, asd])))
        x_max = float(np.nanmax(np.concatenate([nt, asd])))
        if x_max > x_min:
            xs = np.linspace(x_min, x_max, 160)
            for vals, color in [(nt, "#16a34a"), (asd, "#ea580c")]:
                if len(vals) >= 3 and np.nanstd(vals) > 1e-8:
                    kde = stats.gaussian_kde(vals)
                    axes[1].plot(xs, kde(xs), color=color, lw=1.5)
        axes[1].set_title("Histogram + KDE")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, axis="y", alpha=0.3)

        rng = np.random.RandomState(0)
        for group_code, color, xpos in [
            (GROUP_NON_AUTISTIC, "#16a34a", 0),
            (GROUP_AUTISTIC, "#ea580c", 1),
        ]:
            gdf = sub[sub["group_code"] == group_code].copy()
            x = xpos + rng.uniform(-0.06, 0.06, size=len(gdf))
            axes[2].scatter(x, gdf["value"], color=color, alpha=0.8, s=22)
            vals = gdf["value"].to_numpy()
            med = np.nanmedian(vals)
            mad = np.nanmedian(np.abs(vals - med)) + 1e-8
            rz = np.abs((vals - med) / (1.4826 * mad))
            for xi, val, sid, z in zip(x, vals, gdf["subject_id"], rz):
                if z >= 3.5:
                    axes[2].text(xi, val, str(sid), fontsize=6, rotation=30)
        axes[2].set_xticks([0, 1])
        axes[2].set_xticklabels(["non-autistic", "autistic"], rotation=15)
        axes[2].set_title("Subject scatter with outlier labels")
        axes[2].grid(True, axis="y", alpha=0.3)

        fig.suptitle(title, fontsize=11, fontweight="semibold")
        fig.tight_layout()
        safe_effect = effect_name.replace("/", "_")
        fig.savefig(
            os.path.join(fig_dir, f"{safe_effect}_parcel-{parcel_index:03d}.png"),
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(fig)
