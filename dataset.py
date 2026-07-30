import os
import glob
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from nilearn.maskers import NiftiLabelsMasker
from nilearn.glm.first_level import make_first_level_design_matrix

import config

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=".*labels were removed.*",
    category=UserWarning,
    module="nilearn.maskers.nifti_labels_masker",
)


# load events TSV
def load_events(events_path: str) -> pd.DataFrame:
    df = pd.read_csv(events_path, sep="\t")

    if "condition" in df.columns:
        df["condition"] = pd.to_numeric(df["condition"], errors="coerce")
    elif "trial_type" in df.columns:
        def _trial_type_to_cond(tt):
            s = str(tt).strip()
            if s.startswith("cond"):
                num_str = s[4:].split(".")[0]
                if num_str.isdigit():
                    return int(num_str)
            return np.nan
        df["condition"] = df["trial_type"].apply(_trial_type_to_cond)
    else:
        raise ValueError(
            f"Events file {events_path} has neither 'condition' nor "
            f"'trial_type' column. Columns: {list(df.columns)}"
        )

    # Drop rows with unkown condition
    df = df[df["condition"].notna()].copy()
    if df.empty:
        return df
    df["condition"] = df["condition"].astype(int)

    # Parse numeric onset/duration columns
    df["onset_num"] = pd.to_numeric(df["onset"], errors="coerce")
    df["duration_num"] = pd.to_numeric(df["duration"], errors="coerce")

    # return empty if no numeric onset
    if not df["onset_num"].notna().any():
        return pd.DataFrame()

    # Reconstruct onsets for rest/beep rows from trial ordering
    # onset_rest = onset_prev + duration_prev + isi_prev
    # duration_rest = iti
    if df["onset_num"].isna().any() and "trial_number" in df.columns:
        df_s = df.sort_values("trial_number").reset_index(drop=True)
        isi_series = pd.to_numeric(
            df_s["isi"] if "isi" in df_s.columns else pd.Series([0.0] * len(df_s)),
            errors="coerce",
        ).fillna(0.0)
        df_s["isi_num"] = isi_series.values

        last_end = None # tracks end-time of the recent active trial
        for i, row in df_s.iterrows():
            if pd.notna(row["onset_num"]):
                # end_of_active = onset + duration + isi
                last_end = row["onset_num"] + row["duration_num"] + row["isi_num"]
            elif last_end is not None:
                # iti column as the duration of the rest/beep event
                iti = pd.to_numeric(
                    row["iti"] if "iti" in df_s.columns else np.nan,
                    errors="coerce",
                )
                dur = float(iti) if pd.notna(iti) else config.TR # fallback = 1 TR
                df_s.at[i, "onset_num"] = last_end
                df_s.at[i, "duration_num"] = dur
                last_end = last_end + dur
        df = df_s

    # Keep only valid onsets
    df_valid = df[df["onset_num"].notna()].copy()
    df_valid["onset"] = df_valid["onset_num"].astype(float)
    df_valid["duration"] = df_valid["duration_num"].astype(float)

    return df_valid.reset_index(drop=True)


def build_condition_regressors(
    events_path: str,
    n_timepoints: int,
    tr: float,):
    """
    Convolve condition boxcars with the Glover HRF and return a
    (T, N_CONDITIONS) array aligned to scanner TRs.
    """
    df = load_events(events_path)

    # nilearn columns: onset, duration, trial_type
    df_nilearn = df.copy()
    df_nilearn["trial_type"] = df_nilearn["condition"].map(
        lambda c: config.CONDITION_MAP.get(c, f"cond{c}")
    )

    frame_times = np.arange(n_timepoints) * tr

    try:
        dm = make_first_level_design_matrix(
            frame_times,
            df_nilearn[["onset", "duration", "trial_type"]],
            hrf_model="glover",
            drift_model=None, # done by fMRIPrep
        )
    except Exception as e:
        print(f" Design matrix failed for {events_path}: {e}")
        return np.zeros((n_timepoints, config.N_CONDITIONS))

    cond_names = [config.CONDITION_MAP[k] for k in sorted(config.CONDITION_MAP)]
    regressors = np.zeros((n_timepoints, config.N_CONDITIONS))
    for i, name in enumerate(cond_names):
        if name in dm.columns:
            regressors[:, i] = dm[name].values

    return regressors.astype(np.float32)


# Atlas masker
def build_masker(atlas_path: str):
    """
    NiftiLabelsMasker for the Brainnetome 274 atlas.
    standardize=True -> z-score each parcel's time-series.
    detrend=False -> fMRIPrep already detrended the BOLD signal.
    """
    masker = NiftiLabelsMasker(
        labels_img=atlas_path,
        standardize=True,
        detrend=False,
        t_r=config.TR,
        memory="nilearn_cache",
        memory_level=1,
        verbose=0,
    )
    return masker


# Per-run time-series cache
def get_run_timeseries(
    subject_id: str,
    run_str: str,
    masker: NiftiLabelsMasker,
    cache_dir: str = None,
):
    """
    Returns the (T, N_parcels) BOLD time-series for one run.
    Returns None if the BOLD file cannot be located.
    """
    # Locate BOLD file
    bold_pattern = os.path.join(
        config.FMRIPREP_DIR, f"sub-{subject_id}", "func",
        f"sub-{subject_id}_task-laughter_run-{run_str}"
        "_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
    )
    bold_files = glob.glob(bold_pattern)
    if not bold_files:
        bold_pattern = os.path.join(
            config.FMRIPREP_DIR, f"sub-{subject_id}", "func",
            f"sub-{subject_id}_task-laughter_run-{run_str}"
            "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        )
        bold_files = glob.glob(bold_pattern)
    if not bold_files:
        return None
    bold_path = bold_files[0]

    # Cache path
    ts_cache_path = None
    if cache_dir:
        ts_cache_dir = os.path.join(cache_dir, "timeseries")
        os.makedirs(ts_cache_dir, exist_ok=True)
        ts_cache_path = os.path.join(
            ts_cache_dir, f"sub-{subject_id}_run-{run_str}.npy"
        )

    # Return cached time-series if available
    if ts_cache_path and os.path.exists(ts_cache_path):
        ts = np.load(ts_cache_path)
        ts = np.nan_to_num(ts, nan=0.0) # zero out any NaN
        print(f"  sub-{subject_id} run-{run_str}: loaded TS from cache {ts.shape}")
        return ts

    # Extract via masker
    try:
        ts = masker.fit_transform(bold_path)
    except Exception as e:
        print(f"  Masking failed for sub-{subject_id} run-{run_str}: {e}")
        return None

    # Zero out NaNs
    # after atlas resampling (standardise=True -> 0/0 = NaN for empty parcels)
    n_nan = int(np.isnan(ts).sum())
    if n_nan > 0:
        print(f"  sub-{subject_id} run-{run_str}: replaced {n_nan} NaN values with 0")
    ts = np.nan_to_num(ts, nan=0.0)

    if ts_cache_path:
        np.save(ts_cache_path, ts)
        print(f"  sub-{subject_id} run-{run_str}: extracted & cached TS {ts.shape}")
    else:
        print(f"  sub-{subject_id} run-{run_str}: BOLD shape = {ts.shape}")
    return ts


# Window extraction
def trial_locked_windows(
    time_series: np.ndarray, # (T, N_parcels)
    regressors: np.ndarray, # (T, N_conditions)
    events_df: pd.DataFrame,
    tr: float,
    window_trs: int,
    onset_offset_sec: float = 0.0,
):
    """
    Extract fixed-length windows centred on each trial onset.

    Returns:
    windows : list of (N, W, 1) (parcels x time x channel)
    cond_wins : list of (W, C) (time x conditions)
    labels : list of condition index 0-4
    """
    N = time_series.shape[1]
    windows, cond_wins, labels = [], [], []

    onset_offset_trs = int(onset_offset_sec / tr)

    for _, row in events_df.iterrows():
        onset_tr = int(row["onset"] / tr) - onset_offset_trs
        end_tr = onset_tr + window_trs

        if onset_tr < 0 or end_tr > time_series.shape[0]:
            continue # skip trials that would violate the constraints

        ts_win = time_series[onset_tr:end_tr, :] # (W, N)
        cond_win = regressors[onset_tr:end_tr, :] # (W, C)

        # STGCNBlock shape expected: (N, W, 1)
        windows.append(
            ts_win.T[:, :, np.newaxis].astype(np.float32)
        )
        cond_wins.append(cond_win.astype(np.float32))
        labels.append(int(row["condition"]) - 1) # 0-indexed

    return windows, cond_wins, labels


def first_window_per_condition(windows, cond_wins, labels):
    """
    Return the first available trial-locked window for each condition.
    """
    examples = {}
    for win, cond_win, label in zip(windows, cond_wins, labels):
        if label not in examples:
            examples[int(label)] = (win, cond_win)
        if len(examples) == config.N_CONDITIONS:
            break
    return examples


def sliding_windows(
    time_series: np.ndarray, # (T, N_parcels)
    regressors: np.ndarray, # (T, N_conditions)
    window_trs: int,
    stride_trs: int,
):
    """
    Slide a fixed-length window over the whole run.

    Returns:
    windows : list of (N, W, 1) arrays
    cond_wins : list of (W, C) arrays
    dom_conds : list of index of dominant condition in that window
    """
    T, N = time_series.shape
    windows, cond_wins, dom_conds = [], [], []

    start = 0
    while start + window_trs <= T:
        end = start + window_trs
        ts_win = time_series[start:end, :]
        cond_win = regressors[start:end, :]

        windows.append(ts_win.T[:, :, np.newaxis].astype(np.float32))
        cond_wins.append(cond_win.astype(np.float32))

        # Dominant condition = argmax of mean regressor in window
        dom = int(cond_win.mean(axis=0).argmax())
        dom_conds.append(dom)

        start += stride_trs

    return windows, cond_wins, dom_conds


# Per-subject data extraction
def extract_subject_data(
    subject_id: str,
    masker: NiftiLabelsMasker,
    use_sliding: bool = True,
    cache_dir: str = None,
):
    """
    Find all 4 runs for *subject_id*, extract BOLD time-series and
    build windowed samples.

    Returns:
    all_windows : list of (N, W, 1) arrays
    all_cond_wins : list of (W, C) arrays
    all_labels : list of int
    """
    all_windows, all_cond_wins, all_labels = [], [], []

    for run in range(1, 5): # runs 1,2,3,4,5
        run_str = f"{run:02d}"

        # Get time-series — uses cache if available
        ts = get_run_timeseries(subject_id, run_str, masker, cache_dir)
        if ts is None:
            print(f"  sub-{subject_id} run-{run_str}: BOLD not found")
            continue

        # locate events TSV
        events_pattern = os.path.join(
            config.BIDS_ROOT,
            f"sub-{subject_id}",
            "func",
            f"sub-{subject_id}_task-laughter_run-{run_str}_events.tsv",
        )
        events_files = glob.glob(events_pattern)
        if not events_files:
            print(f"  sub-{subject_id} run-{run_str}: events TSV not found")
            continue
        events_path = events_files[0]

        T, N = ts.shape

        # condition regressors
        regressors = build_condition_regressors(events_path, T, config.TR)

        # trial-locked windows
        events_df = load_events(events_path)
        wins, conds, labs = trial_locked_windows(
            ts, regressors, events_df,
            tr=config.TR,
            window_trs=config.WINDOW_TRS,
            onset_offset_sec=config.ONSET_OFFSET_SEC,
        )
        all_windows.extend(wins)
        all_cond_wins.extend(conds)
        all_labels.extend(labs)

        # sliding windows for data augmentation
        if use_sliding:
            s_wins, s_conds, s_labs = sliding_windows(
                ts, regressors,
                window_trs=config.SLIDING_WINDOW_TRS,
                stride_trs=config.SLIDING_STRIDE_TRS,
            )
            all_windows.extend(s_wins)
            all_cond_wins.extend(s_conds)
            all_labels.extend(s_labs)

    return all_windows, all_cond_wins, all_labels


# Dataset
class fMRIWindowDataset(Dataset):
    """
    fMRI windows for autistic / non-autistic binary classification.

    Each item:
        X : (N_parcels, W, 1) - BOLD window
        cond : (W, C) - HRF-convolved condition regressors
        y : 0 = non-autistic, 1 = autistic
        y_cond : 0-4, condition index
        static_feat : (F,) z-scored demographic features, or zeros if
                      static_feat_map is none

    When augment=True (training), two transforms are applied per sample:
      1. Gaussian noise: x' = x + epsilon, epsilon ~ N(0, sigma^2)
      2. Parcel dropout: each parcel zeroed independently with probability p
    """

    def __init__(
        self,
        subject_list,
        masker,
        use_sliding=True,
        cache_dir=None,
        condition_filter=None,
        parcel_indices=None,
        augment=False,
        laughter_mode=False,
        static_feat_map=None,
    ):
        self.samples = [] # list of (X_tensor, cond_tensor, y_group, y_cond, static_tensor)
        self.subject_ids = [] # parallel list tracking which subject each window belongs to
        self.condition_filter = condition_filter
        self.parcel_indices = (
            np.asarray(parcel_indices, dtype=int)
            if parcel_indices is not None else None
        )
        self.augment = augment
        self.laughter_mode = laughter_mode
        self.static_feat_map = static_feat_map or {}

        # Static feature dimensionality inferred from the map (0 if no map)
        _sample_vec = next(iter(self.static_feat_map.values()), None)
        self.static_dim = int(_sample_vec.shape[0]) if _sample_vec is not None else 0

        for subj_id, group_label in subject_list:
            print(f"Processing sub-{subj_id} (label={group_label}).")

            cache_path = None
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                slide_tag = f"s{config.SLIDING_STRIDE_TRS}_rb" if use_sliding else "s0_rb"
                cache_path = os.path.join(
                    cache_dir,
                    f"sub-{subj_id}_w{config.WINDOW_TRS}_{slide_tag}.npz",
                )

            if cache_path and os.path.exists(cache_path):
                data = np.load(cache_path, allow_pickle=True)
                def _to_float32(arr):
                    if isinstance(arr, np.ndarray) and arr.dtype == object:
                        # 0-d object wrapping the inner array
                        if arr.ndim == 0:
                            arr = arr.item()
                        else:
                            # N-d object array
                            return arr.astype(np.float32)
                    return np.asarray(arr, dtype=np.float32)
                windows = [_to_float32(w) for w in data["windows"]]
                cond_wins = [_to_float32(c) for c in data["cond_wins"]]
                labels = list(np.asarray(data["labels"]))
                print(f"  Loaded {len(windows)} windows from cache.")
            else:
                windows, cond_wins, labels = extract_subject_data(
                    subj_id, masker, use_sliding=use_sliding, cache_dir=cache_dir
                )
                if cache_path and windows:
                    np.savez_compressed(
                        cache_path,
                        windows=np.stack(windows).astype(np.float32),
                        cond_wins=np.stack(cond_wins).astype(np.float32),
                        labels=np.array(labels),
                    )
                print(f"  Extracted {len(windows)} windows.")

            # Static demographic feature vector
            # Shape (F,) with F = len(STATIC_FEATURES), or (0,) when disabled
            raw_static = self.static_feat_map.get(subj_id, None)
            if raw_static is not None:
                static_tensor = torch.from_numpy(raw_static.astype(np.float32))
            else:
                static_tensor = torch.zeros(self.static_dim, dtype=torch.float32)

            for win, cond_reg, cond_label in zip(windows, cond_wins, labels):
                # condition filter if set
                if condition_filter is not None and cond_label != condition_filter:
                    continue
                # Laughter mode: spontaneous (0) and conversational (1)
                if laughter_mode and cond_label not in (0, 1):
                    continue

                # Classification target and condition input
                if laughter_mode:
                    # Target: laughter type (0=spontaneous, 1=conversational)
                    # Condition regressors zeroed to prevent leakage through FiLM
                    target = float(cond_label)
                    cond_tensor = torch.zeros_like(torch.from_numpy(cond_reg))
                else:
                    # Target: group label (0=non-autistic, 1=autistic)
                    target = float(group_label)
                    cond_tensor = torch.from_numpy(cond_reg)

                # Keep only selected parcels after cache loading/extraction.
                # win shape is (N_parcels, W, 1), so parcel_indices selects rows.
                if self.parcel_indices is not None:
                    win = win[self.parcel_indices, :, :]

                self.samples.append((
                    torch.from_numpy(win), # (N, W, 1)
                    cond_tensor, # (W, C)
                    torch.tensor(target, dtype=torch.float32),
                    torch.tensor(cond_label, dtype=torch.long), # condition index 0-4
                    static_tensor, # (F,) per-subject demographic features
                ))
                self.subject_ids.append(subj_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, cond, y_group, y_cond, static_feat = self.samples[idx]
        if self.augment:
            # Gaussian noise: x' = x + epsilon, epsilon ~ N(0, sigma^2)
            if config.AUGMENT_NOISE_STD > 0:
                X = X + torch.randn_like(X) * config.AUGMENT_NOISE_STD
            # Parcel dropout
            if config.AUGMENT_PARCEL_DROP > 0:
                keep = (torch.rand(X.shape[0]) > config.AUGMENT_PARCEL_DROP).float()
                X = X * keep.unsqueeze(1).unsqueeze(2)
        return X, cond, y_group, y_cond, static_feat


# DataLoader
def collate_fn(batch):
    X       = torch.stack([b[0] for b in batch]) # (B, N, W, 1)
    cond    = torch.stack([b[1] for b in batch]) # (B, W, C)
    y       = torch.stack([b[2] for b in batch]) # (B,) group label
    y_cond  = torch.stack([b[3] for b in batch]) # (B,) condition index 0-4
    static  = torch.stack([b[4] for b in batch]) # (B, F) demographic features
    return X, cond, y, y_cond, static
