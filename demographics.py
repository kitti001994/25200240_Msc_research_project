"""
Demographic data from the Excel files.

Variables:
aq: Autism Quotient (0-50). Non-autistic mean ~14, autistic mean ~37.
    AQ is retained in the demographic files but excluded from model inputs
    by default because it can dominate autistic/non-autistic classification.
age: Included as a confound regressor to make sure the model does not learn
     age-related connectivity patterns.
gender: Binary sex code (1=Male, 2=Female).
fsiq: Full-Scale IQ (WAIS III/IV). Included as a confound again.
"""

import os
import numpy as np
import pandas as pd
import config

_COLUMN_RENAME = {
    "ID": "id",
    "Group": "group",
    "Age": "age",
    "Gender (1=M; 2=F)": "gender",
    "AQ": "aq",
    "VIQ": "viq",
    "IQ (WAIS III & IV) PIQ": "piq",
    "FSIQ": "fsiq",
}

# ID discrepancies between the Excel demographic and BIDS participants.tsv.
# manually corrected against participants.tsv:
#   211119YX: initials reversed in Excel-> BIDS uses 211119XY
#   220304RC: date differs in Excel (04 vs 17); BIDS uses 220317RC
_ID_CORRECTIONS = {
    "211119YX": "211119XY",
    "220304RC":  "220317RC",
}


def load_demographics(
    nt_path: str = None,
    asd_path: str = None,):
    nt_path  = nt_path  or config.DEMOGRAPHIC_NT_PATH
    asd_path = asd_path or config.DEMOGRAPHIC_ASD_PATH

    frames = []
    for path in (nt_path, asd_path):
        df = pd.read_excel(path)
        # Row 0 -> true column headers
        df.columns = df.iloc[0]
        df = df.drop(0).reset_index(drop=True)
        # Drop any column whose header is NaN
        df = df.loc[:, df.columns.notna()]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns=_COLUMN_RENAME)

    # change the columns to numerics
    for col in ("age", "gender", "aq", "viq", "piq", "fsiq"):
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        if combined[col].isna().any():
            combined[col] = combined[col].fillna(combined[col].median())

    # ID strings
    combined["id"] = combined["id"].astype(str).str.strip()

    # Corrections
    combined["id"] = combined["id"].replace(_ID_CORRECTIONS)

    return combined.reset_index(drop=True)


def build_feature_map(
    df: pd.DataFrame,
    feature_names: list,):
    """
    Raw (unnormalised) feature map by subject ID.
    """
    feat_map = {}
    for _, row in df.iterrows():
        vals = np.array([float(row[f]) for f in feature_names], dtype=np.float32)
        vals = np.nan_to_num(vals, nan=0.0)
        feat_map[str(row["id"])] = vals
    return feat_map


def compute_normalisation_stats(
    feat_map: dict,
    subject_ids: list,):
    """
    Compute per-feature mean and standard deviation over a subject list.
    z_i = (x_i - mu) / (sigma + eps)
    """
    vecs = [feat_map[sid] for sid in subject_ids if sid in feat_map]
    if not vecs:
        raise ValueError("No subjects from subject_ids found in feat_map.")
    arr = np.stack(vecs) # (N, F)
    return arr.mean(axis=0).astype(np.float32), arr.std(axis=0).astype(np.float32)


def normalise_feature_map(
    feat_map: dict,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-8,):
    """
    Apply z-score normalisation to all entries in feat_map.
    z = (x - mean) / (std + eps)
    """
    return {
        sid: (vec - mean) / (std + eps)
        for sid, vec in feat_map.items()
    }


def get_feature_dim():
    """Return the number of static features based on config.STATIC_FEATURES."""
    return len(config.STATIC_FEATURES)
