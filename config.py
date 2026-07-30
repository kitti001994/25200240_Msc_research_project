import os

BIDS_ROOT = "/SAN/medic/BUCNIfMRI/BUCNIRawData_BIDS"
# Mac: /Users/kittipinter/Desktop/UCL_modules/fMRI/BUCNIRawData_BIDS
# HPC: /SAN/medic/BUCNIfMRI/BUCNIRawData_BIDS

# Preprocessed BOLD: derivatives/sub-<id>/func/
PROJECT_ROOT = os.path.dirname(BIDS_ROOT)
FMRIPREP_DIR = os.path.join(PROJECT_ROOT, "derivatives")

# Brainnetome 274-region atlas
ATLAS_PATH = "/SAN/medic/BUCNIfMRI/derivatives/atlases/BN_Atlas_274_combined.nii"
# Mac: /Users/kittipinter/Desktop/UCL_modules/fMRI/derivatives/atlases/BN_Atlas_274_combined.nii
# HPC: /SAN/medic/BUCNIfMRI/derivatives/atlases/BN_Atlas_274_combined.nii

# Preprocessed numpy arrays and graph edges
CACHE_DIR = "./cache"

# Per-run BOLD time-series cache
TS_CACHE_DIR = "./cache/timeseries"

# checkpoints
CHECKPOINT_DIR = "./checkpoints"

# ST-GCN only checkpoint
STGCN_ONLY_CHECKPOINT_DIR = "./checkpoints_stgcn_only"

# ST-GCN only LOSO checkpoint
LOSO_STGCN_ONLY_CHECKPOINT_DIR = "./checkpoints_loso_stgcn_only"
FIGURE_FORMATS = ("png",)

# Demographic features
# Used when --demographics flag is passed to train.py
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEMOGRAPHIC_NT_PATH  = os.path.join(_MODEL_DIR, "NT_demographic.xlsx")
DEMOGRAPHIC_ASD_PATH = os.path.join(_MODEL_DIR, "ASD_demographic.xlsx")

# Features are z-scored with training-set stats when --demographics is used.
# AQ is excluded
STATIC_FEATURES = ["age", "gender", "fsiq"]
EXCLUDED_STATIC_FEATURES = ["aq"]

CHECKPOINT_EVERY = 10

# Resume the training if it already exists
RESUME_TRAINING = True

TR = 3.0 # Repetition Time
N_PARCELS = 274 # BN_Atlas_274_combined - 274 parcels (246 cortical + 28 subcortical)

# Conditions
# Maps the integer stored in the 'condition' column of the TSV to a name.
CONDITION_MAP = {
    1: "spontaneous_laughter",
    2: "conversational_laughter",
    3: "non_emotional_sound",
    4: "rest",
    5: "beep",
}
N_CONDITIONS = len(CONDITION_MAP) # 5

# K-fold cross-validation
K_FOLDS = 5

# Window length: 20 TRs at TR=3s = 60s window.
# Stimuli last 1.54 to 3.29s.

ONSET_OFFSET_SEC = 0.0 # window starts exactly at stimulus onset
WINDOW_DURATION_SEC = 60.0 # 60s at TR 3s
WINDOW_TRS = 20 # number of TRs per window

# Sliding window augmentation (training data only)
SLIDING_WINDOW_TRS = WINDOW_TRS # same window length as trial locked
SLIDING_STRIDE_TRS = 10 # stride 10 TRs (30s), 50% overlap

# Graph construction
CORR_THRESHOLD = 0.3 # Pearson |r| threshold
SELF_LOOPS = True # self-loops on every node

# Model architecture
IN_CHANNELS = 1 # single BOLD signal per parcel per TR
HIDDEN_CHANNELS = [16, 32, 64] # ST-GCN layer output sizes
TEMPORAL_KERNEL = 3 # temporal conv kernel size
FILM_HIDDEN_DIM = 32 # hidden units inside FiLM MLP
DROPOUT = 0.3 

# Training
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2 # micro-batch = BATCH_SIZE // GRAD_ACCUM_STEPS = 8 per forward pass
LR = 5e-4
WEIGHT_DECAY = 1e-3 # L2 regularisation
EPOCHS = 150
PATIENCE = 30
LABEL_SMOOTHING = 0.1

# Learning rate warmup
LR_WARMUP_EPOCHS = 10

# Gaussian noise: x_aug = x + epsilon, epsilon ~ N(0, sigma^2)
# Applied to z-scored BOLD signal, so sigma = 0.1 means ~10% of 1-std noise.
AUGMENT_NOISE_STD = 0.05

AUGMENT_PARCEL_DROP = 0.02

# Generates interpolated samples: x_mix = lam*x_i + (1-lam)*x_j
MIXUP_ALPHA = 0.1
VAL_SPLIT = 0.15 # validation fraction (of total subjects, approximate per fold)
TEST_SPLIT = 0.15 # held-out test set fraction
SEED = 42

# Labels
# Binary classification: autistic = 1, non-autistic = 0.
GROUP_COLUMN = "group"
ASD_LABEL = "ASD"
NT_LABEL = "NT"

# LOSO cross-validation
# Platt scaling: p_calibrated = sigma(a * logit + b), a/b fit on inner validation.
# Subject probability: p_subject = mean(p_calibrated) over all windows.

LOSO_CHECKPOINT_DIR = "./checkpoints_loso"
LOSO_INNER_VAL_FRAC = 0.20 # 20% of the N-1 training subjects for inner validation
LOSO_EPOCHS = 100 # max epochs per LOSO fold
LOSO_PATIENCE = 20 # early stopping patience per LOSO fold
