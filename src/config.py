from pathlib import Path

# ===============================
# Paths
# ===============================

def find_project_root():
    """ Find the project root by walking upward until a .git folder is found."""
    p = Path.cwd()

    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
        
    raise FileNotFoundError("Project root (with .git) not found")

PROJECT_ROOT = find_project_root()

DATA_dir = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_dir/ "raw" / "nm000114"
SEGMENTED_DATA_DIR = PROJECT_ROOT / "data" / "segmented_data"

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "table"
FIGURE_DIR = RESULTS_DIR / "figures"

SEGMENTED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# Experiment Settings
# ============================


# ============================
# Reproducibility
# ============================
RANDOM_STATE = 42

# ============================
# Cross-validation
# ============================
N_SPLITS = 5

RUN_DEEP_MODELS = False    # keep False unless you intentionally want to retrain CNN/LSTM

# ============================
# EEG Settings
# ============================
COMMON_CHANNELS = [
'EEG Fp1-LE', 'EEG F3-LE', 'EEG C3-LE', 'EEG P3-LE', 'EEG O1-LE',
 'EEG F7-LE', 'EEG T3-LE', 'EEG T5-LE', 'EEG Fz-LE', 'EEG Fp2-LE', 
 'EEG F4-LE', 'EEG C4-LE', 'EEG P4-LE', 'EEG O2-LE', 'EEG F8-LE', 
 'EEG T4-LE', 'EEG T6-LE', 'EEG Cz-LE', 'EEG Pz-LE', 'EEG A2-A1'
]

BANDS = {
    'delta': (1,4),
    'theta': (4,8),
    'alpha': (8,13),
    'beta': (13,30)
}

# =========================
# Window Settings
# =========================

WELCH_WINDOW_SIZES = [5.0, 10.0]
RAW_WINDOW_SIZE = 10.0
LSTM_WINDOW_SIZES = [10.0, 15.0]