# scripts/deep_model_comparison.py

import sys
from pathlib import Path

# Allow imports from project root when running script directly 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.config import TABLES_DIR

def main():
    print("Checking deep learning comparison table...")

    path = TABLES_DIR / "deep_model_comparison.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    
    df = pd.read_csv(path)

    df = df.rename(columns={"roc-auc": "roc_auc", "n-folds": "n_folds"})

    out_path = TABLES_DIR / "deep_model_comparison.csv"
    df.to_csv(out_path, index=False)

    print(f"Deep comparison table confirmed: {out_path}")
    print(df)


    if __name__ == "__main__":
        main()