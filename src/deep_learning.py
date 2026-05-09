# src/deep_learning.py

import pandas as pd

from .config import TABLES_DIR, N_SPLITS

CACHE_DIR = TABLES_DIR / "deep_model_comparison"

def load_cached_deep_results():
    """
    Load cached deep learning comparison results.
    """

    path = TABLES_DIR / "deep_model_comparison.csv"

    df = pd.read_csv(path)

    df = df.rename(columns={"roc-auc": "roc_auc", "n-folds": "n_folds"})

    if "n_folds" not in df.columns:
        df["n_folds"] = N_SPLITS
    else:
        df["n_folds"] = df["n_folds"].fillna(N_SPLITS)

    return df


def clean_deep_result_names(df):
    """ 
    Standardize deep model labels for final comparison table.
    """
    df = df.copy()

    model_map = {
        "Raw + MultiScale CNN": "CNN (Multi-Scale)",
        "Raw + Multiscale CNN": "CNN (Multi-Scale)",
        "Raw + Muti-Scale CNN": "CNN (Multi-Scale)",
        "Raw + Baseline CNN": "CNN (Baseline)",
        "Raw + Deep CNN": "CNN (Deep)",
        "Raw + CNN-LSTM": "CNN-LSTM",
        "RAW + LSTM": "LSTM",
        "Raw + LSTM": "LSTM",
    }

    df["model_name"] = df["model_name"].replace(model_map)

    if "method" in df.columns:
        df["method"] = (
            df["method"]
            .str.strip()
            .str.replace(r"Raw \+", "", regex=True)
        )

    return df 

def get_cached_deep_comparison():
    """
    Load and clean cached deep learning comparison table. 
    """
    df = load_cached_deep_results()
    df = clean_deep_result_names(df)

    return df 

def build_deep_model_comparison(results_map, output_name="deep_model_comparison.csv"):
    """
    Combine cache deep=learning summary files into one comparison table.

    Parameters
    ----------
    results_map : dict
        Dictionary mapping model_key -> display method name. 
    ouput_name : str
        Output CSV filename saved under results/tables/.
    """

    rows = []

    for model_key, display_name in results_map.items():
        summary_path = CACHE_DIR / f"{model_key}_summary.csv"

        if summary_path.exists():
            row = pd.read_csv(summary_path).iloc[0].to_dict()
            row["method"] = display_name
            rows.append(row)
        else:
            print(f"Missing cache for {display_name}: {summary_path}")

    comparison = pd.DataFrame(rows)

    if not comparison.empty:
        comparison = comparison.rename(columns={"roc-auc":"roc_auc", "n-folds":"n_folds"})

        if "n_folds" not in comparison.columns:
            comparison["n_folds"] = N_SPLITS
        else:
            comparison["n_folds"] = comparison["n_folds"].fillna(N_SPLITS)

        cols = [
            "method",
            "model_name",
            "window_sec",
            "accuracy",
            "f1",
            "roc_auc",
            "n_folds",
            "notes"
        ]
        
        comparison = comparison[[c for c in cols if c in comparison.columns]]

        output_path = TABLES_DIR / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output_path, index=False)

        print(f"Saved comparison table to: {output_path}")

    return comparison
