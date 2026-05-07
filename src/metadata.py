# scr/metadata.py

from pathlib import Path
import pandas as pd

from .config import RAW_DATA_DIR


def infer_label_from_subject(subject_id: str) -> int:
    """
    Infer the label (0 for healthy control, 1 for MDD) from the subject ID.
    """
    subject_upper = subject_id.upper()

    if "HS" in subject_upper:
        return 0
    elif "MDD" in subject_upper:
        return 1 
    else: 
        raise ValueError(f"Could not infer label from subject ID: {subject_id}")
    

def parse_condition_from_filename(filepath: Path) -> str:
    """
    Parse the condition (eyesClosed, eyesOpen, P300) from the filename.
    """
    name = filepath.name 

    if "task-eyesClosed" in name:
        return "eyesClosed"
    elif "task-eyesOpen" in name:
        return "eyesOpen"
    elif "task-P300" in name:
        return "P300"
    else:
        raise ValueError(f"Could not parse condition from filename: {name}")


def build_metadata(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """ Build recording metadata table from EDF files."""        
    edf_files =sorted(raw_data_dir.glob("sub*/eeg/*.edf"))

    rows = []

    for filepath in edf_files:
        subject_id = filepath.parts[-3] 

        rows.append(
            {
                "patient_id": subject_id,
                "recording_id": filepath.stem,
                "label": infer_label_from_subject(subject_id),
                "condition": parse_condition_from_filename(filepath),
                "filepath": str(filepath),
            }
        )

    metadata_df = pd.DataFrame(rows)

    if metadata_df.empty:
        raise FileNotFoundError(f"No EDF files found in {raw_data_dir}")

    return metadata_df  