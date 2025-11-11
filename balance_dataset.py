"""
Utility script to oversample the minority classes of the DEMALE-HSJM dataset
so that each class has the same number of samples as the majority class.

The script reads the original Excel file, performs bootstrap resampling for
the minority classes, shuffles the resulting frame, and saves a new spreadsheet
under data/DEMALE-HSJM_2025_data_balanced.xlsx.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SOURCE_FILE = DATA_DIR / "DEMALE-HSJM_2025_data.xlsx"
TARGET_FILE = DATA_DIR / "DEMALE-HSJM_2025_data_balanced.xlsx"
TARGET_COLUMN = "diagnosis"
RANDOM_STATE = 42


def balance_dataset() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source dataset not found at {SOURCE_FILE}")

    df = pd.read_excel(SOURCE_FILE)
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' was not found in the source dataset."
        )

    value_counts = df[TARGET_COLUMN].value_counts()
    max_count = int(value_counts.max())

    balanced_frames = []
    for label, group in df.groupby(TARGET_COLUMN):
        group_size = len(group)
        if group_size == max_count:
            balanced_frames.append(group)
            continue
        upsampled = group.sample(
            n=max_count,
            replace=True,
            random_state=RANDOM_STATE,
        )
        balanced_frames.append(upsampled)

    balanced_df = (
        pd.concat(balanced_frames)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    balanced_df.to_excel(TARGET_FILE, index=False)

    print("Balanced dataset saved to", TARGET_FILE)
    print("Class distribution after balancing:")
    for label, count in balanced_df[TARGET_COLUMN].value_counts().sort_index().items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    balance_dataset()
