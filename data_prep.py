"""
data_prep.py

Loads the CICIDS2017 MachineLearningCVE CSVs, cleans known dataset quirks
(whitespace in column names, mangled 'Web Attack' labels, infinite values
in Flow Bytes/s and Flow Packets/s), removes duplicate rows and redundant
columns, filters out classes with too few samples to train/evaluate
reliably, and returns a train/test split ready for modeling.

Usage:
    from data_prep import load_and_prepare_data
    X_train, X_test, y_train, y_test, label_encoder = load_and_prepare_data(
        "C:/Users/mved0/Downloads/MachineLearningCSV/MachineLearningCVE"
    )
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Default dataset location -- override with --data-dir or the CICIDS_DIR
# environment variable instead of editing each script.
DEFAULT_DATA_DIR = os.environ.get(
    "CICIDS_DIR", r"C:\Users\mved0\Downloads\MachineLearningCSV\MachineLearningCVE"
)

# Classes with fewer than ~1,000 samples were dropped -- there isn't enough
# data for a model to learn a reliable pattern, or for us to meaningfully
# evaluate performance on them. See README for the full class distribution.
# The dropped classes are still used as "never seen before" attacks to
# evaluate the anomaly detection layer (see train_anomaly.py).
KEEP_CLASSES = [
    'BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye',
    'FTP-Patator', 'SSH-Patator', 'DoS slowloris',
    'DoS Slowhttptest', 'Bot', 'Web Attack - Brute Force'
]

# Columns that carry no information in this dataset: the first 8 are zero
# for every one of the 2.83M flows, the rest are exact copies of another
# column (a known CICFlowMeter quirk -- e.g. 'Fwd Header Length.1' is a
# duplicated header, and 'SYN Flag Count' is identical to 'Fwd PSH Flags').
REDUNDANT_COLUMNS = [
    'Bwd PSH Flags', 'Bwd URG Flags',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Fwd Header Length.1', 'Subflow Fwd Packets', 'Subflow Bwd Packets',
    'SYN Flag Count', 'CWE Flag Count',
]


def load_flows(csv_dir):
    """Loads and cleans every flow (all classes, including the rare ones).
    Returns a DataFrame of features plus 'Label' and 'Day' (source file)
    columns. Row order within each file is preserved."""
    csv_files = sorted(glob.glob(f"{csv_dir}/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    df = pd.concat(
        [pd.read_csv(f).assign(Day=os.path.basename(f)) for f in csv_files],
        ignore_index=True,
    )

    # CICIDS2017 column names have inconsistent leading/trailing whitespace
    df.columns = df.columns.str.strip()

    # The "Web Attack" labels contain a mangled encoding artifact character
    # in place of a dash. Match on the reliable substring instead of the
    # exact (broken) string.
    df.loc[
        df['Label'].str.contains('Brute Force') & df['Label'].str.startswith('Web Attack'),
        'Label'
    ] = 'Web Attack - Brute Force'
    df.loc[df['Label'].str.contains('XSS'), 'Label'] = 'Web Attack - XSS'
    df.loc[df['Label'].str.contains('Sql Injection'), 'Label'] = 'Web Attack - Sql Injection'

    df = df.drop(columns=REDUNDANT_COLUMNS)

    # ~308k rows are exact duplicates of another flow. Left in, a random
    # split puts identical rows in both train and test (58% of PortScan
    # test rows had a copy in train), so the test score partly measures
    # memorization. Keep the first occurrence only.
    df = df.drop_duplicates(subset=[c for c in df.columns if c != 'Day'])

    # Flow Bytes/s and Flow Packets/s divide by flow duration, which is
    # occasionally 0 -- producing infinity. Mark as missing here; they are
    # filled later from training-set statistics only.
    feature_cols = [c for c in df.columns if c not in ('Label', 'Day')]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return df


def split_flows(df, test_size=0.2, random_state=42, split="random"):
    """Splits cleaned flows (from load_flows) into train/test.

    split="random":  stratified random split of rows.
    split="ordered": for each (day, class), the first (1 - test_size) of its
                     flows in file order go to train and the rest to test --
                     an approximately chronological split, so the model is
                     tested on later traffic from an attack than it saw in
                     training (the CSVs have no timestamp column).
    """
    df = df[df['Label'].isin(KEEP_CLASSES)]

    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(df['Label']), index=df.index)
    X = df.drop(columns=['Label', 'Day'])

    if split == "random":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    elif split == "ordered":
        position = df.groupby(['Day', 'Label']).cumcount()
        group_size = df.groupby(['Day', 'Label'])['Label'].transform('size')
        is_train = position < (1 - test_size) * group_size
        X_train, X_test = X[is_train], X[~is_train]
        y_train, y_test = y[is_train], y[~is_train]
    else:
        raise ValueError(f"Unknown split: {split}")

    # Fill missing values using the TRAINING set's median (never the test
    # set's, to avoid leaking test-set statistics into preprocessing).
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    return X_train, X_test, y_train, y_test, label_encoder


def load_and_prepare_data(csv_dir, test_size=0.2, random_state=42, split="random"):
    return split_flows(load_flows(csv_dir), test_size, random_state, split)


def data_dir_arg(description):
    """Shared command-line parser so every script takes --data-dir."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="folder containing the MachineLearningCVE CSVs")
    return parser


if __name__ == "__main__":
    args = data_dir_arg("Smoke test for the data pipeline").parse_args()
    X_train, X_test, y_train, y_test, le = load_and_prepare_data(args.data_dir)
    print("Train:", X_train.shape, " Test:", X_test.shape)
    print("Classes:", list(le.classes_))
