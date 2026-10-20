"""
data_prep.py

Loads the CICIDS2017 MachineLearningCVE CSVs, cleans known dataset quirks
(whitespace in column names, mangled 'Web Attack' labels, infinite values
in Flow Bytes/s and Flow Packets/s), filters out classes with too few
samples to train/evaluate reliably, and returns a train/test split ready
for modeling.

Usage:
    from data_prep import load_and_prepare_data
    X_train, X_test, y_train, y_test, label_encoder = load_and_prepare_data(
        "C:/Users/mved0/Downloads/MachineLearningCSV/MachineLearningCVE"
    )
"""

import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Classes with fewer than ~1,000 samples were dropped -- there isn't enough
# data for a model to learn a reliable pattern, or for us to meaningfully
# evaluate performance on them. See README for the full class distribution.
KEEP_CLASSES = [
    'BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye',
    'FTP-Patator', 'SSH-Patator', 'DoS slowloris',
    'DoS Slowhttptest', 'Bot', 'Web Attack - Brute Force'
]


def load_and_prepare_data(csv_dir, test_size=0.2, random_state=42):
    # --- Load and combine the 8 daily CSV files ---
    csv_files = glob.glob(f"{csv_dir}/*.csv")
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

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

    # Keep only classes with enough samples to be learnable/evaluable
    df_filtered = df[df['Label'].isin(KEEP_CLASSES)].copy()

    # Encode text labels as integers
    label_encoder = LabelEncoder()
    df_filtered['Label_encoded'] = label_encoder.fit_transform(df_filtered['Label'])

    X = df_filtered.drop(columns=['Label', 'Label_encoded'])
    y = df_filtered['Label_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Flow Bytes/s and Flow Packets/s divide by flow duration, which is
    # occasionally 0 -- producing infinity. Replace with NaN, then fill
    # using the TRAINING set's median (never the test set's, to avoid
    # leaking test-set statistics into preprocessing).
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

    return X_train, X_test, y_train, y_test, label_encoder


if __name__ == "__main__":
    # Quick smoke test -- update this path to your local dataset location
    X_train, X_test, y_train, y_test, le = load_and_prepare_data(
        r"C:\Users\mved0\Downloads\MachineLearningCSV\MachineLearningCVE"
    )
    print("Train:", X_train.shape, " Test:", X_test.shape)
    print("Classes:", list(le.classes_))
