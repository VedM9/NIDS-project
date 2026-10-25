"""
anomaly.py

Helpers shared by train_anomaly.py and the dashboard for the anomaly
detection layer: an Isolation Forest trained on BENIGN traffic only, which
flags flows that don't look like normal traffic -- including attack types
the classifier has never seen.
"""

import numpy as np


def signed_log1p(X):
    """Flow features are extremely heavy-tailed (bytes/s spans 0 to ~2e9).
    Isolation Forest picks split points uniformly between a feature's min
    and max, so on raw values almost every split lands in the empty tail.
    A log scale spreads the data out; the sign is kept because a few
    features (e.g. Init_Win_bytes_forward = -1) are negative."""
    X = np.asarray(X, dtype=np.float64)
    return np.sign(X) * np.log1p(np.abs(X))


def anomaly_scores(detector, X):
    """Higher = more anomalous. `detector` is the dict saved by
    train_anomaly.py."""
    return -detector["pipeline"].score_samples(X)


def is_anomalous(detector, X):
    return anomaly_scores(detector, X) > detector["threshold"]
