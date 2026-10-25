"""
train_anomaly.py

Trains the anomaly detection layer and evaluates the hybrid detector
(Random Forest + anomaly layer).

Why: a supervised classifier can only answer with a class it was trained
on. The original Random Forest labelled 100% of Heartbleed and
Infiltration flows (classes dropped from training) as BENIGN, with ~95%
confidence. Real NIDS products pair signature/supervised detection with
an anomaly layer for exactly this reason.

How:
  - An Isolation Forest is fitted on BENIGN training flows only.
  - Its alert threshold is set from a slice of BENIGN training flows held
    out from the fit (never the test set), at a target false-positive rate.
  - Hybrid rule: alert if the RF predicts an attack, OR the RF says BENIGN
    but the anomaly score is above the threshold.
  - The classes dropped in data_prep.py (Heartbleed, Infiltration, Web
    Attack - XSS / Sql Injection) are used as never-seen attacks.

Requires rf_model.pkl -- run train_random_forest.py first.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from anomaly import signed_log1p, anomaly_scores
from data_prep import load_flows, split_flows, data_dir_arg, KEEP_CLASSES
from evaluation import REPORT_DIR

# Share of benign flows we accept being raised as anomalies. Evaluated
# alongside stricter/looser options below; 0.5% already catches every
# Heartbleed flow, and doubling it to 1% only adds 3 more Infiltration
# flows (out of 36) for twice the false alerts.
TARGET_FPR = 0.005
FPR_OPTIONS = [0.001, 0.005, 0.01, 0.02, 0.05]


def main():
    args = data_dir_arg("Train the anomaly layer and evaluate the hybrid detector").parse_args()

    flows = load_flows(args.data_dir)
    X_train, X_test, y_train, y_test, le = split_flows(flows)
    benign = list(le.classes_).index('BENIGN')

    rf = joblib.load("rf_model.pkl")
    train_medians = joblib.load("train_medians.pkl")

    # Never-seen attacks: the rare classes excluded from classifier training
    unseen = flows[~flows['Label'].isin(KEEP_CLASSES)]
    X_unseen = unseen[X_train.columns].fillna(train_medians)
    y_unseen = unseen['Label'].to_numpy()
    del flows

    # Fit on BENIGN training flows; keep 10% aside to calibrate the threshold
    X_benign = X_train[y_train == benign]
    X_calib = X_benign.sample(frac=0.1, random_state=0)
    X_fit = X_benign.drop(X_calib.index)

    pipeline = make_pipeline(
        FunctionTransformer(signed_log1p),
        IsolationForest(n_estimators=200, max_samples=1024, random_state=42, n_jobs=-1),
    )
    pipeline.fit(X_fit)
    detector = {"pipeline": pipeline}

    calib_scores = anomaly_scores(detector, X_calib)
    test_scores = anomaly_scores(detector, X_test)
    unseen_scores = anomaly_scores(detector, X_unseen)

    rf_test_attack = rf.predict(X_test) != benign
    rf_unseen = le.inverse_transform(rf.predict(X_unseen))
    test_is_benign = (y_test == benign).to_numpy()

    rows = []
    for fpr in FPR_OPTIONS:
        threshold = np.quantile(calib_scores, 1 - fpr)
        hybrid_test = rf_test_attack | (test_scores > threshold)
        caught_unseen = (rf_unseen != 'BENIGN') | (unseen_scores > threshold)
        row = {
            "target_fpr": fpr,
            "threshold": float(threshold),
            "benign_fpr_rf_only": float(rf_test_attack[test_is_benign].mean()),
            "benign_fpr_hybrid": float(hybrid_test[test_is_benign].mean()),
            "known_attack_detection_hybrid": float(hybrid_test[~test_is_benign].mean()),
        }
        for cls in sorted(set(y_unseen)):
            m = y_unseen == cls
            row[f"unseen_detected_rf_only: {cls}"] = float((rf_unseen[m] != 'BENIGN').mean())
            row[f"unseen_detected_hybrid: {cls}"] = float(caught_unseen[m].mean())
        rows.append(row)

    table = pd.DataFrame(rows).set_index("target_fpr").T
    pd.set_option("display.width", 200)
    print("\nHybrid detector at different anomaly thresholds (columns = target benign FPR):")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nUnseen-class counts:", pd.Series(y_unseen).value_counts().to_dict())

    detector["threshold"] = float(np.quantile(calib_scores, 1 - TARGET_FPR))
    detector["target_fpr"] = TARGET_FPR
    joblib.dump(detector, "anomaly_model.pkl", compress=3)
    table.to_csv(f"{REPORT_DIR}/hybrid_detector.csv")
    with open(f"{REPORT_DIR}/hybrid_detector.json", "w") as f:
        json.dump({"chosen_target_fpr": TARGET_FPR, "results": rows,
                   "unseen_counts": pd.Series(y_unseen).value_counts().to_dict()}, f, indent=2)
    print(f"\nSaved anomaly_model.pkl (target FPR {TARGET_FPR:.1%})")


if __name__ == "__main__":
    main()
