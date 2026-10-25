"""
evaluation.py

Shared evaluation for every model, so they're compared on the same metrics.

Accuracy is near-useless here (always predicting BENIGN scores ~80%), so
the headline numbers are:
  - macro-F1: averages F1 over classes, so rare attacks count as much as
    BENIGN
  - BENIGN false-positive rate: the share of normal traffic raised as an
    attack. This is what decides alert volume in a real SOC -- 1% of a
    million daily flows is 10,000 false alerts.
  - attack detection rate: share of attack flows flagged as *any* attack
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

REPORT_DIR = "reports"


def summarize(y_true, y_pred, label_encoder, name):
    """Prints a report, saves metrics JSON + confusion matrix CSV under
    reports/, and returns the metrics dict."""
    classes = list(label_encoder.classes_)
    benign = classes.index('BENIGN')
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    report = classification_report(
        y_true, y_pred, labels=range(len(classes)), target_names=classes,
        output_dict=True, zero_division=0,
    )
    metrics = {
        "model": name,
        "macro_f1": f1_score(y_true, y_pred, average='macro'),
        "benign_false_positive_rate": float(np.mean(y_pred[y_true == benign] != benign)),
        "attack_detection_rate": float(np.mean(y_pred[y_true != benign] != benign)),
        "accuracy": report["accuracy"],
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")}
                      for c in classes},
    }

    print(f"\n--- {name} ---")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3, zero_division=0))
    print(f"Macro-F1:                  {metrics['macro_f1']:.4f}")
    print(f"BENIGN false-positive rate: {metrics['benign_false_positive_rate']:.4%}")
    print(f"Attack detection rate:     {metrics['attack_detection_rate']:.4%}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    with open(f"{REPORT_DIR}/{slug}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=range(len(classes))),
        index=[f"true: {c}" for c in classes], columns=[f"pred: {c}" for c in classes],
    ).to_csv(f"{REPORT_DIR}/{slug}_confusion.csv")
    return metrics
