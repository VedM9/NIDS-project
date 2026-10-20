"""
train_random_forest.py

Trains a Random Forest classifier as an interpretable baseline for the
NIDS multi-class problem, then evaluates it on the held-out test set.

Random Forest was chosen as the baseline because it handles unscaled,
tabular data well, trains quickly, and produces feature importances --
useful in a security context where an analyst often needs to explain
*why* something was flagged, not just that it was.
"""

import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from data_prep import load_and_prepare_data

DATA_DIR = r"C:\Users\mved0\Downloads\MachineLearningCSV\MachineLearningCVE"
MODEL_DIR = "."  # where to save the trained artifacts


def main():
    X_train, X_test, y_train, y_test, le = load_and_prepare_data(DATA_DIR)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    start = time.time()
    rf.fit(X_train, y_train)
    print(f"Training took {time.time() - start:.1f} seconds")

    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Which features the model actually relies on -- a sanity check against
    # the model exploiting something too convenient (e.g. Destination Port
    # alone), and useful material for a project write-up.
    importances = rf.feature_importances_
    top_features = sorted(
        zip(X_train.columns, importances), key=lambda x: x[1], reverse=True
    )[:15]
    print("\nTop 15 features by importance:")
    for name, score in top_features:
        print(f"  {name}: {score:.4f}")

    # Save everything the dashboard needs to score new data:
    # the model itself, the label encoder (to turn predictions back into
    # class names), the training column order (new data must match it),
    # and the training medians (to fill any infinite/missing values the
    # same way we did during training).
    joblib.dump(rf, f"{MODEL_DIR}/rf_model.pkl")
    joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")
    joblib.dump(list(X_train.columns), f"{MODEL_DIR}/feature_columns.pkl")
    joblib.dump(X_train.median(), f"{MODEL_DIR}/train_medians.pkl")
    print(f"\nSaved model artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
from data_prep import load_and_prepare_data

X_train, X_test, y_train, y_test, le = load_and_prepare_data(
    r"C:\Users\mved0\Downloads\MachineLearningCSV\MachineLearningCVE"
)

# Grab 200 random test rows to use as a sample upload for the dashboard
sample = X_test.sample(200, random_state=1)
sample.to_csv("sample_flows.csv", index=False)
print("Saved sample_flows.csv")