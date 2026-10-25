"""
train_random_forest.py

Trains a Random Forest classifier as an interpretable baseline for the
NIDS multi-class problem, then evaluates it on the held-out test set.

Random Forest was chosen as the baseline because it handles unscaled,
tabular data well, trains quickly, and produces feature importances --
useful in a security context where an analyst often needs to explain
*why* something was flagged, not just that it was.

Usage:
    python train_random_forest.py                  # random split, saves model artifacts
    python train_random_forest.py --split ordered  # chronological-style split, evaluation only
"""

import time
import joblib
from sklearn.ensemble import RandomForestClassifier

from data_prep import load_and_prepare_data, data_dir_arg
from evaluation import summarize

MODEL_DIR = "."  # where to save the trained artifacts


def main():
    parser = data_dir_arg("Train and evaluate the Random Forest baseline")
    parser.add_argument("--split", choices=["random", "ordered"], default="random")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, le = load_and_prepare_data(args.data_dir, split=args.split)
    print("Train:", X_train.shape, " Test:", X_test.shape)

    # min_samples_leaf stops each tree growing until every leaf holds a
    # single flow. Fully-grown trees here reached depth 54 and made a
    # 116 MB model; a leaf minimum keeps it small enough to ship and
    # explain (SHAP) with little accuracy cost.
    rf = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=3, random_state=42, n_jobs=-1
    )

    start = time.time()
    rf.fit(X_train, y_train)
    print(f"Training took {time.time() - start:.1f} seconds")

    summarize(y_test, rf.predict(X_test), le, f"Random Forest ({args.split} split)")

    # Which features the model actually relies on -- a sanity check against
    # the model exploiting something too convenient (e.g. Destination Port
    # alone), and useful material for a project write-up.
    top_features = sorted(
        zip(X_train.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True
    )[:15]
    print("\nTop 15 features by importance:")
    for name, score in top_features:
        print(f"  {name}: {score:.4f}")

    if args.split != "random":
        return  # the dashboard model is always the random-split one

    # Save everything the dashboard needs to score new data:
    # the model itself, the label encoder (to turn predictions back into
    # class names), the training column order (new data must match it),
    # and the training medians (to fill any infinite/missing values the
    # same way we did during training).
    joblib.dump(rf, f"{MODEL_DIR}/rf_model.pkl", compress=3)
    joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")
    joblib.dump(list(X_train.columns), f"{MODEL_DIR}/feature_columns.pkl")
    joblib.dump(X_train.median(), f"{MODEL_DIR}/train_medians.pkl")
    print(f"\nSaved model artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
