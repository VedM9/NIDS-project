"""
make_sample.py

Writes sample_flows.csv: a small demo upload for the dashboard. It holds
random held-out test flows plus a handful of attacks the classifier never
saw in training (Heartbleed, Infiltration, ...), so the anomaly layer has
something to catch. The true label is kept in a 'Label' column -- the
dashboard shows it as ground truth but never uses it as a feature.
"""

import pandas as pd

from data_prep import load_flows, split_flows, data_dir_arg, KEEP_CLASSES


def main():
    args = data_dir_arg("Create sample_flows.csv for the dashboard").parse_args()

    flows = load_flows(args.data_dir)
    X_train, X_test, y_train, y_test, le = split_flows(flows)

    known = X_test.sample(200, random_state=1)
    known["Label"] = le.inverse_transform(y_test[known.index])

    unseen = flows[~flows["Label"].isin(KEEP_CLASSES)]
    unseen = unseen.groupby("Label").sample(n=5, random_state=1)[list(X_test.columns) + ["Label"]]

    sample = pd.concat([known, unseen]).sample(frac=1, random_state=2)
    sample.to_csv("sample_flows.csv", index=False)
    print(f"Saved sample_flows.csv ({len(known)} test flows + {len(unseen)} unseen-class flows)")


if __name__ == "__main__":
    main()
