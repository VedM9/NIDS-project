"""
app.py

A SOC-style dashboard for the NIDS project. Upload a CSV of network flow
data (same feature columns as CICIDS2017) and each flow is scored by a
hybrid detector:

  - the trained Random Forest names the attack type it recognises, and
  - an anomaly layer (Isolation Forest trained on benign traffic only)
    flags flows the Random Forest calls BENIGN but that don't look like
    normal traffic -- e.g. attack types it was never trained on.

Every flagged flow can be explained with SHAP: which feature values pushed
the model towards its verdict.

This is a scoring/triage tool, not a training script -- it loads the
artifacts saved by train_random_forest.py and train_anomaly.py rather than
retraining anything.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

from anomaly import signed_log1p, anomaly_scores

st.set_page_config(page_title="NIDS Dashboard", layout="wide")

UNKNOWN = "Unknown anomaly"

# Attack types considered higher severity for the purposes of this
# dashboard's color-coding -- a judgment call, not a scientific ranking.
# DoS/DDoS and brute-force attacks are flagged HIGH since they typically
# require faster response; scans, bots and unknown anomalies are MEDIUM.
SEVERITY = {
    'BENIGN': 'None',
    'DDoS': 'High', 'DoS Hulk': 'High', 'DoS GoldenEye': 'High',
    'DoS slowloris': 'High', 'DoS Slowhttptest': 'High',
    'FTP-Patator': 'High', 'SSH-Patator': 'High',
    'Web Attack - Brute Force': 'High',
    'PortScan': 'Medium', 'Bot': 'Medium', UNKNOWN: 'Medium',
}

SEVERITY_COLOR = {'None': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}

# Below this Random Forest vote share, a prediction is marked for analyst
# review. The RF's "confidence" is the fraction of trees agreeing, not a
# calibrated probability, so low values are a hint, not a guarantee.
LOW_CONFIDENCE = 0.6

# Columns shown next to each verdict so an analyst can see which flow it is
CONTEXT_COLUMNS = ['Destination Port', 'Flow Duration', 'Total Fwd Packets',
                   'Total Backward Packets', 'Flow Bytes/s']


@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    train_medians = joblib.load("train_medians.pkl")
    detector = joblib.load("anomaly_model.pkl") if os.path.exists("anomaly_model.pkl") else None
    return model, label_encoder, feature_columns, train_medians, detector


@st.cache_resource
def load_explainers(_model, _detector):
    rf_explainer = shap.TreeExplainer(_model)
    iforest_explainer = shap.TreeExplainer(_detector["pipeline"][-1]) if _detector else None
    return rf_explainer, iforest_explainer


def prepare_uploaded_data(df, feature_columns, train_medians):
    """Cleans an uploaded CSV the same way the training data was cleaned,
    and aligns its columns to exactly what the model expects."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Uploaded file is missing {len(missing_cols)} expected "
                         f"columns, e.g. {missing_cols[:5]}")

    # Selecting only the training features also drops any 'Label' column,
    # so the "answer" can never be used as a feature.
    df = df[feature_columns]
    df = df.apply(pd.to_numeric, errors='coerce')  # stray text -> NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(train_medians)
    return df


def score_flows(X, model, label_encoder, detector):
    probabilities = model.predict_proba(X)
    predicted = label_encoder.inverse_transform(probabilities.argmax(axis=1))
    results = pd.DataFrame({
        "RF prediction": predicted,
        "Confidence": probabilities.max(axis=1),
    }, index=X.index)

    verdict = results["RF prediction"].copy()
    if detector is not None:
        results["Anomaly score"] = anomaly_scores(detector, X)
        anomalous = results["Anomaly score"] > detector["threshold"]
        verdict[(verdict == "BENIGN") & anomalous] = UNKNOWN

    results.insert(0, "Verdict", verdict)
    results.insert(1, "Severity", [SEVERITY.get(v, "Unrated") for v in verdict])
    results["Review"] = (results["Confidence"] < LOW_CONFIDENCE) | (verdict == UNKNOWN)
    return results


def explain_flow(row_index, X, results, model, label_encoder, detector, explainers):
    """Returns a table of the features that pushed this flow towards its
    verdict, largest effect first."""
    rf_explainer, iforest_explainer = explainers
    x = X.loc[[row_index]]

    if results.at[row_index, "Verdict"] == UNKNOWN:
        # Isolation Forest output is higher for *normal* flows, so negate:
        # positive = pushed towards anomalous.
        contributions = -np.asarray(iforest_explainer.shap_values(signed_log1p(x)))[0]
        target = "anomalous"
    else:
        values = rf_explainer.shap_values(x)
        cls = list(label_encoder.classes_).index(results.at[row_index, "RF prediction"])
        # shap returns (rows, features, classes); older versions a list per class
        contributions = values[cls][0] if isinstance(values, list) else values[0, :, cls]
        target = results.at[row_index, "RF prediction"]

    table = pd.DataFrame({
        "Feature": X.columns,
        "Value": x.iloc[0].to_numpy(),
        f"Push towards '{target}'": contributions,
    })
    order = np.argsort(-np.abs(contributions))[:10]
    return table.iloc[order].set_index("Feature")


def main():
    st.title("Network Intrusion Detection Dashboard")
    st.caption(
        "Upload network flow data (CICIDS2017-style CSV) to classify each "
        "flow as benign, a known attack type, or an unknown anomaly."
    )

    model, label_encoder, feature_columns, train_medians, detector = load_artifacts()
    if detector is None:
        st.warning("anomaly_model.pkl not found -- run train_anomaly.py to enable "
                   "detection of unknown attack types. Showing classifier results only.")

    uploaded_file = st.file_uploader("Upload flow data (CSV)", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV to get started. Expects the same flow-level "
                 "columns as the CICIDS2017 dataset (Flow Duration, packet "
                 "counts, byte counts, etc.). Try sample_flows.csv.")
        return

    try:
        raw_df = pd.read_csv(uploaded_file)
        raw_df.columns = raw_df.columns.str.strip()
        X = prepare_uploaded_data(raw_df, feature_columns, train_medians)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as e:
        st.error(f"Could not read this file: {e}")
        return

    results = score_flows(X, model, label_encoder, detector)
    if "Label" in raw_df.columns:  # ground truth, display only
        results["True label"] = raw_df["Label"]
    context = raw_df[[c for c in CONTEXT_COLUMNS if c in raw_df.columns]]
    results = results.join(context)

    # --- Summary metrics ---
    total = len(results)
    attack_count = (results["Verdict"] != "BENIGN").sum()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total flows scanned", total)
    col2.metric("Flagged", attack_count, f"{attack_count / total:.1%}", delta_color="off")
    col3.metric("High severity", (results["Severity"] == "High").sum())
    col4.metric("Unknown anomalies", (results["Verdict"] == UNKNOWN).sum())

    # --- Breakdown by class ---
    st.subheader("Detections by type")
    st.bar_chart(results["Verdict"].value_counts())

    # --- Detailed table ---
    st.subheader("Flagged traffic")
    flagged = results[results["Verdict"] != "BENIGN"].sort_values(
        ["Severity", "Confidence"], ascending=[True, False]
    )

    if flagged.empty:
        st.success("No attack traffic detected in this file.")
    else:
        def highlight_severity(row):
            color = SEVERITY_COLOR.get(row["Severity"], "#ffffff")
            return [f"background-color: {color}33"] * len(row)  # 33 = alpha

        formats = {"Confidence": "{:.1%}", "Anomaly score": "{:.3f}"}
        st.dataframe(
            flagged.style.apply(highlight_severity, axis=1)
                   .format({k: v for k, v in formats.items() if k in flagged}),
            use_container_width=True,
        )

        # --- Explanation for one flow ---
        st.subheader("Why was this flow flagged?")
        row_index = st.selectbox(
            "Pick a flagged flow (row number in the uploaded file)",
            flagged.index,
            format_func=lambda i: f"Row {i}: {flagged.at[i, 'Verdict']}",
        )
        explainers = load_explainers(model, detector)
        with st.spinner("Computing SHAP explanation..."):
            explanation = explain_flow(row_index, X, results, model, label_encoder,
                                       detector, explainers)
        left, right = st.columns([1, 1])
        left.dataframe(explanation.style.format("{:.4g}"), use_container_width=True)
        right.bar_chart(explanation.iloc[:, 1], horizontal=True)
        st.caption("SHAP values: how much each feature's value moved the model's "
                   "output towards this verdict for this specific flow. Negative "
                   "bars pushed against it.")

    with st.expander("Show full results (including benign traffic)"):
        st.dataframe(results, use_container_width=True)


if __name__ == "__main__":
    main()
