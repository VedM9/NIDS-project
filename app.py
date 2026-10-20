"""
app.py

A SOC-style dashboard for the NIDS project. Upload a CSV of network flow
data (same feature columns as CICIDS2017) and the trained Random Forest
model flags each row as BENIGN or a specific attack type, with a
confidence score and a severity label.

This is a scoring/triage tool, not a training script -- it loads the
artifacts saved by train_random_forest.py rather than retraining anything.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="NIDS Dashboard", layout="wide")

# Attack types considered higher severity for the purposes of this
# dashboard's color-coding -- a judgment call, not a scientific ranking.
# DoS/DDoS and brute-force attacks are flagged HIGH since they typically
# require faster response; scans and bots are MEDIUM.
SEVERITY = {
    'BENIGN': 'None',
    'DDoS': 'High', 'DoS Hulk': 'High', 'DoS GoldenEye': 'High',
    'DoS slowloris': 'High', 'DoS Slowhttptest': 'High',
    'FTP-Patator': 'High', 'SSH-Patator': 'High',
    'Web Attack - Brute Force': 'High',
    'PortScan': 'Medium', 'Bot': 'Medium',
}

SEVERITY_COLOR = {'None': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}


@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    train_medians = joblib.load("train_medians.pkl")
    return model, label_encoder, feature_columns, train_medians


def prepare_uploaded_data(df, feature_columns, train_medians):
    """Cleans an uploaded CSV the same way the training data was cleaned,
    and aligns its columns to exactly what the model expects."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Drop label columns if the uploaded file happens to include them --
    # we don't want the "answer" accidentally used as a feature.
    df = df.drop(columns=[c for c in ['Label', 'Label_encoded'] if c in df.columns])

    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Uploaded file is missing expected columns: {missing_cols[:5]}...")

    df = df[feature_columns]  # enforce the exact training column order
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(train_medians)
    return df


def main():
    st.title("Network Intrusion Detection Dashboard")
    st.caption(
        "Upload network flow data (CICIDS2017-style CSV) to classify each "
        "flow as benign or a specific attack type."
    )

    model, label_encoder, feature_columns, train_medians = load_artifacts()

    uploaded_file = st.file_uploader("Upload flow data (CSV)", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV to get started. Expects the same flow-level "
                 "columns as the CICIDS2017 dataset (Flow Duration, packet "
                 "counts, byte counts, etc.).")
        return

    raw_df = pd.read_csv(uploaded_file)

    try:
        X = prepare_uploaded_data(raw_df, feature_columns, train_medians)
    except ValueError as e:
        st.error(str(e))
        return

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    predicted_labels = label_encoder.inverse_transform(predictions)
    confidence = probabilities.max(axis=1)

    results = pd.DataFrame({
        "Prediction": predicted_labels,
        "Confidence": confidence,
        "Severity": [SEVERITY.get(label, "Medium") for label in predicted_labels],
    })

    # --- Summary metrics ---
    total = len(results)
    attack_count = (results["Prediction"] != "BENIGN").sum()
    high_severity_count = (results["Severity"] == "High").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total flows scanned", total)
    col2.metric("Flagged as attacks", attack_count, f"{attack_count / total:.1%}")
    col3.metric("High severity", high_severity_count)

    # --- Breakdown by class ---
    st.subheader("Detections by type")
    st.bar_chart(results["Prediction"].value_counts())

    # --- Detailed table ---
    st.subheader("Flagged traffic")
    flagged = results[results["Prediction"] != "BENIGN"].sort_values(
        "Confidence", ascending=False
    )

    if flagged.empty:
        st.success("No attack traffic detected in this file.")
    else:
        def highlight_severity(row):
            color = SEVERITY_COLOR.get(row["Severity"], "#ffffff")
            return [f"background-color: {color}33"] * len(row)  # 33 = alpha

        st.dataframe(
            flagged.style.apply(highlight_severity, axis=1).format({"Confidence": "{:.1%}"}),
            use_container_width=True,
        )

    with st.expander("Show full results (including benign traffic)"):
        st.dataframe(results, use_container_width=True)


if __name__ == "__main__":
    main()
