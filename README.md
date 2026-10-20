# Network Intrusion Detection System (NIDS) — Multi-Model Comparison

A flow-based network intrusion detection system trained on the CICIDS2017
dataset, comparing a Random Forest baseline against a 1D CNN — including
what happened when the CNN's minority-class performance was corrected for
class imbalance.

## Problem

Network Intrusion Detection Systems inspect traffic and flag malicious
activity. This project works on **flow-level metadata** (duration, packet
counts, byte counts, inter-arrival timing) rather than packet contents —
the same category of data produced by tools like NetFlow, Zeek, or
firewall logs, and a realistic basis for production-scale monitoring
(it also works on encrypted traffic, unlike deep packet inspection).

## Dataset

[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) —
~2.83M labeled network flows across 8 days of simulated traffic, covering
14 attack categories plus benign traffic.

**Class imbalance**: BENIGN traffic outnumbers some attack types by close
to 100,000:1. Classes with fewer than ~1,000 samples (`Infiltration`,
`Heartbleed`, `Web Attack - XSS`, `Web Attack - Sql Injection`) were
dropped — there wasn't enough data to train or reliably evaluate them.
That leaves **11 classes**: `BENIGN`, `DoS Hulk`, `PortScan`, `DDoS`,
`DoS GoldenEye`, `FTP-Patator`, `SSH-Patator`, `DoS slowloris`,
`DoS Slowhttptest`, `Bot`, `Web Attack - Brute Force`.

Because of this imbalance, **accuracy alone is misleading** — a model
that always predicts BENIGN would score ~80% while being useless.
Results below are reported as precision / recall / F1 per class.

## Models

| Model | Why |
|---|---|
| Random Forest | Interpretable baseline — handles unscaled tabular data well, and produces feature importances an analyst can use to justify a flagged alert |
| 1D CNN | Tests whether a deep learning approach can learn feature interactions a tree ensemble might miss |
| 1D CNN + class weighting | Tests whether penalizing minority-class errors more heavily in the loss function fixes the CNN's weak minority-class recall |

## Results

| Class | RF (P / R) | CNN unweighted (P / R) | CNN weighted (P / R) |
|---|---|---|---|
| BENIGN | 1.00 / 1.00 | 0.99 / 0.99 | 1.00 / 0.92 |
| Bot | 0.89 / 0.78 | 0.97 / 0.39 | 0.03 / 0.98 |
| DDoS | 1.00 / 1.00 | 0.96 / 1.00 | 0.95 / 1.00 |
| DoS GoldenEye | 1.00 / 1.00 | 0.99 / 0.99 | 0.91 / 0.99 |
| DoS Hulk | 1.00 / 1.00 | 0.96 / 1.00 | 0.86 / 1.00 |
| DoS Slowhttptest | 0.99 / 0.99 | 0.89 / 0.99 | 0.69 / 0.99 |
| DoS slowloris | 0.99 / 0.99 | 0.99 / 0.99 | 0.61 / 0.99 |
| FTP-Patator | 1.00 / 1.00 | 1.00 / 0.99 | 0.60 / 1.00 |
| PortScan | 0.99 / 0.99 | 0.99 / 0.94 | 0.82 / 1.00 |
| SSH-Patator | 1.00 / 1.00 | 0.97 / 0.98 | 0.45 / 0.99 |
| Web Attack - Brute Force | 0.99 / 0.97 | 1.00 / 0.14 | 0.13 / 0.96 |
| **Overall accuracy** | **~1.00** | **~0.99** | **~0.94** |

## Key finding

Contrary to my initial expectation, **the CNN underperformed the Random
Forest baseline**, particularly on minority classes: the unweighted CNN's
recall on `Web Attack - Brute Force` was only 0.14 — missing 86% of real
brute-force attacks in the test set. I attribute this to the CNN's higher
data requirements (the smallest classes had well under 2,000 samples) and
the lack of a genuine spatial/sequential relationship between the tabular
flow features a 1D convolution slides across.

Applying `class_weight='balanced'` to penalize minority-class errors more
heavily **fixed recall but overcorrected**: Bot recall rose from 0.39 to
0.98, but precision collapsed to 0.03 — only 3% of the model's "Bot"
predictions were actually correct, an unacceptable false-positive rate.
`'balanced'` weighting is computed purely from class frequency with no
cap, so the ~1,500-sample minority classes received disproportionately
large weights relative to BENIGN's 2.3M samples.

**Random Forest remained the strongest model overall** — it didn't need
class weighting to handle the imbalance reasonably well in the first
place. This is a useful, practical finding on its own: more model
complexity isn't automatically better, and for this problem the simpler,
more interpretable baseline won.

## What I'd try next

- A moderate, manually-capped class weighting (e.g. capped at ~5–10x
  rather than fully balanced) instead of an all-or-nothing choice
- Oversampling minority classes (e.g. SMOTE) instead of loss reweighting
- Validating against real or more adversarial traffic — CICIDS2017 is
  synthetic/lab-generated and known in the literature to be more
  separable than real-world traffic, which likely flatters all three
  models' reported accuracy somewhat

## Dashboard

`app.py` is a Streamlit dashboard that loads the trained Random Forest
model (chosen as the production model, per the results above) and scores
uploaded flow data, flagging each row's predicted class, confidence, and
a severity label -- turning the model into something closer to a usable
triage tool than a notebook full of metrics.

## Project structure

```
data_prep.py           # loads, cleans, and splits the CICIDS2017 data
train_random_forest.py # baseline model + feature importances + saves model artifacts
train_cnn.py            # unweighted vs. class-weighted CNN comparison
app.py                  # Streamlit dashboard for scoring new flow data
```

## Running it

1. Download the `MachineLearningCVE` CSVs from CICIDS2017
2. Update `DATA_DIR` in `train_random_forest.py` / `train_cnn.py` to your local path
3. `pip install pandas numpy scikit-learn tensorflow streamlit joblib`
4. `python train_random_forest.py` — trains the model and saves it (`rf_model.pkl`, etc.) to the project folder
5. `python train_cnn.py` — optional, runs the CNN comparison
6. `streamlit run app.py` — launches the dashboard in your browser, using the model saved in step 4
