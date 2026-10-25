# Network Intrusion Detection System (NIDS): RF vs CNN, plus an anomaly layer

A flow-based network intrusion detection system trained on the CICIDS2017
dataset. It compares a Random Forest against a 1D CNN (three class-weighting
variants), adds an anomaly detection layer for attack types the classifier
was never trained on, and serves the result through a Streamlit triage
dashboard with SHAP explanations for every alert.

## Problem

Network Intrusion Detection Systems inspect traffic and flag malicious
activity. This project works on **flow-level metadata** (duration, packet
counts, byte counts, inter-arrival timing) rather than packet contents.
That is the same category of data produced by NetFlow, Zeek or firewall
logs, and it still works on encrypted traffic, unlike deep packet
inspection.

## Dataset and cleaning

[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html): ~2.83M
labelled flows over 5 days of lab-generated traffic, 14 attack categories
plus benign.

Cleaning steps (`data_prep.py`), several of which change the results
noticeably:

- **Duplicates removed.** ~308k rows are exact copies of another flow. Left
  in, a random split puts identical rows in both train and test (58% of
  PortScan test rows had a copy in train), so the test score partly
  measures memorisation. Earlier versions of this project skipped this
  step and reported inflated numbers.
- **13 uninformative columns dropped**: 8 are zero for every flow, 5 are
  exact copies of another column (a known CICFlowMeter quirk).
- Mangled `Web Attack` labels fixed; infinite `Flow Bytes/s` /
  `Flow Packets/s` values filled with **training-set** medians only.
- Classes with fewer than ~1,000 samples (`Heartbleed`, `Infiltration`,
  `Web Attack - XSS`, `Web Attack - Sql Injection`) are excluded from
  classifier training. They are reused later as **never-seen attacks** to
  test the anomaly layer.

That leaves **11 classes** and ~2.52M flows (2.02M train / 504k test).
BENIGN is ~83% of the data and the smallest class (`Web Attack - Brute
Force`) has under 1,500 flows, so **accuracy is misleading**. A model that
always says BENIGN scores ~83%. All models are scored the same way
(`evaluation.py`):

- **Macro-F1**: F1 averaged over classes, so rare attacks count as much as
  BENIGN.
- **Benign false-positive rate (FPR)**: share of normal traffic raised as
  an attack. This sets alert volume in a real SOC: 1% of a million daily
  flows is 10,000 false alerts.
- **Attack detection rate**: share of attack flows flagged as *any* attack.

Full per-class reports and confusion matrices are in `reports/`, and raw
training output is in `reports/logs/`.

## Results: known attack types

Random (stratified) 80/20 split, after de-duplication:

| Model | Macro-F1 | Benign FPR | Attack detection | Accuracy |
|---|---|---|---|---|
| **Random Forest** | **0.978** | **0.06%** | **99.7%** | 99.9% |
| CNN, unweighted | 0.839 | 1.76% | 96.2% | 97.9% |
| CNN, sqrt class weights | 0.698 | 5.77% | 97.8% | 94.7% |
| CNN, balanced class weights | 0.511 | 16.8% | 100.0% | 85.8% |

Per-class precision / recall:

| Class | Test flows | RF | CNN unweighted | CNN sqrt | CNN balanced |
|---|---|---|---|---|---|
| BENIGN | 419,297 | 1.00 / 1.00 | 0.99 / 0.98 | 1.00 / 0.94 | 1.00 / 0.83 |
| Bot | 390 | 0.91 / 0.71 | 0.96 / 0.37 | 0.16 / 0.44 | 0.02 / 0.98 |
| DDoS | 25,603 | 1.00 / 1.00 | 1.00 / 0.98 | 0.80 / 1.00 | 0.75 / 1.00 |
| DoS GoldenEye | 2,057 | 1.00 / 0.99 | 0.97 / 0.97 | 0.86 / 0.99 | 0.49 / 1.00 |
| DoS Hulk | 34,570 | 1.00 / 1.00 | 0.98 / 0.95 | 0.94 / 0.95 | 0.81 / 0.97 |
| DoS Slowhttptest | 1,046 | 0.99 / 0.99 | 0.87 / 0.88 | 0.59 / 0.98 | 0.36 / 0.92 |
| DoS slowloris | 1,077 | 1.00 / 1.00 | 0.92 / 0.97 | 0.52 / 0.94 | 0.17 / 0.99 |
| FTP-Patator | 1,187 | 1.00 / 1.00 | 0.99 / 0.98 | 0.96 / 0.98 | 0.30 / 0.99 |
| PortScan | 18,164 | 0.99 / 1.00 | 0.74 / 0.98 | 0.67 / 1.00 | 0.71 / 1.00 |
| SSH-Patator | 644 | 1.00 / 0.99 | 1.00 / 0.91 | 0.17 / 0.95 | 0.05 / 0.96 |
| Web Attack - Brute Force | 294 | 0.99 / 0.99 | 1.00 / 0.10 | 0.15 / 0.92 | 0.04 / 0.97 |

### Findings

**The Random Forest beats every CNN variant on every headline metric.**
The unweighted CNN misses 90% of `Web Attack - Brute Force` flows and 63%
of `Bot` flows. Likely reasons: the smallest classes have only a few
hundred training examples per feature pattern, and a 1D convolution
assumes neighbouring features are related, while the column order of
tabular flow statistics is arbitrary.

**Class weighting trades the CNN's misses for false alarms, and never
gets close to the RF.** Fully `balanced` weights push recall to ~0.97 on
every class, but precision collapses: only 2% of "Bot" alerts are real,
and 16.8% of benign traffic is flagged. The softer square-root weighting
cuts the benign FPR to 5.8%, but that is still ~100x the RF's, and
macro-F1 (0.70) is worse than not weighting at all (0.84). Neither
weighting scheme fixes the underlying problem.

**More complexity wasn't better here.** The Random Forest needs no
scaling or class weighting, trains in about 2 minutes on a CPU, and its
decisions can be explained per alert (see *Dashboard*). It is the model
the dashboard uses.

## Results: a harder, more honest test split

A random split lets the model train on flows from the same attack burst
it is tested on, often seconds apart. The **ordered split**
(`python train_random_forest.py --split ordered`) instead trains on the
first 80% of each class's flows within each day's file and tests on the
last 20%, so the model always predicts *later* traffic than it saw. (The
CSVs have no timestamp column, so file order stands in for time.)

| Random Forest | Macro-F1 | Benign FPR | Attack detection |
|---|---|---|---|
| Random split | 0.978 | 0.06% | 99.7% |
| Ordered split | **0.867** | 0.01% | **95.9%** |

Most classes hold up, but some drop sharply:

- **Bot recall: 0.71 → 0.01.** The later bot traffic looks unlike the
  earlier bot traffic, so the model labels it BENIGN.
- **DoS slowloris recall: 1.00 → 0.66**, **DoS Hulk recall: 1.00 → 0.90**.

This is the more realistic estimate of how the model would do on new
traffic, and the main caveat on the random-split numbers above.

## Anomaly layer: attack types never seen in training

A supervised classifier can only answer with a class it was trained on.
Before this layer was added, the Random Forest labelled **100% of
Heartbleed and Infiltration flows as BENIGN**, with ~95% confidence.

`train_anomaly.py` adds an **Isolation Forest fitted on benign training
flows only**. Its alert threshold is calibrated on a held-out slice of
benign *training* flows (never the test set) to a target benign FPR of
0.5%. The hybrid rule is: alert if the RF predicts an attack, **or** the
RF says BENIGN but the flow's anomaly score is above the threshold.

Detection of the four never-seen classes, with the 0.5% threshold:

| Never-seen class | Flows | RF only | RF + anomaly layer |
|---|---|---|---|
| Heartbleed | 11 | 0% | **100%** |
| Infiltration | 36 | 0% | **31%** |
| Web Attack - Sql Injection | 21 | 29% | 29% |
| Web Attack - XSS | 652 | 92% | 92% |
| **Benign FPR** | | 0.06% | 0.54% |

- Heartbleed (huge, unusual flows) is caught outright.
- Infiltration is partly caught. Loosening the threshold to a 2% FPR raises
  this to 69%, at roughly 4x the false alerts (all thresholds are in
  `reports/hybrid_detector.json`).
- The RF already flags most XSS (as `Web Attack - Brute Force`, a similar
  HTTP pattern). SQL injection flows look like normal web requests at the
  flow level, so neither layer catches most of them. That is a
  limitation of flow metadata, not something a threshold can fix.

## Dashboard

`app.py` is a Streamlit triage dashboard. Upload a CICIDS2017-style CSV
and each flow gets:

- a **verdict**: a known attack type (from the RF), `Unknown anomaly`
  (RF says benign, but the anomaly layer disagrees), or BENIGN
- a **severity** (High / Medium) and the RF's vote share, with
  low-confidence and unknown verdicts marked for analyst review
- a **SHAP explanation** for any flagged flow: the features whose values
  pushed the model towards that verdict. This is the analyst's answer to
  "why was this flagged?"

`sample_flows.csv` is a ready-made demo upload: 200 random test flows plus
5 flows from each never-seen attack class. Its `Label` column is shown as
ground truth but never used as a feature. On this file, the dashboard
classifies every known-attack flow correctly and flags all 5 Heartbleed
flows as unknown anomalies, with one false alarm out of 163 benign flows.

## Limitations

- **CICIDS2017 is lab-generated.** It is known to be easier to separate
  than real traffic, which flatters every model here. The ordered-split
  results are a better guide, and validating on a different dataset
  (e.g. CSE-CIC-IDS2018 or UNSW-NB15) is the real test.
- RF "confidence" is the share of trees that agree, not a calibrated
  probability.
- The anomaly layer's 0.5% FPR is measured on this dataset's benign
  traffic. On a real network it would need recalibrating on that
  network's own normal traffic.

## Project structure

```
data_prep.py            # load, clean, de-duplicate and split CICIDS2017 (random or ordered)
evaluation.py           # shared metrics + saves reports/<model>.json and confusion matrices
train_random_forest.py  # Random Forest; saves the model artifacts the dashboard uses
train_cnn.py            # 1D CNN: unweighted vs balanced vs sqrt class weights
anomaly.py              # anomaly-score helpers shared by training and the dashboard
train_anomaly.py        # Isolation Forest layer + hybrid detector evaluation
make_sample.py          # builds sample_flows.csv for the dashboard demo
app.py                  # Streamlit dashboard with hybrid verdicts and SHAP explanations
reports/                # saved metrics, confusion matrices and training logs
```

## Running it

1. Download the `MachineLearningCVE` CSVs from CICIDS2017.
2. Point the scripts at them with `--data-dir <path>` or the `CICIDS_DIR`
   environment variable.
3. `pip install -r requirements.txt`
4. `python train_random_forest.py`: trains the RF and saves `rf_model.pkl`
   and related artifacts (add `--split ordered` for the evaluation-only
   ordered split).
5. `python train_anomaly.py`: trains the anomaly layer (needs step 4).
6. `python train_cnn.py`: optional, reruns the CNN comparison (~15 min on
   a CPU).
7. `python make_sample.py`: optional, rebuilds `sample_flows.csv`.
8. `streamlit run app.py`, then upload `sample_flows.csv`.

Trained models (`*.pkl`) aren't committed (the RF is over GitHub's file
size limit). Steps 4 and 5 regenerate them.
