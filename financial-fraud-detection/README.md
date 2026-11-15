# Financial Fraud Detection System

Sentinel is a Streamlit decision-support app for reviewing unusual financial transactions. It uses Pandas for transaction preparation, Scikit-learn's Isolation Forest for unsupervised anomaly detection, and transparent behavioral rules to explain why activity may deserve attention.

The app starts with synthetic UGX activity. Analysts can instead upload a CSV, select its currency, tune sensitivity and review threshold, inspect alert reasons, record a session review outcome, and export scored transactions.

## Quick start

Requires Python 3.10 or newer.

```bash
cd financial-fraud-detection
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL, usually http://localhost:8501.

## Main capabilities

- Isolation Forest flags unusual combinations of transaction amount, UTC hour, weekday, category, country, and channel.
- Customer-specific rules compare each transaction with that customer's earlier activity: amount spikes, new categories, new countries, and bursts of three or more transactions in ten minutes.
- A percentile rule highlights amounts at or above the 99th percentile of the uploaded batch.
- Search and risk filters help narrow the queue; an investigation panel shows signal reasons and the prior median when enough history exists.
- Analysts can mark a selected transaction Pending, Cleared, Escalated, or Confirmed fraud. These decisions last for the current app session and appear in the CSV export.
- Results can be downloaded as a CSV, including risk signals, currency, and analyst review outcome.

## Uploading data

The CSV needs these columns:

```text
transaction_id,customer_id,timestamp,amount,merchant,category,country,channel
```

Use unique transaction IDs, non-empty customer IDs, positive numeric amounts, and timestamps parseable by Pandas. Use one currency throughout a file and select that currency in the sidebar; the app labels values but does not convert them. Timestamps are normalized to UTC for time-based signals.

Download [sample_transactions.csv](sample_transactions.csv) to see the expected structure and try a clear spending/location anomaly.

See [Data format and sample guide](docs/data-format.md) for the full data dictionary and validation behavior.

## How scoring works

The score is a capped sum of rule points and an Isolation Forest outlier signal. Default review threshold is 50/100; the expected outlier share tunes only the forest's sensitivity. Risk bands are Low (0–24), Medium (25–49), High (50–74), and Critical (75–100).

Customer rules only use earlier transactions. The first five rows per customer establish a baseline and do not trigger historical spend/category/country checks. See [Scoring and validation](docs/scoring-and-validation.md) for the signals, weights, assumptions, and evaluation guidance.

A score is a queue-prioritization aid, not a calibrated fraud probability. A flagged record is not proof of fraud, and a low score does not establish that a transaction is safe. Human review is required.

## Deployment

For a local, container, or Streamlit Community Cloud setup, see [Deployment guide](docs/deployment.md). Before using real customer records, review the privacy, access, retention, and validation guidance there.

## Project files

- app.py — Streamlit interface, preprocessing, rule signals, and model scoring.
- requirements.txt — Python dependencies.
- sample_transactions.csv — small, fictional upload example.
- docs/ — data format, scoring, validation, and deployment notes.
