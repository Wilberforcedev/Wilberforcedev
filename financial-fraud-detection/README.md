# Financial Fraud Detection System

A Streamlit decision-support prototype that combines an unsupervised Scikit-learn Isolation Forest with customer-history signals calculated using Pandas.

## Features

- Scores each transaction from 0–100 and assigns Low, Medium, High, or Critical risk.
- Detects spending spikes, a new category or country, late-night activity, rapid transaction bursts, and unusually large amounts.
- Uploads CSV data, filters the review queue, explains contributing signals, and exports scored results.
- Generates synthetic sample activity on first launch, including a few injected anomalies to demonstrate the workflow.

## Run locally

Use Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## CSV format

Required columns: `transaction_id`, `customer_id`, `timestamp`, `amount`, `merchant`, `category`, `country`, and `channel`.

Use unique transaction IDs, numeric amounts, and ISO 8601 timestamps. Customer identifiers should be stable, and each customer should have enough history to establish a meaningful baseline.

## Scoring notes

The Isolation Forest learns unusual combinations of amount, time, weekday, category, country, and channel without labelled fraud examples. Transparent rules add score points for explainable behavior changes. The sidebar lets analysts tune model sensitivity and the review threshold.

The score is a prioritization aid, not a calibrated probability of fraud. This prototype does not connect to a bank, process live payments, or make automated account decisions. Review alerts and validate the approach on representative, appropriately labelled historical data before operational use. Handle uploaded data under your institution's privacy and security requirements.
