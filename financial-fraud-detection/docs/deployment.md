# Deployment and operations

## Local run

From this directory, create an isolated Python environment, install dependencies, and start Streamlit:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Use Python 3.10 or later. Streamlit displays the URL to open, usually http://localhost:8501.

## Streamlit Community Cloud

1. Connect the GitHub repository in Streamlit Community Cloud.
2. Select the main branch.
3. Set the app file path to financial-fraud-detection/app.py.
4. Deploy and confirm the dependency installation completes.

The app currently has no API keys or secrets. Do not add private transaction data, credentials, or secret configuration to the repository.

## Docker

Build and run from the financial-fraud-detection directory:

```bash
docker build -t sentinel-fraud .
docker run --rm -p 8501:8501 sentinel-fraud
```

Open http://localhost:8501. The included Dockerfile binds Streamlit to all container interfaces on port 8501. Place the app behind an approved HTTPS reverse proxy and access control layer before exposing it beyond a trusted local network.

## Privacy and access

Uploaded transaction data is processed by the Streamlit app process and may be present in memory or Streamlit's cache for the running session. Hosting providers may have their own logs, storage, and retention behavior. Review the chosen host's terms and configuration before uploading personal or financial data. Restrict access, use an approved environment, minimize retained data, and delete exported files according to organizational policy.

The app has no authentication, database, audit log, role controls, encryption-at-rest configuration, payment processor connection, or durable review workflow. Session review outcomes are not shared with other users and are lost when the session ends. Add and review these controls before using sensitive production records.

## Operational checklist

- Confirm data rights, privacy notice, retention period, access control, and hosting location.
- Validate CSV schema, timezone convention, currency consistency, and customer history quality.
- Evaluate on labelled historical data and document false-positive/false-negative tradeoffs.
- Define who investigates alerts, how findings are recorded, and how urgent cases are escalated.
- Monitor alert volume, data drift, and review outcomes; recalibrate only through a documented change process.
- Keep human review in the decision loop. Do not treat the score as proof of fraud or as an automatic basis to deny service.
