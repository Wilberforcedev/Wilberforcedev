from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

st.set_page_config(page_title="Sentinel | Fraud Detection", page_icon="🛡️", layout="wide")
REQUIRED = ["transaction_id", "customer_id", "timestamp", "amount", "merchant", "category", "country", "channel"]
CATS = ["category", "country", "channel"]
RISK = {"Critical": "🔴 Critical", "High": "🟠 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}


def demo_data(seed=21):
    rng = np.random.default_rng(seed)
    ranges = {"Groceries": (18, 145), "Fuel": (25, 105), "Dining": (12, 92), "Shopping": (35, 230),
              "Utilities": (28, 165), "Travel": (80, 380), "Entertainment": (15, 120)}
    merchants = {
        "Groceries": ["Market Square", "Fresh Basket", "City Mart"],
        "Fuel": ["Shell", "TotalEnergies", "City Fuel"],
        "Dining": ["Cafe Javas", "The Bistro", "Garden Grill"],
        "Shopping": ["Metro Boutique", "Style House", "Home & More"],
        "Utilities": ["UMEME", "NWSC", "Airtime"],
        "Travel": ["Swift Coaches", "Skyline Travel", "Kampala Cabs"],
        "Entertainment": ["Century Cinema", "Game Zone", "Music Hall"],
    }
    rows = []
    for number in range(1, 36):
        customer = f"C{number:04d}"
        factor = rng.lognormal(0, .28)
        for n in range(22):
            category = str(rng.choice(list(ranges)))
            low, high = ranges[category]
            when = pd.Timestamp("2026-10-01") + pd.Timedelta(
                days=int(rng.integers(0, 45)), hours=int(rng.choice([7, 8, 9, 10, 12, 13, 16, 17, 18, 19, 20, 21])),
                minutes=int(rng.integers(0, 60)))
            rows.append({
                "transaction_id": f"TX-{number:03d}-{n+1:03d}", "customer_id": customer,
                "timestamp": when, "amount": round(float(rng.uniform(low, high) * factor), 2),
                "merchant": str(rng.choice(merchants[category])), "category": category,
                "country": str(rng.choice(["Uganda", "Uganda", "Uganda", "Kenya", "Rwanda", "Tanzania"])),
                "channel": str(rng.choice(["POS", "Online", "Mobile", "ATM"], p=[.42, .27, .24, .07])),
            })
    for customer, txn, amount, category, country, channel in [
        ("C0003", "TX-DEMO-001", 8250, "Electronics", "Nigeria", "Online"),
        ("C0011", "TX-DEMO-002", 5900, "Travel", "United Arab Emirates", "Online"),
        ("C0022", "TX-DEMO-003", 7400, "Shopping", "United Kingdom", "Mobile"),
    ]:
        rows.append({"transaction_id": txn, "customer_id": customer,
                     "timestamp": pd.Timestamp("2026-11-14 02:17") + pd.Timedelta(minutes=len(rows) % 11),
                     "amount": amount, "merchant": "New merchant", "category": category,
                     "country": country, "channel": channel})
    return pd.DataFrame(rows)


def normalize(raw):
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    data = raw[REQUIRED].copy()
    data["transaction_id"] = data["transaction_id"].astype(str).str.strip()
    data["customer_id"] = data["customer_id"].astype(str).str.strip()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    for col in ["merchant", "category", "country", "channel"]:
        data[col] = data[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    invalid = data["timestamp"].isna() | data["amount"].isna() | (data["amount"] < 0)
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} row(s) have invalid timestamps or amounts.")
    if not len(data):
        raise ValueError("The CSV contains no transactions.")
    if data["transaction_id"].duplicated().any():
        raise ValueError("Transaction IDs must be unique.")
    return data.sort_values("timestamp").reset_index(drop=True)


def behavior(data):
    d = data.copy()
    med = d.groupby("customer_id")["amount"].transform("median")
    mad = d.assign(_median=med).groupby("customer_id")["amount"].transform(
        lambda x: (x - x.median()).abs().median())
    limit = med + np.maximum(4 * mad, med * 1.5)
    d["spending_spike"] = (d["amount"] > limit) & (d["amount"] > med * 2)
    d["new_category"] = d.groupby(["customer_id", "category"])["transaction_id"].transform("count") <= 1
    d["new_country"] = d.groupby(["customer_id", "country"])["transaction_id"].transform("count") <= 1
    hour = d["timestamp"].dt.hour
    d["night_activity"] = (hour < 5) | (hour >= 23)
    d = d.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    burst = np.zeros(len(d), dtype=bool)
    for _, idx in d.groupby("customer_id", sort=False).groups.items():
        pos = np.asarray(list(idx), dtype=int)
        times = d.loc[pos, "timestamp"].astype("int64").to_numpy()
        left = 0
        for right, current in enumerate(times):
            while current - times[left] > pd.Timedelta(minutes=10).value:
                left += 1
            burst[pos[right]] = right - left + 1 >= 3
    d["velocity_burst"] = burst
    d["high_amount"] = d["amount"] >= d["amount"].quantile(.99)
    return d.sort_values("timestamp", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def score(data, contamination):
    d = behavior(data)
    features = d.copy()
    features["log_amount"] = np.log1p(features["amount"])
    features["hour"] = features["timestamp"].dt.hour
    features["weekday"] = features["timestamp"].dt.dayofweek
    numeric = ["log_amount", "hour", "weekday"]
    model = Pipeline([
        ("prep", ColumnTransformer([
            ("num", RobustScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATS),
        ])),
        ("forest", IsolationForest(n_estimators=180, contamination=contamination, random_state=17, n_jobs=-1)),
    ])
    d["model_outlier"] = model.fit_predict(features[numeric + CATS]) == -1
    weights = {"spending_spike": 30, "new_category": 14, "new_country": 22,
               "night_activity": 12, "velocity_burst": 25, "high_amount": 14, "model_outlier": 24}
    d["risk_score"] = sum(d[k].astype(int) * v for k, v in weights.items()).clip(0, 100).astype(int)
    d["risk_level"] = pd.cut(d["risk_score"], [-1, 24, 49, 74, 100],
                             labels=["Low", "Medium", "High", "Critical"]).astype(str)
    reasons = {"spending_spike": "Spend exceeds customer's usual range",
               "new_category": "Category is new for this customer",
               "new_country": "Country is new for this customer",
               "night_activity": "Transaction occurred late at night",
               "velocity_burst": "Three or more transactions within 10 minutes",
               "high_amount": "Amount is in the top 1% of this file",
               "model_outlier": "Isolation Forest identified an unusual combination"}
    d["reasons"] = d.apply(lambda r: " • ".join(label for key, label in reasons.items() if r[key])
                           or "No strong risk signals found", axis=1)
    return d


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
.stApp{background:#f7f9f7;color:#182822;font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"]{background:#f0f5f1;border-right:1px solid #e4ebe7}
h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.035em;color:#182822}
.hero{background:linear-gradient(112deg,#173f33,#255b47);color:white;border-radius:20px;padding:30px 34px;margin:4px 0 26px;box-shadow:0 12px 28px #143b2d1f}
.hero h1{color:white;margin:4px 0 5px;font-size:clamp(1.8rem,4vw,2.55rem)}
.hero p{color:#d4e5dc;margin:0;font-size:1rem}
.eyebrow,.section{letter-spacing:.12em;font-size:.72rem;font-weight:700;color:#8ed0aa}
.section{color:#6c8177;margin:8px 0 12px}
[data-testid="stMetric"]{background:white;border:1px solid #e4ebe7;border-radius:15px;padding:17px 18px;box-shadow:0 4px 15px #14322609}
[data-testid="stMetricLabel"]{color:#71817a}[data-testid="stMetricValue"]{color:#163c31;font-family:'Manrope',sans-serif}
[data-testid="stVerticalBlockBorderWrapper"]{background:white;border-color:#e4ebe7;border-radius:13px}
.note{background:#eef3ef;border-left:3px solid #85a996;color:#53665b;padding:13px 16px;border-radius:8px;margin-top:24px;font-size:.88rem}
div[data-testid="stDownloadButton"] button{background:#163c31;color:white;border:0;border-radius:9px;font-weight:600}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🛡️ Sentinel")
st.sidebar.caption("Transaction monitoring workspace")
contamination = st.sidebar.slider("Expected outlier share", .01, .15, .05, .01,
                                  help="Controls model sensitivity; this is not a fraud probability.")
threshold = st.sidebar.slider("Review score threshold", 25, 85, 50, 5)
st.sidebar.markdown("---")
upload = st.sidebar.file_uploader("Upload transaction CSV", type=["csv"])
with st.sidebar.expander("Required CSV columns"):
    st.code(", ".join(REQUIRED))
    st.caption("Use ISO timestamps and numeric amounts. Transaction IDs must be unique.")

st.markdown('<div class="hero"><div class="eyebrow">FRAUD &amp; ANOMALY MONITORING</div><h1>Financial Fraud Detection</h1><p>Find transactions that break a customer’s normal pattern and send the right activity for review.</p></div>', unsafe_allow_html=True)
try:
    raw = pd.read_csv(upload) if upload else demo_data()
    source = normalize(raw)
except Exception as exc:
    st.error(f"Could not load the transaction file: {exc}")
    st.stop()

with st.spinner("Comparing activity with customer behavior…"):
    result = score(source, contamination)
result["review_flag"] = result["risk_score"] >= threshold
flagged = result[result["review_flag"]]
st.caption(f"Source: {upload.name if upload else 'Synthetic demonstration data'} · {len(result):,} transactions · {result['customer_id'].nunique():,} customers")
m = st.columns(4)
m[0].metric("Transactions reviewed", f"{len(result):,}")
m[1].metric("Flagged for review", f"{len(flagged):,}", f"{len(flagged)/max(len(result),1):.1%} of activity")
m[2].metric("Critical risk", f"{(result.risk_level == 'Critical').sum():,}")
m[3].metric("Customers observed", f"{result.customer_id.nunique():,}")

st.markdown('<div class="section">MONITORING OVERVIEW</div>', unsafe_allow_html=True)
left, right = st.columns([1.1, 1])
with left:
    st.markdown("### Risk distribution")
    chart = result.risk_level.value_counts().reindex(["Critical","High","Medium","Low"], fill_value=0).rename_axis("Risk level").to_frame("Transactions")
    st.bar_chart(chart, color="#e2573e", height=235)
with right:
    st.markdown("### Review queue")
    if flagged.empty:
        st.success("No transactions meet the current review threshold.")
    else:
        for _, row in flagged.nlargest(6, "risk_score").iterrows():
            with st.container(border=True):
                a, b = st.columns([1.5, .7])
                a.markdown(f"**{row.transaction_id}** · {row.customer_id}")
                a.caption(f"{row.merchant} · {row.country} · {row.timestamp.strftime('%d %b, %H:%M UTC')}")
                b.markdown(f"### {row.risk_score} / 100")
                b.caption(RISK.get(row.risk_level, row.risk_level))

st.markdown('<div class="section">TRANSACTION ACTIVITY</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1.4, .8, .8])
search = c1.text_input("Search activity", placeholder="Transaction, customer, merchant…")
risk = c2.selectbox("Risk level", ["All", "Critical", "High", "Medium", "Low"])
only_flagged = c3.toggle("Flagged only", value=False)
view = result.copy()
if search:
    mask = view[["transaction_id","customer_id","merchant","country"]].astype(str).apply(
        lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
    view = view[mask]
if risk != "All":
    view = view[view.risk_level == risk]
if only_flagged:
    view = view[view.review_flag]
table = view[["transaction_id","timestamp","customer_id","merchant","category","amount","country","channel","risk_score","risk_level","review_flag"]].copy()
table["timestamp"] = table.timestamp.dt.strftime("%Y-%m-%d %H:%M UTC")
table["risk_level"] = table.risk_level.map(RISK)
table = table.rename(columns={"transaction_id":"Transaction","timestamp":"Time (UTC)","customer_id":"Customer","merchant":"Merchant","category":"Category","amount":"Amount","country":"Country","channel":"Channel","risk_score":"Risk score","risk_level":"Risk","review_flag":"Review"})
st.dataframe(table, hide_index=True, use_container_width=True, column_config={
    "Amount": st.column_config.NumberColumn(format="$%.2f"),
    "Risk score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
    "Review": st.column_config.CheckboxColumn(),
})

st.markdown("### Transaction investigation")
ids = view.transaction_id.astype(str).tolist()
if ids:
    chosen = st.selectbox("Choose a transaction to inspect", ids)
    row = result[result.transaction_id.astype(str) == chosen].iloc[0]
    a, b = st.columns([1, 1.3])
    with a:
        st.metric("Risk score", f"{row.risk_score} / 100", RISK.get(row.risk_level))
        st.write(f"**Customer:** {row.customer_id}")
        st.write(f"**Transaction:** {row.merchant} · {row.category}")
        st.write(f"**Amount:** USD {row.amount:,.2f}")
        st.write(f"**Time:** {row.timestamp.strftime('%d %b %Y, %H:%M UTC')}")
        st.write(f"**Location / channel:** {row.country} · {row.channel}")
    with b:
        st.markdown("**Signals contributing to the score**")
        if row.reasons == "No strong risk signals found":
            st.info(row.reasons)
        else:
            for reason in row.reasons.split(" • "):
                st.markdown(f"- {reason}")
        st.caption("Review the underlying transaction and customer context before taking action.")
else:
    st.info("No transactions match these filters.")

export = result.copy()
export["timestamp"] = export.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
st.download_button("Download scored transactions (CSV)", export.to_csv(index=False).encode("utf-8"),
                   "sentinel_scored_transactions.csv", "text/csv")
st.markdown('<div class="note"><strong>Decision support only.</strong> Flags identify unusual patterns, not confirmed fraud. The default sample is synthetic. Validate every alert with a human reviewer and assess the model on representative historical data before operational use.</div>', unsafe_allow_html=True)
