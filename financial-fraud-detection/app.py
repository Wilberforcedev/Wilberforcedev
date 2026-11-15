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
CATEGORICAL = ["category", "country", "channel"]
MIN_HISTORY = 5
RISK_LABELS = {
    "Critical": "🔴 Critical",
    "High": "🟠 High",
    "Medium": "🟡 Medium",
    "Low": "🟢 Low",
}
SIGNAL_WEIGHTS = {
    "spending_spike": 30,
    "new_category": 14,
    "new_country": 22,
    "night_activity": 12,
    "velocity_burst": 25,
    "high_amount": 14,
    "model_outlier": 24,
}
SIGNAL_LABELS = {
    "spending_spike": "Amount is well above the customer's prior spending baseline",
    "new_category": "Category has not appeared in this customer's earlier activity",
    "new_country": "Country has not appeared in this customer's earlier activity",
    "night_activity": "Transaction occurred between 00:00 and 04:59 UTC, or from 23:00 UTC",
    "velocity_burst": "At least three transactions occurred within a rolling 10-minute window",
    "high_amount": "Amount is at or above the 99th percentile for this uploaded dataset",
    "model_outlier": "Isolation Forest found an unusual combination of transaction features",
}


def make_demo_data(seed: int = 21) -> pd.DataFrame:
    """Create synthetic UGX activity and three injected unusual transactions."""
    rng = np.random.default_rng(seed)
    ranges = {
        "Groceries": (18_000, 145_000),
        "Fuel": (25_000, 105_000),
        "Dining": (12_000, 92_000),
        "Shopping": (35_000, 230_000),
        "Utilities": (28_000, 165_000),
        "Travel": (80_000, 380_000),
        "Entertainment": (15_000, 120_000),
    }
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
    for customer_number in range(1, 36):
        customer = f"C{customer_number:04d}"
        spending_factor = float(rng.lognormal(0, 0.28))
        for transaction_number in range(22):
            category = str(rng.choice(list(ranges)))
            low, high = ranges[category]
            timestamp = pd.Timestamp("2026-10-01") + pd.Timedelta(
                days=int(rng.integers(0, 45)),
                hours=int(rng.choice([7, 8, 9, 10, 12, 13, 16, 17, 18, 19, 20, 21])),
                minutes=int(rng.integers(0, 60)),
            )
            rows.append({
                "transaction_id": f"TX-{customer_number:03d}-{transaction_number + 1:03d}",
                "customer_id": customer,
                "timestamp": timestamp,
                "amount": round(float(rng.uniform(low, high) * spending_factor), 2),
                "merchant": str(rng.choice(merchants[category])),
                "category": category,
                "country": str(rng.choice(["Uganda", "Uganda", "Uganda", "Kenya", "Rwanda", "Tanzania"])),
                "channel": str(rng.choice(["POS", "Online", "Mobile", "ATM"], p=[0.42, 0.27, 0.24, 0.07])),
            })

    injected = [
        ("C0003", "TX-DEMO-001", 8_250_000, "Electronics", "Nigeria", "Online"),
        ("C0011", "TX-DEMO-002", 5_900_000, "Travel", "United Arab Emirates", "Online"),
        ("C0022", "TX-DEMO-003", 7_400_000, "Shopping", "United Kingdom", "Mobile"),
    ]
    for customer, transaction_id, amount, category, country, channel in injected:
        rows.append({
            "transaction_id": transaction_id,
            "customer_id": customer,
            "timestamp": pd.Timestamp("2026-11-14 02:17") + pd.Timedelta(minutes=len(rows) % 11),
            "amount": amount,
            "merchant": "New merchant",
            "category": category,
            "country": country,
            "channel": channel,
        })
    return pd.DataFrame(rows)


def normalize_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED if column not in raw.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    data = raw[REQUIRED].copy()
    for column in ["transaction_id", "customer_id"]:
        data[column] = data[column].fillna("").astype(str).str.strip()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    for column in ["merchant", "category", "country", "channel"]:
        data[column] = data[column].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    invalid = (
        data["transaction_id"].eq("")
        | data["customer_id"].eq("")
        | data["timestamp"].isna()
        | data["amount"].isna()
        | (data["amount"] <= 0)
    )
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} row(s) need a transaction ID, customer ID, valid timestamp, "
            "and a positive numeric amount."
        )
    if data.empty:
        raise ValueError("The CSV contains no transactions.")
    if data["transaction_id"].duplicated().any():
        raise ValueError("Transaction IDs must be unique.")
    return data.sort_values(["customer_id", "timestamp", "transaction_id"]).reset_index(drop=True)


def add_behavioral_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Compare each row with earlier rows for that customer; never use future activity."""
    d = data.sort_values(["customer_id", "timestamp", "transaction_id"]).reset_index(drop=True).copy()
    row_count = len(d)
    d["prior_transaction_count"] = np.zeros(row_count, dtype=int)
    d["prior_median_amount"] = np.nan
    d["spending_spike"] = False
    d["new_category"] = False
    d["new_country"] = False

    for _, indexes in d.groupby("customer_id", sort=False).groups.items():
        positions = np.asarray(list(indexes), dtype=int)
        amounts = d.loc[positions, "amount"].to_numpy(dtype=float)
        categories = d.loc[positions, "category"].to_numpy()
        countries = d.loc[positions, "country"].to_numpy()
        for offset, position in enumerate(positions):
            prior_amounts = amounts[:offset]
            prior_count = len(prior_amounts)
            d.at[position, "prior_transaction_count"] = prior_count
            if prior_count < MIN_HISTORY:
                continue

            median = float(np.median(prior_amounts))
            mad = float(np.median(np.abs(prior_amounts - median)))
            d.at[position, "prior_median_amount"] = median
            spike_limit = median + max(4 * mad, median * 1.5)
            d.at[position, "spending_spike"] = amounts[offset] > max(spike_limit, median * 2)
            d.at[position, "new_category"] = categories[offset] not in set(categories[:offset])
            d.at[position, "new_country"] = countries[offset] not in set(countries[:offset])

    hour_utc = d["timestamp"].dt.hour
    d["night_activity"] = (hour_utc < 5) | (hour_utc >= 23)

    burst = np.zeros(row_count, dtype=bool)
    for _, indexes in d.groupby("customer_id", sort=False).groups.items():
        positions = np.asarray(list(indexes), dtype=int)
        times = d.loc[positions, "timestamp"].astype("int64").to_numpy()
        left = 0
        for right, current in enumerate(times):
            while current - times[left] > pd.Timedelta(minutes=10).value:
                left += 1
            burst[positions[right]] = right - left + 1 >= 3
    d["velocity_burst"] = burst
    d["high_amount"] = d["amount"] >= d["amount"].quantile(0.99)
    return d.sort_values(["timestamp", "transaction_id"], ascending=[False, True]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def score_transactions(data: pd.DataFrame, contamination: float) -> pd.DataFrame:
    scored = add_behavioral_signals(data)
    features = scored.copy()
    features["log_amount"] = np.log1p(features["amount"])
    features["hour_utc"] = features["timestamp"].dt.hour
    features["weekday_utc"] = features["timestamp"].dt.dayofweek
    numeric_features = ["log_amount", "hour_utc", "weekday_utc"]
    model = Pipeline([
        ("prepare", ColumnTransformer([
            ("numeric", RobustScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])),
        ("isolation_forest", IsolationForest(
            n_estimators=180,
            contamination=contamination,
            random_state=17,
            n_jobs=-1,
        )),
    ])
    scored["model_outlier"] = model.fit_predict(features[numeric_features + CATEGORICAL]) == -1

    raw_score = sum(
        scored[signal].astype(int) * weight
        for signal, weight in SIGNAL_WEIGHTS.items()
    )
    scored["risk_score"] = raw_score.clip(0, 100).astype(int)
    scored["risk_level"] = pd.cut(
        scored["risk_score"],
        bins=[-1, 24, 49, 74, 100],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)
    scored["reasons"] = scored.apply(
        lambda row: " • ".join(
            f"{SIGNAL_LABELS[signal]} (+{SIGNAL_WEIGHTS[signal]})"
            for signal in SIGNAL_WEIGHTS
            if row[signal]
        ) or "No strong risk signals found",
        axis=1,
    )
    return scored


def format_amount(value: float, currency: str) -> str:
    decimals = 2 if currency in {"USD", "EUR", "GBP"} else 0
    return f"{currency} {value:,.{decimals}f}"


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
currency = st.sidebar.selectbox(
    "Amount currency",
    ["UGX", "USD", "EUR", "GBP", "KES", "TZS", "RWF"],
    index=0,
    help="Select the single currency used by every row in this CSV. Amounts are not converted.",
)
contamination = st.sidebar.slider(
    "Expected outlier share", 0.01, 0.15, 0.05, 0.01,
    help="Controls Isolation Forest sensitivity. This is not a fraud probability.",
)
review_threshold = st.sidebar.slider("Review score threshold", 25, 85, 50, 5)
uploaded = st.sidebar.file_uploader("Upload transaction CSV", type=["csv"])
with st.sidebar.expander("Required CSV columns"):
    st.code(", ".join(REQUIRED))
    st.caption("Use ISO timestamps, a single currency, positive numeric amounts, and unique transaction IDs.")

st.markdown(
    '<div class="hero"><div class="eyebrow">FRAUD &amp; ANOMALY MONITORING</div>'
    '<h1>Financial Fraud Detection</h1>'
    '<p>Find transactions that break a customer’s normal pattern and send the right activity for review.</p></div>',
    unsafe_allow_html=True,
)
try:
    raw_data = pd.read_csv(uploaded) if uploaded else make_demo_data()
    transactions = normalize_transactions(raw_data)
except Exception as error:
    st.error(f"Could not load the transaction file: {error}")
    st.stop()

with st.spinner("Comparing activity with customer behavior…"):
    scored = score_transactions(transactions, contamination)
scored["review_flag"] = scored["risk_score"] >= review_threshold
flagged = scored[scored["review_flag"]]
source_label = uploaded.name if uploaded else "Synthetic demonstration data"
st.caption(
    f"Source: {source_label} · {len(scored):,} transactions · "
    f"{scored['customer_id'].nunique():,} customers · {currency}"
)
metrics = st.columns(4)
metrics[0].metric("Transactions reviewed", f"{len(scored):,}")
metrics[1].metric(
    "Flagged for review",
    f"{len(flagged):,}",
    f"{len(flagged) / max(len(scored), 1):.1%} of activity",
)
metrics[2].metric("Critical risk", f"{(scored['risk_level'] == 'Critical').sum():,}")
metrics[3].metric("Customers observed", f"{scored['customer_id'].nunique():,}")

st.markdown('<div class="section">MONITORING OVERVIEW</div>', unsafe_allow_html=True)
left, right = st.columns([1.1, 1])
with left:
    st.markdown("### Risk distribution")
    distribution = (
        scored["risk_level"].value_counts()
        .reindex(["Critical", "High", "Medium", "Low"], fill_value=0)
        .rename_axis("Risk level")
        .to_frame("Transactions")
    )
    st.bar_chart(distribution, color="#e2573e", height=235)
with right:
    st.markdown("### Review queue")
    if flagged.empty:
        st.success("No transactions meet the current review threshold.")
    else:
        for _, item in flagged.nlargest(6, "risk_score").iterrows():
            with st.container(border=True):
                first, second = st.columns([1.5, 0.7])
                first.markdown(f"**{item['transaction_id']}** · {item['customer_id']}")
                first.caption(
                    f"{item['merchant']} · {item['country']} · "
                    f"{item['timestamp'].strftime('%d %b, %H:%M UTC')}"
                )
                second.markdown(f"### {item['risk_score']} / 100")
                second.caption(RISK_LABELS.get(item["risk_level"], item["risk_level"]))

st.markdown('<div class="section">TRANSACTION TREND</div>', unsafe_allow_html=True)
daily_activity = (
    scored.assign(day=scored["timestamp"].dt.floor("D"))
    .groupby("day")
    .agg(
        Transactions=("transaction_id", "count"),
        Flagged=("review_flag", "sum"),
    )
)
st.line_chart(daily_activity, height=230)

st.markdown('<div class="section">TRANSACTION ACTIVITY</div>', unsafe_allow_html=True)
search_col, risk_col, flagged_col, customer_col = st.columns([1.3, 0.8, 0.7, 1.2])
search = search_col.text_input("Search activity", placeholder="Transaction, merchant…")
risk_filter = risk_col.selectbox("Risk level", ["All", "Critical", "High", "Medium", "Low"])
flagged_only = flagged_col.toggle("Flagged only", value=False)
customer_filter = customer_col.multiselect(
    "Customer",
    sorted(scored["customer_id"].unique()),
    placeholder="All customers",
)

view = scored.copy()
if search:
    matches = view[["transaction_id", "customer_id", "merchant", "country"]].astype(str).apply(
        lambda column: column.str.contains(search, case=False, na=False)
    ).any(axis=1)
    view = view[matches]
if risk_filter != "All":
    view = view[view["risk_level"] == risk_filter]
if flagged_only:
    view = view[view["review_flag"]]
if customer_filter:
    view = view[view["customer_id"].isin(customer_filter)]

table = view[[
    "transaction_id", "timestamp", "customer_id", "merchant", "category",
    "amount", "prior_median_amount", "country", "channel", "risk_score",
    "risk_level", "review_flag",
]].copy()
table["timestamp"] = table["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
table["risk_level"] = table["risk_level"].map(RISK_LABELS)
table = table.rename(columns={
    "transaction_id": "Transaction",
    "timestamp": "Time (UTC)",
    "customer_id": "Customer",
    "merchant": "Merchant",
    "category": "Category",
    "amount": "Amount",
    "prior_median_amount": "Prior median",
    "country": "Country",
    "channel": "Channel",
    "risk_score": "Risk score",
    "risk_level": "Risk",
    "review_flag": "Review",
})
st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        "Amount": st.column_config.NumberColumn(format="%,.0f"),
        "Prior median": st.column_config.NumberColumn(format="%,.0f"),
        "Risk score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
        "Review": st.column_config.CheckboxColumn(),
    },
)

st.markdown("### Transaction investigation")
transaction_ids = view["transaction_id"].astype(str).tolist()
if transaction_ids:
    selected_id = st.selectbox("Choose a transaction to inspect", transaction_ids)
    item = scored.loc[scored["transaction_id"].astype(str) == selected_id].iloc[0]
    detail, signals = st.columns([1, 1.3])
    with detail:
        st.metric("Risk score", f"{item['risk_score']} / 100", RISK_LABELS.get(item["risk_level"]))
        st.write(f"**Customer:** {item['customer_id']}")
        st.write(f"**Transaction:** {item['merchant']} · {item['category']}")
        st.write(f"**Amount:** {format_amount(item['amount'], currency)}")
        st.write(f"**Prior customer median:** {format_amount(item['prior_median_amount'], currency) if pd.notna(item['prior_median_amount']) else 'Building baseline'}")
        st.write(f"**Time:** {item['timestamp'].strftime('%d %b %Y, %H:%M UTC')}")
        st.write(f"**Location / channel:** {item['country']} · {item['channel']}")
    with signals:
        st.markdown("**Signals contributing to the score**")
        if item["reasons"] == "No strong risk signals found":
            st.info(item["reasons"])
        else:
            for reason in item["reasons"].split(" • "):
                st.markdown(f"- {reason}")
        st.caption("A customer baseline is shown after at least five earlier transactions.")
        if "review_decisions" not in st.session_state:
            st.session_state.review_decisions = {}
        previous_decision = st.session_state.review_decisions.get(selected_id, "Pending")
        decision_options = ["Pending", "Cleared", "Escalated", "Confirmed fraud"]
        decision = st.selectbox(
            "Analyst review outcome",
            decision_options,
            index=decision_options.index(previous_decision),
            key=f"decision_{selected_id}",
        )
        st.session_state.review_decisions[selected_id] = decision
else:
    st.info("No transactions match these filters.")

export = scored.copy()
export["timestamp"] = export["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
decisions = st.session_state.get("review_decisions", {})
review_outcomes = export["transaction_id"].astype(str).map(decisions)
default_outcomes = pd.Series(
    np.where(export["review_flag"], "Pending", "Not flagged"),
    index=export.index,
)
export["analyst_review_outcome"] = review_outcomes.where(
    review_outcomes.notna(), default_outcomes
)
export["currency"] = currency
st.download_button(
    "Download scored transactions (CSV)",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name="sentinel_scored_transactions.csv",
    mime="text/csv",
)
st.markdown(
    '<div class="note"><strong>Decision support only.</strong> A flag indicates an unusual pattern, not confirmed fraud. '
    'The default data is synthetic. Review alerts with a human analyst and validate this approach on representative '
    'historical data before operational use.</div>',
    unsafe_allow_html=True,
)
