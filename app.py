import streamlit as st
import joblib
import numpy as np

# Load models
rf = joblib.load("random_forest.pkl")
xgb = joblib.load("XGBoost.pkl")
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("kmeans_scaler.pkl")

st.title("📊 Trader Behavior Insights")

st.sidebar.header("Input Trade Details")

# Inputs
size_usd = st.sidebar.number_input("Size USD", value=1000.0)
execution_price = st.sidebar.number_input("Execution Price", value=100.0)
fee = st.sidebar.number_input("Fee", value=1.0)
start_position = st.sidebar.number_input("Start Position", value=0.0)

# ✅ NEW: PnL input (fixes clustering issue)
pnl = st.sidebar.number_input("Closed PnL", value=0.0)

# Sentiment dropdown
sentiment_options = [
    "Extreme Fear",
    "Fear",
    "Neutral",
    "Greed",
    "Extreme Greed"
]

selected_sentiment = st.sidebar.selectbox(
    "Market Sentiment",
    sentiment_options
)

sentiment_map = {
    "Extreme Fear": 0,
    "Fear": 1,
    "Neutral": 2,
    "Greed": 3,
    "Extreme Greed": 4
}

sentiment = sentiment_map[selected_sentiment]

# Feature array
features = np.array([[size_usd, execution_price, fee, start_position, sentiment]])

# -------------------------------
# 🔮 Prediction Section
# -------------------------------
st.subheader("🔮 Profitability Prediction")

if st.button("Predict Profit"):
    with st.spinner("Analyzing trade..."):
        rf_pred = rf.predict(features)[0]
        xgb_pred = xgb.predict(features)[0]

    st.success("Prediction Complete!")
    st.write(f"Random Forest: {'Profit' if rf_pred else 'Loss'}")
    st.write(f"XGBoost: {'Profit' if xgb_pred else 'Loss'}")

# -------------------------------
# 🧠 Clustering Section
# -------------------------------
st.subheader("🧠 Trader Type (Clustering)")

cluster_input = np.array([[size_usd, pnl, fee]])
scaled_input = scaler.transform(cluster_input)
cluster = kmeans.predict(scaled_input)[0]

# ⚠️ IMPORTANT: These labels should match your data distribution
cluster_map = {
    0: "Low-risk / Small traders",
    1: "High-risk / Aggressive traders",
    2: "Medium-risk / Active traders"
}

st.write(f"Cluster ID: {cluster}")  # debug (keep for now)
st.write(f"Trader Type: {cluster_map.get(cluster, 'Unknown')}")

# -------------------------------
# 💡 Explanation Section
# -------------------------------
st.markdown("### ℹ️ Interpretation")
st.write("""
- **Low-risk traders** → small trades, low fees, stable returns  
- **High-risk traders** → large trades, high volatility, high fees  
- **Medium-risk traders** → moderate activity and outcomes  
""")