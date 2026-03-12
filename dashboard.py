import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from main import fetch_stock_data

st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# ----------- DARK THEME STYLE -----------

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

h1,h2,h3{
color:white;
}

</style>
""",unsafe_allow_html=True)

# ----------- TITLE -----------

st.title("📈 AI Stock Trading Dashboard")
st.write("Machine Learning Based Buy / Sell Prediction System")

# ----------- STOCK LIST -----------

stocks = [
"ASIANPAINT.NS",
"BHARTIARTL.NS",
"HDFCBANK.NS",
"ICICIBANK.NS",
"INFY.NS",
"ITC.NS",
"LT.NS",
"RELIANCE.NS",
"SBIN.NS",
"TCS.NS"
]

stock = st.selectbox("Select Stock", stocks)

# ----------- LOAD DATA -----------

@st.cache_data
def load_data(stock):
    X, y, data = fetch_stock_data(stock)
    return X, data

X, data = load_data(stock)

# ----------- LOAD MODEL -----------

model_path = f"models/{stock.replace('.NS','')}_model.pkl"
model = joblib.load(model_path)

prediction = model.predict(X.tail(1))[0]

if prediction == 2:
    signal = "BUY"
elif prediction == 0:
    signal = "SELL"
else:
    signal = "HOLD"

# ----------- METRICS -----------

latest_price = float(data["Close"].iloc[-1])
prev_price = float(data["Close"].iloc[-2])

change = latest_price - prev_price
change_percent = (change / prev_price) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("AI Signal", signal)
col2.metric("Current Price", round(latest_price,2))
col3.metric("Daily Change", round(change,2))
col4.metric("Change %", round(change_percent,2))

st.divider()

# ----------- CHART SECTION -----------

col1, col2 = st.columns(2)

# PRICE + MOVING AVERAGE
with col1:

    st.subheader("📉 Price & Moving Average")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(data["Close"].tail(200), label="Close Price", linewidth=2)
    ax.plot(data["MA10"].tail(200), label="MA10", linewidth=2)
    ax.plot(data["MA20"].tail(200), label="MA20", linewidth=2)

    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")

    ax.tick_params(colors="white")

    ax.legend()

    ax.grid(alpha=0.3)

    st.pyplot(fig)

# RSI CHART
with col2:

    st.subheader("📊 RSI Indicator")

    fig2, ax2 = plt.subplots(figsize=(8,4))

    ax2.plot(data["RSI"].tail(200), linewidth=2)

    ax2.axhline(70, linestyle="--")
    ax2.axhline(30, linestyle="--")

    ax2.set_facecolor("#0E1117")
    fig2.patch.set_facecolor("#0E1117")

    ax2.tick_params(colors="white")

    ax2.grid(alpha=0.3)

    st.pyplot(fig2)

st.divider()

# ----------- FEATURE IMPORTANCE -----------

st.subheader("🧠 AI Feature Importance")

importance = model.feature_importances_
features = X.columns

imp_df = pd.DataFrame({
"Feature":features,
"Importance":importance
}).sort_values("Importance",ascending=False)

st.bar_chart(imp_df.set_index("Feature"))

st.divider()

# ----------- MARKET DATA -----------

st.subheader("📋 Recent Market Data")

st.dataframe(data.tail(20))

st.divider()

st.caption("AI Trading System | RandomForest ML Model | Streamlit Dashboard")