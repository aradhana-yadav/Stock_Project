import streamlit as st
import plotly.graph_objects as go
from main import run_all
from email_alert import send_email_alert

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Trading Dashboard", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.block-container {
    padding-top: 1.5rem;
}
.card {
    background: linear-gradient(145deg, #161B22, #1c222c);
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 6px 12px rgba(0,0,0,0.4);
}
.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return run_all()

results = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

stock_list = [res["stock"] for res in results]
selected_stock = st.sidebar.selectbox("📌 Select Stock", stock_list)

selected_data = next(res for res in results if res["stock"] == selected_stock)

# ---------------- HEADER ----------------
st.title("📊 AI Trading Dashboard")

# ---------------- SIGNAL ----------------
signal = selected_data["signal"]
color = "#00FF9C" if signal == "BUY" else "#FF4B4B" if signal == "SELL" else "#FFA500"

# ---------------- METRICS ----------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("Accuracy", f"{selected_data['accuracy']*100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("Strategy Return", f"{selected_data['strategy_return']:.2f}x")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("Market Return", f"{selected_data['market_return']:.2f}x")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:{color}'>Signal: {signal}</h3>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- CONTROL PANEL ----------------
st.markdown("---")
st.markdown('<div class="section-title">⚡ Controls Panel</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

# Alert
with c1:
    if st.button("📧 Send Alert"):
        send_email_alert(signal, selected_stock, "aaradhanay8@gmail.com")
        st.success("Alert Sent!")

# Backtesting dropdown
with c2:
    graph_type = st.selectbox("📊 Backtest View", [
        "Strategy vs Market",
        "Only Strategy",
        "Only Market"
    ])

# Graph style dropdown
with c3:
    chart_style = st.selectbox("📈 Chart Type", [
        "Line",
        "Area"
    ])

# ---------------- BACKTESTING METRICS ----------------
st.markdown("---")
st.markdown('<div class="section-title">📊 Backtesting Metrics</div>', unsafe_allow_html=True)

metrics = selected_data["metrics"]

m1, m2, m3 = st.columns(3)
m1.metric("Sharpe Ratio", f"{metrics['Sharpe']:.2f}")
m2.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2f}")
m3.metric("Win Rate", f"{metrics['Win Rate']*100:.2f}%")

# ---------------- CHARTS ----------------
st.markdown("---")
st.markdown('<div class="section-title">📈 Charts</div>', unsafe_allow_html=True)

df = selected_data["df"]

col1, col2 = st.columns(2)

# -------- PRICE CHART --------
with col1:
    fig1 = go.Figure()

    if chart_style == "Area":
        fig1.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            fill='tozeroy',
            name="Price"
        ))
    else:
        fig1.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name="Price"
        ))

    fig1.update_layout(
        title="Stock Price",
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(fig1, use_container_width=True)

# -------- BACKTEST GRAPH --------
with col2:
    fig2 = go.Figure()

    if graph_type == "Strategy vs Market":
        fig2.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Strategy'], name="Strategy"))
        fig2.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Market'], name="Market"))

    elif graph_type == "Only Strategy":
        fig2.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Strategy'], name="Strategy"))

    else:
        fig2.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Market'], name="Market"))

    fig2.update_layout(
        title="Backtesting Graph",
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- DATA TABLE ----------------
st.markdown("---")
st.markdown('<div class="section-title">📋 Recent Data</div>', unsafe_allow_html=True)

st.dataframe(df.tail(), use_container_width=True)