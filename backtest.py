import joblib
import numpy as np
from main import fetch_stock_data

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

print("\nAI Strategy Backtesting Report\n")

for stock in stocks:

    X, y, data = fetch_stock_data(stock)

    if X.empty:
        print("No data for", stock)
        continue

    model_path = f"models/{stock.replace('.NS','')}_model.pkl"

    model = joblib.load(model_path)

    predictions = model.predict(X)

    data = data.iloc[-len(predictions):].copy()
    data["Predicted_Signal"] = predictions

    # Close prices ko 1D array bana do
    prices = np.array(data["Close"]).flatten()

    profit = 0.0
    trades = 0

    for i in range(len(prices) - 1):

        signal = data["Predicted_Signal"].iloc[i]

        today = float(prices[i])
        next_day = float(prices[i + 1])

        if signal == 2:  # BUY
            profit += next_day - today
            trades += 1

        elif signal == 0:  # SELL
            profit += today - next_day
            trades += 1

    latest_signal = int(data["Predicted_Signal"].iloc[-1])

    if latest_signal == 2:
        alert = "BUY"
    elif latest_signal == 0:
        alert = "SELL"
    else:
        alert = "HOLD"

    print("\nStock:", stock.replace(".NS", ""))
    print("Total Trades:", trades)
    print("Strategy Profit:", round(profit, 2))
    print("Current Signal:", alert)

print("\nBacktesting Completed for All Stocks")