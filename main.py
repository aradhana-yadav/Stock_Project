import yfinance as yf
import pandas as pd
import numpy as np

def fetch_stock_data(stock):

    data = yf.download(stock, period="10y", interval="1d")

    if data.empty:
        return pd.DataFrame(), pd.Series(), data

    data = data.reset_index()

    # Moving averages
    data["MA10"] = data["Close"].rolling(10).mean()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()

    # Returns
    data["Return"] = data["Close"].pct_change()

    # RSI
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()

    data["MACD"] = ema12 - ema26
    data["MACD_signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # Volatility
    data["Volatility"] = data["Return"].rolling(10).std()

    # Volume change
    data["Volume_Change"] = data["Volume"].pct_change()

    # Momentum
    data["Momentum"] = data["Close"].pct_change(3)

    # Future return
    data["Future_Return"] = data["Close"].shift(-5) / data["Close"] - 1

    # BUY SELL HOLD
    def signal(x):
        if x > 0.02:
            return 2
        elif x < -0.02:
            return 0
        else:
            return 1

    data["Signal"] = data["Future_Return"].apply(signal)

    # Remove infinity
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove NaN
    data.dropna(inplace=True)

    if len(data) < 200:
        return pd.DataFrame(), pd.Series(), data

    features = data[
        [
            "MA10",
            "MA20",
            "MA50",
            "Return",
            "RSI",
            "MACD",
            "MACD_signal",
            "Volatility",
            "Volume_Change",
            "Momentum"
        ]
    ]

    target = data["Signal"]

    return features, target, data