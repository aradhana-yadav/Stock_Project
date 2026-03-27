import yfinance as yf
from model_training import train_model
from backtest import backtest
# Email alert triggered manually from dashboard
# from email_alert import send_email_alert

def run_all():
    stocks = [
        "ASIANPAINT.NS","TCS.NS","INFY.NS","RELIANCE.NS",
        "HDFCBANK.NS","ICICIBANK.NS","LT.NS","SBIN.NS",
        "ITC.NS","HINDUNILVR.NS"
    ]

    results = []
    user_email = "aaradhanay8@gmail.com"

    for stock in stocks:
        df = yf.download(stock, period="1y")
        df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)

        # Features
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['Return'] = df['Close'].pct_change()
        df['Signal'] = 0
        df.loc[df['MA10'] > df['MA50'], 'Signal'] = 1
        df.loc[df['MA10'] < df['MA50'], 'Signal'] = -1
        df.dropna(inplace=True)

        # Train model
        model, accuracy = train_model(df, stock)

        # Prediction
        latest = df[['MA10','MA50','Return']].iloc[-1:]
        pred = model.predict(latest)[0]
        signal = "BUY" if pred==1 else "SELL" if pred==-1 else "HOLD"

        # Backtesting
        strat_ret, market_ret, df, metrics = backtest(df)

        results.append({
            "stock": stock,
            "accuracy": accuracy,
            "signal": signal,
            "strategy_return": strat_ret,
            "market_return": market_ret,
            "df": df,
            "metrics": metrics
        })

    return results