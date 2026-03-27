import numpy as np

def backtest(df):
    df = df.copy()

    df['Strategy'] = df['Signal'].shift(1) * df['Return']
    df['Cumulative_Market'] = (1 + df['Return']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy']).cumprod()

    # Metrics
    sharpe = np.mean(df['Strategy']) / np.std(df['Strategy']) * np.sqrt(252)
    drawdown = (df['Cumulative_Strategy'].cummax() - df['Cumulative_Strategy']).max()
    win_rate = (df['Strategy'] > 0).sum() / len(df)

    metrics = {"Sharpe": sharpe, "Max Drawdown": drawdown, "Win Rate": win_rate}

    return df['Cumulative_Strategy'].iloc[-1], df['Cumulative_Market'].iloc[-1], df, metrics