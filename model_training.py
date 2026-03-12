from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

from main import fetch_stock_data

if not os.path.exists("models"):
    os.makedirs("models")

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
    "TCS.NS",
]

print("\nProfessional 5-Day AI Trading Signals:\n")

for stock in stocks:

    X, y, data = fetch_stock_data(stock)

    if X.empty or y.empty:
        print(f"No training data for {stock}, skipping...")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=18,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nReport for {stock.replace('.NS','')}")
    print(classification_report(y_test, y_pred, zero_division=0))

    latest = X.tail(1)
    signal = model.predict(latest)[0]

    if signal == 2:
        signal_text = "BUY"
    elif signal == 0:
        signal_text = "SELL"
    else:
        signal_text = "HOLD"

    print(
        f"{stock.replace('.NS','')}: Accuracy = {accuracy:.2f} | Current Signal = {signal_text}"
    )

    model_path = f"models/{stock.replace('.NS','')}_model.pkl"
    joblib.dump(model, model_path)

print("\nAll models trained and saved successfully.")