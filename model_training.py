import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ==============================
# 1️⃣ Load Clean Data
# ==============================

df = pd.read_csv("clean_stock_data.csv")

df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

df.dropna(inplace=True)

# ==============================
# 2️⃣ Feature Engineering
# ==============================

df['Price_Change'] = df['Close'].pct_change()
df['MA_10'] = df['Close'].rolling(10).mean()
df['MA_20'] = df['Close'].rolling(20).mean()
df['MA_50'] = df['Close'].rolling(50).mean()

df['Volatility'] = df['Close'].rolling(10).std()

# RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Momentum
df['Momentum'] = df['Close'] - df['Close'].shift(5)

df.dropna(inplace=True)

# ==============================
# 3️⃣ NEW TARGET (Trend Based)
# ==============================

# Predict short-term trend instead of raw return
df['Target'] = (df['MA_10'] > df['MA_20']).astype(int)

print("Target Distribution:")
print(df['Target'].value_counts(normalize=True))

# ==============================
# 4️⃣ Define Features & Target
# ==============================

features = [
    'MA_10', 'MA_20', 'MA_50',
    'Volume',
    'Price_Change',
    'Volatility',
    'RSI',
    'Momentum'
]

X = df[features]
y = df['Target']

# ==============================
# 5️⃣ Train-Test Split (Time Based)
# ==============================

split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# ==============================
# 6️⃣ Scaling
# ==============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 7️⃣ XGBoost Model
# ==============================

model = XGBClassifier(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# ==============================
# 8️⃣ Prediction & Accuracy
# ==============================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)






# ==============================
# 9️⃣ Latest Signal Output
# ==============================

latest_prediction = model.predict(X_test[-1].reshape(1, -1))[0]

if latest_prediction == 1:
    signal = "BUY"
else:
    signal = "SELL"

print("Latest Trading Signal:", signal)

print("Last 5 Predictions:")
print(predictions[-5:])


print("Prediction Distribution:")
print(pd.Series(predictions).value_counts(normalize=True))