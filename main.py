import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Download Stock Data
stock = "TCS.NS"
data = yf.download(stock, start="2023-01-01", end="2024-01-01")

print("Raw Data:")
print(data.head())

# Step 2: Clean Data
data = data.dropna()
data = data.reset_index()
data['Date'] = pd.to_datetime(data['Date'])

# Step 3: Feature Engineering
data['MA_10'] = data['Close'].rolling(10).mean()
data['MA_50'] = data['Close'].rolling(50).mean()
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(10).std()

data = data.dropna()

# Step 4: Save Clean Data
data.to_csv("clean_stock_data.csv", index=False)

print("Feature Engineering Completed")
print(data.head())

# Step 5: Visualization
plt.figure(figsize=(10,5))
plt.plot(data['Date'], data['Close'], label="Close Price")
plt.plot(data['Date'], data['MA_10'], label="MA 10")
plt.legend()
plt.show()
