import requests
import pandas as pd

# API URL
url = "https://api.binance.com/api/v3/trades"

# Parameters
params = {
    "symbol": "BTCUSDT",
    "limit": 2000
}

# Get data from API
response = requests.get(url, params=params)

# Convert response to JSON
data = response.json()

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# Convert columns
df["price"] = df["price"].astype(float)
df["qty"] = df["qty"].astype(float)
df["quoteQty"] = df["quoteQty"].astype(float)

# Convert time
df["time"] = pd.to_datetime(df["time"], unit="ms")

# Print data
print(df)

# Save data to CSV
df.to_csv("binance_trades.csv", index=False)

print("CSV file created successfully!")