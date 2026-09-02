"""
1. What is an API?  API = Application Programming Interface
        An API is a way for one software application to communicate with another software application.

2. Why are APIs used?
    APIs allow applications to access data and services without needing to know how the other application internally works.   
    examples:     
    | API              | Possible data                |
    | ---------------- | ---------------------------- |
    | Binance API      | Cryptocurrency trades/prices |
    | Google Maps API  | Maps/location information    |
    | Weather API      | Temperature/weather          |
    | GitHub API       | Repositories/users/issues    |
    | Stock market API | Stock prices                 |
    | Payment API      | Payment transactions         |
    | News API         | News articles                |
    | YouTube API      | Videos/channels              |
    For machine learning, 
    APIs are especially useful because they allow you to automatically collect data instead of manually downloading datasets.
    
3. What is API data extraction?
    API data extraction means:
    Sending a request to an API, receiving the response, converting the response into a usable format, 
    and storing or processing the data.    
        API DATA EXTRACTION
                                    Python Program
                                        │
                                        │ 1. Request
                                        ↓
                                    API Endpoint
                                        │
                                        │ 2. Server processes request
                                        ↓
                                    API Server
                                        │
                                        │ 3. Response
                                        ↓
                                    JSON Data
                                        │
                                        │ 4. Convert
                                        ↓
                                    Pandas DataFrame
                                        │
                                        ├── CSV
                                        ├── Database
                                        ├── Machine Learning
                                        └── Dashboard
"""


import requests
import pandas as pd

# API URL
url = "https://api.binance.com/api/v3/trades"

# Parameters
params = {
    "symbol": "BTCUSDT",
    "limit": 1000
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