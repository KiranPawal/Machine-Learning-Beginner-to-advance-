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

# ---------------------------------------------------------------------------
# STEP 1: Fetch live data from the API
# ---------------------------------------------------------------------------

def fetch_recent_trades(symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
    ''' This function gets recent trades from Binance:    symbol = "BTCUSDT"
        means  BTC → Bitcoin     USDT → Tether       So you're analyzing Bitcoin/USDT trades.'''

    url = "https://api.binance.com/api/v3/trades"                    # This is the endpoint from which the program requests recent trades.
    params = {"symbol": symbol, "limit": limit}               # creates the parameters.

    response = requests.get(url, params=params, timeout=10)                    # The program waits at most about 10 seconds for the request
    response.raise_for_status()            # If the API returns an HTTP error, Python raises an exception.
    data = response.json()                #The API returns JSON: 

    df = pd.DataFrame(data)                   # converts it into a table.

    # Convert types — Convert columns to float: The API returns numbers as strings.
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    df["quoteQty"] = df["quoteQty"].astype(float) 
    df["time"] = pd.to_datetime(df["time"], unit="ms")               # Binance provides timestamps in milliseconds.

    print(data)
    return df

# ---------------------------------------------------------------------------
# STEP 2: Call the function
# ---------------------------------------------------------------------------

df = fetch_recent_trades("BTCUSDT", 1000)


# ---------------------------------------------------------------------------
# STEP 3: Print the data
# ---------------------------------------------------------------------------

print(df)


# ---------------------------------------------------------------------------
# STEP 4: Save the API data into CSV
# ---------------------------------------------------------------------------

df.to_csv("binance_trades.csv", index=False)

print("\nCSV file created successfully!")
print("File name: binance_trades.csv")