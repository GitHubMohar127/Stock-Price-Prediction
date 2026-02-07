import os
import requests
import pandas as pd
from config import API_KEY, STOCK_SYMBOLS

BASE_URL = "https://www.alphavantage.co/query"

for symbol in STOCK_SYMBOLS:
    print(f"Fetching data for {symbol}")

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY,
        "outputsize": "compact"
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    time_series = data.get("Time Series (Daily)")
    if time_series is None:
        print(f"API error for {symbol}")
        continue

    df = pd.DataFrame.from_dict(time_series, orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    os.makedirs(f"data/{symbol}", exist_ok=True)
    df.to_csv(f"data/{symbol}/stock_data.csv")

    print(f"{symbol} data saved")
