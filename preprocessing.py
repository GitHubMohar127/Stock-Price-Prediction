import pandas as pd
import os
from config import STOCK_SYMBOLS

for symbol in STOCK_SYMBOLS:
    print(f"Preprocessing {symbol}")

    path = f"data/{symbol}/stock_data.csv"
    if not os.path.exists(path):
        print(f"Missing data for {symbol}")
        continue

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = ["open", "high", "low", "close", "volume"]

    df["daily_return"] = df["close"].pct_change()
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["high_low_diff"] = df["high"] - df["low"]
    df["prev_close"] = df["close"].shift(1)
    df["target"] = df["close"].shift(-1)

    df.dropna(inplace=True)

    df.to_csv(f"data/{symbol}/processed_stock_data.csv")
