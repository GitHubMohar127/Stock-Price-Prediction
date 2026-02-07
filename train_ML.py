import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from config import STOCK_SYMBOLS
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)

for symbol in STOCK_SYMBOLS:
    print(f"Training model for {symbol}")

    path = f"data/{symbol}/processed_stock_data.csv"
    if not os.path.exists(path):
        print(f"No processed data for {symbol}")
        continue

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    X = df.drop(columns=["target"])
    y = df["target"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    joblib.dump(model, f"models/{symbol}_model.pkl")

    tomorrow = model.predict(X.iloc[-1:].values)[0]
    print(f"{symbol} RMSE: {rmse}")
    print(f"{symbol} Tomorrow prediction: {tomorrow}")
