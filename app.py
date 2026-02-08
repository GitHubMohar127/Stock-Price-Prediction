import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
# from config import STOCK_SYMBOLS

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

st.title("📈 Real-Time Stock Price Prediction")

# ---- STOCK LIST (Option 3) ----
STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL"]

# Sidebar
st.sidebar.header("Select Stock")
symbol = st.sidebar.selectbox("Stock Symbol", STOCK_SYMBOLS)

# Load data
data_path = f"data/{symbol}/processed_stock_data.csv"
model_path = f"models/{symbol}_model.pkl"

df = pd.read_csv(data_path, index_col=0, parse_dates=True)
model = joblib.load(model_path)

# Latest prediction
latest_features = df.drop(columns=["target"]).iloc[-1:]
prediction = model.predict(latest_features)[0]

st.metric(
    label="Tomorrow's Predicted Close Price",
    value=f"${prediction:.2f}"
)

# Price chart
st.subheader(f"{symbol} Closing Price Trend")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["close"],
    mode="lines",
    name="Actual Price"
))

st.plotly_chart(fig, use_container_width=True)

# # Feature importance
# st.subheader("Feature Importance")

# importance = model.feature_importances_
# features = latest_features.columns

# imp_df = pd.DataFrame({
#     "Feature": features,
#     "Importance": importance
# }).sort_values(by="Importance", ascending=False)

# st.bar_chart(imp_df.set_index("Feature"))
