# stock_predictor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset (replace with your stock data CSV)
data = pd.read_csv("data/stock_data.csv")

# Check first few rows
print("Dataset preview:\n", data.head())

# Assume CSV has columns: Date, Open, High, Low, Close, Volume
data["Date"] = pd.to_datetime(data["Date"])
data["Day"] = data["Date"].dt.day
data["Month"] = data["Date"].dt.month
data["Year"] = data["Date"].dt.year

# Features (X) and Target (y)
X = data[["Open", "High", "Low", "Volume", "Day", "Month", "Year"]]
y = data["Close"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Plot
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label="Actual Price", color="blue")
plt.plot(predictions, label="Predicted Price", color="red")
plt.legend()
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()
print("Stock Price Predictor finished successfully!")
