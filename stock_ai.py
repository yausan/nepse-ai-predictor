import yfinance as yf
import pandas as pd

# Download 10 years of Apple stock data
ticker = "AAPL"
df = yf.download(ticker, start="2015-01-01", end="2025-01-01")

# Explore the data
print("=== First 5 rows ===")
print(df.head())

print("\n=== Last 5 rows ===")
print(df.tail())

print("\n=== Shape (rows, columns) ===")
print(df.shape)

print("\n=== Column names ===")
print(df.columns)