import os
import argparse
import pandas as pd
from datetime import datetime
from nepsealpha_scraper import fetch_nepsealpha
from merolagani_scraper import fetch_stock_merolagani, fetch_nepse_merolagani

def merge_and_save(new_df: pd.DataFrame, filepath: str):
    if new_df.empty:
        print(f"  [Update] No new data to save for {filepath}")
        return

    # Convert Date to standard string format before saving
    new_df["Date"] = pd.to_datetime(new_df["Date"]).dt.strftime("%Y/%m/%d")

    if not os.path.exists(filepath):
        new_df.to_csv(filepath, index=False)
        print(f"  [Update] Saved {len(new_df)} rows to {filepath}")
        return

    try:
        old = pd.read_csv(filepath)
        combined = pd.concat([old, new_df], ignore_index=True)
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce").dt.strftime("%Y/%m/%d")
        combined.dropna(subset=["Date"], inplace=True)
        combined.drop_duplicates(subset=["Date"], keep="last", inplace=True)
        combined.sort_values("Date", inplace=True)
        combined.to_csv(filepath, index=False)
        print(f"  [Update] Merged data. Total: {len(combined)} rows in {filepath}")
    except Exception as e:
        print(f"  [Update] Failed to merge with existing data: {e}")
        new_df.to_csv(filepath, index=False)
        print(f"  [Update] Overwrote {filepath} with {len(new_df)} rows")

def update_symbol_data(symbol, pages="1", frame=1000, fsk=None):
    symbol = symbol.upper()
    outfile = os.path.join("data", "nepse_index_data.csv") if symbol == "NEPSE" else os.path.join("data", f"{symbol.lower()}_data.csv")

    print(f"\n{'='*55}")
    print(f"  Data Fetcher | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Symbol: {symbol}")
    print(f"{'='*55}")

    # 1. Try NepseAlpha (Primary, Ultra-Fast)
    print("-> Attempting NepseAlpha API...")
    df = fetch_nepsealpha(symbol, frame=frame, provided_fsk=fsk)

    # 2. Fallback to Merolagani
    if df.empty:
        print("-> NepseAlpha failed or blocked. Falling back to Merolagani scraper...")
        if symbol == "NEPSE":
            df = fetch_nepse_merolagani()
        else:
            # Default to 1 page on fallback to avoid huge wait times unless 'all' is requested
            pages_to_scrape = pages if pages.lower() == 'all' else 1
            df = fetch_stock_merolagani(symbol, pages_to_scrape=pages_to_scrape)

    if df.empty:
        print(f"-> [ERROR] Both scrapers failed for {symbol}.")
        return

    merge_and_save(df, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NEPSE data using Fallback Architecture")
    parser.add_argument("--symbol", type=str, default="NEPSE", help="Stock symbol e.g. LICN, NABIL, NEPSE")
    parser.add_argument("--pages", type=str, default="1", help="Pages for Merolagani fallback (e.g. 1, all)")
    parser.add_argument("--frame", type=int, default=1000, help="Candles for NepseAlpha (default: 1000)")
    parser.add_argument("--fsk", type=str, default=None, help="FSK session key for NepseAlpha")
    args = parser.parse_args()

    update_symbol_data(args.symbol, pages=args.pages, frame=args.frame, fsk=args.fsk)
