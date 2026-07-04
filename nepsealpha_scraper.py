import time
import pandas as pd
from curl_cffi import requests

BASE_URL = "https://www.nepsealpha.com"
HISTORY_EP = f"{BASE_URL}/trading/1/history"
SYMBOLS_EP = f"{BASE_URL}/trading/1/symbols"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{BASE_URL}/trading/chart",
    "X-Requested-With": "XMLHttpRequest",
}

def get_fsk(session, symbol):
    print("  [NepseAlpha] Fetching FSK session key...")
    try:
        session.get(f"{BASE_URL}/trading/chart", timeout=15)
        time.sleep(1)
        r1 = session.get(SYMBOLS_EP, params={"symbol": symbol}, timeout=15)
        try:
            data = r1.json()
            fsk = data.get("fsk") or data.get("session_key") or data.get("key")
            if fsk: return fsk
        except:
            pass
        cookies = dict(session.cookies)
        fsk = cookies.get("fsk") or cookies.get("FSK")
        if fsk: return fsk
    except Exception as e:
        print(f"  [NepseAlpha] Auth failed: {e}")
    return None

def fetch_nepsealpha(symbol, frame=1000, provided_fsk=None):
    session = requests.Session(impersonate="chrome")
    session.headers.update(HEADERS)
    fsk = provided_fsk or get_fsk(session, symbol)
    
    params = {"symbol": symbol, "resolution": "1D", "frame": frame}
    if fsk:
        params["fsk"] = fsk
        
    print(f"  [NepseAlpha] Fetching {symbol} (frame={frame})...")
    try:
        r = session.get(HISTORY_EP, params=params, timeout=20)
        if r.status_code != 200:
            print(f"  [NepseAlpha] HTTP {r.status_code}")
            return pd.DataFrame()
            
        data = r.json()
        if data.get("s") != "ok":
            print(f"  [NepseAlpha] Status not ok: {data.get('s')}")
            return pd.DataFrame()
            
        df = pd.DataFrame({
            "Date": pd.to_datetime(data.get("t", []), unit="s").normalize(),
            "Open": [float(x) for x in data.get("o", [])],
            "High": [float(x) for x in data.get("h", [])],
            "Low": [float(x) for x in data.get("l", [])],
            "Close": [float(x) for x in data.get("c", [])],
            "Volume": [float(x) for x in data.get("v", [])],
        })
        df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
        print(f"  [NepseAlpha] ✓ {len(df)} candles retrieved.")
        return df
    except Exception as e:
        print(f"  [NepseAlpha] Request failed: {e}")
        return pd.DataFrame()
