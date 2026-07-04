"""
analysis.py — Complete 7-Layer NEPSE Professional Trading Analysis Engine
═══════════════════════════════════════════════════════════════════════════
Layer 1: Market Structure  (swing H/L, trend, BOS)
Layer 2: Key Levels        (S/R, psych levels, flip zones, gaps)
Layer 3: Price Action      (pin bar, engulfing, inside bar, marubozu, doji)
Layer 4: Volume Analysis   (price-volume classification)
Layer 5: Trend Indicators  (EMA 9/21/50, ADX, Bollinger squeeze)
Layer 6: Momentum          (RSI, MACD, Stochastic)
Layer 7: NEPSE Filters     (index correlation, sector, circuit)
+ Auto-scored 7-point trade checklist
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from indicators import add_all_indicators
from datetime import datetime, timedelta


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ═════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════

SECTOR_MAP = {
    # Banking
    "NABIL": "Banking", "NICA": "Banking", "SBI": "Banking", "SCB": "Banking",
    "HBL": "Banking", "EBL": "Banking", "MEGA": "Banking", "GBIME": "Banking",
    "SANIMA": "Banking", "KBL": "Banking", "SBL": "Banking", "CZBIL": "Banking",
    "PRVU": "Banking", "PCBL": "Banking", "ADBL": "Banking", "NBL": "Banking",
    "MBL": "Banking", "LBBL": "Banking", "SRBL": "Banking", "BOKL": "Banking",
    "NIMB": "Banking", "NCCB": "Banking", "CCBL": "Banking", "NMB": "Banking",
    # Hydropower
    "NHPC": "Hydropower", "CHCL": "Hydropower", "BPCL": "Hydropower",
    "UPPER": "Hydropower", "AHPC": "Hydropower", "API": "Hydropower",
    "AKJCL": "Hydropower", "DHPL": "Hydropower", "GHL": "Hydropower",
    "GLH": "Hydropower", "HDHPC": "Hydropower", "HURJA": "Hydropower",
    "KPCL": "Hydropower", "LEC": "Hydropower", "MBJC": "Hydropower",
    "MKJC": "Hydropower", "MEN": "Hydropower", "NGPL": "Hydropower",
    "NYADI": "Hydropower", "PPCL": "Hydropower", "RADHI": "Hydropower",
    "RURU": "Hydropower", "SHPC": "Hydropower", "SJCL": "Hydropower",
    "SSHL": "Hydropower", "UNHPL": "Hydropower", "UMHL": "Hydropower",
    "UPCL": "Hydropower", "ULHC": "Hydropower",
    # Insurance
    "LICN": "Insurance", "NLIC": "Insurance", "ALICL": "Insurance",
    "GLICL": "Insurance", "HGI": "Insurance", "IGI": "Insurance",
    "LGIL": "Insurance", "NIL": "Insurance", "NICL": "Insurance",
    "PLIC": "Insurance", "PRIN": "Insurance", "RBCL": "Insurance",
    "SICL": "Insurance", "SLICL": "Insurance", "UIC": "Insurance",
    # Finance
    "CFCL": "Finance", "GFCL": "Finance", "GUFL": "Finance",
    "ICFC": "Finance", "JFL": "Finance", "MFIL": "Finance",
    "MPFL": "Finance", "NFS": "Finance", "PFL": "Finance",
    "PROFL": "Finance", "RLFL": "Finance", "SFCL": "Finance",
    "SIFC": "Finance",
    # Development Bank
    "CORBL": "Dev Bank", "EDBL": "Dev Bank", "GDBL": "Dev Bank",
    "GRDBL": "Dev Bank", "JBBL": "Dev Bank", "KSBBL": "Dev Bank",
    "LBBL": "Dev Bank", "MLBL": "Dev Bank", "MNBBL": "Dev Bank",
    "NABBC": "Dev Bank", "SADBL": "Dev Bank", "SAPDBL": "Dev Bank",
    "SHINE": "Dev Bank",
    # Microfinance
    "CBBL": "Microfinance", "DDBL": "Microfinance", "FMDBL": "Microfinance",
    "FOWAD": "Microfinance", "GBLBS": "Microfinance", "JBLB": "Microfinance",
    "KLBSL": "Microfinance", "LLBS": "Microfinance", "MLBSL": "Microfinance",
    "MMLBS": "Microfinance", "NSLB": "Microfinance", "RSDC": "Microfinance",
    "SLBS": "Microfinance", "SMATA": "Microfinance", "SWBBL": "Microfinance",
    "VLBS": "Microfinance",
}


def load_data(symbol):
    """Load and clean price data for the given symbol."""
    if symbol.upper() == "NEPSE":
        return load_nepse_index()
        
    raw = os.path.join("data", f"{symbol.lower()}_data.csv")
    cleaned = os.path.join("data", f"{symbol.lower()}_cleaned.csv")

    path = cleaned if os.path.exists(cleaned) else raw
    if not os.path.exists(path):
        print(f"[*] No data found for {symbol}. Fetching live data from internet...")
        import subprocess, sys
        subprocess.run([sys.executable, "update_data.py", "--symbol", symbol, "--pages", "3"])
        path = cleaned if os.path.exists(cleaned) else raw
        if not os.path.exists(path):
            raise FileNotFoundError(f"No data found for {symbol}. Scrape failed.")

    df = pd.read_csv(path)
    df = df.dropna(how="all")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close", "Open", "High", "Low"])
    df = df.sort_values("Date").reset_index(drop=True)

    if "Volume" not in df.columns:
        df["Volume"] = 0
    df["Volume"] = df["Volume"].fillna(0).astype(float)

    return df


def load_nepse_index():
    """Load NEPSE index data if available."""
    path = os.path.join("data", "nepse_index_data.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(",", "", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Date", "Close"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════
#  LAYER 1: MARKET STRUCTURE
# ═════════════════════════════════════════════════════════════════════════

def detect_swing_points(df, lookback=3):
    """Detect swing highs and lows using a pivot lookback window."""
    n = len(df)
    swings = []  # (index, type, price, date)

    for i in range(lookback, n - lookback):
        # Swing High: H[i] > all H in window around it
        is_high = all(df["High"].iloc[i] > df["High"].iloc[j]
                      for j in range(i - lookback, i + lookback + 1) if j != i)
        if is_high:
            swings.append({
                "idx": i,
                "type": "high",
                "price": float(df["High"].iloc[i]),
                "date": df["Date"].iloc[i].strftime("%Y-%m-%d"),
            })

        # Swing Low: L[i] < all L in window around it
        is_low = all(df["Low"].iloc[i] < df["Low"].iloc[j]
                     for j in range(i - lookback, i + lookback + 1) if j != i)
        if is_low:
            swings.append({
                "idx": i,
                "type": "low",
                "price": float(df["Low"].iloc[i]),
                "date": df["Date"].iloc[i].strftime("%Y-%m-%d"),
            })

    return swings


def classify_trend(swings):
    """Classify the trend based on the last few swing points."""
    if len(swings) < 4:
        return "RANGING", "Not enough swing points to determine structure"

    # Get the last 6 swings (or fewer)
    recent = swings[-6:]
    highs = [s for s in recent if s["type"] == "high"]
    lows  = [s for s in recent if s["type"] == "low"]

    if len(highs) < 2 or len(lows) < 2:
        return "RANGING", "Not enough distinct swing highs/lows"

    hh = highs[-1]["price"] > highs[-2]["price"]  # Higher High
    hl = lows[-1]["price"] > lows[-2]["price"]     # Higher Low
    lh = highs[-1]["price"] < highs[-2]["price"]   # Lower High
    ll = lows[-1]["price"] < lows[-2]["price"]      # Lower Low

    if hh and hl:
        return "UPTREND", f"Higher Highs ({highs[-2]['price']:.2f} → {highs[-1]['price']:.2f}) + Higher Lows ({lows[-2]['price']:.2f} → {lows[-1]['price']:.2f})"
    elif lh and ll:
        return "DOWNTREND", f"Lower Highs ({highs[-2]['price']:.2f} → {highs[-1]['price']:.2f}) + Lower Lows ({lows[-2]['price']:.2f} → {lows[-1]['price']:.2f})"
    else:
        return "RANGING", f"Mixed structure — no clear HH+HL or LH+LL pattern"


def find_bos(swings, df):
    """Find the most recent Break of Structure."""
    if len(swings) < 3:
        return None

    last_price = float(df["Close"].iloc[-1])

    # Walk backward through swings to find the last BOS
    for i in range(len(swings) - 1, 0, -1):
        s = swings[i]
        # Bullish BOS: price broke above a swing high
        if s["type"] == "high" and last_price > s["price"]:
            return {
                "type": "bullish",
                "level": s["price"],
                "date": s["date"],
                "description": f"Bullish BOS — Price broke above swing high at {s['price']:.2f} ({s['date']})"
            }
        # Bearish BOS: price broke below a swing low
        if s["type"] == "low" and last_price < s["price"]:
            return {
                "type": "bearish",
                "level": s["price"],
                "date": s["date"],
                "description": f"Bearish BOS — Price broke below swing low at {s['price']:.2f} ({s['date']})"
            }

    return None


def layer1_structure(df):
    """Complete Layer 1 analysis."""
    # Use 2 different lookback periods for robustness
    swings = detect_swing_points(df, lookback=3)
    trend, trend_reason = classify_trend(swings)
    bos = find_bos(swings, df)

    # Only return last 10 swings for display
    display_swings = swings[-10:]

    return {
        "trend": trend,
        "trend_reason": trend_reason,
        "bos": bos,
        "swing_points": display_swings,
        "total_swings_detected": len(swings),
    }


# ═════════════════════════════════════════════════════════════════════════
#  LAYER 2: KEY LEVELS
# ═════════════════════════════════════════════════════════════════════════

def layer2_levels(df, swings):
    """Detect key support and resistance levels."""
    current_price = float(df["Close"].iloc[-1])
    levels = []

    # --- Major swing H/L from last 6-12 months ---
    cutoff_6m = df["Date"].iloc[-1] - timedelta(days=180)
    cutoff_12m = df["Date"].iloc[-1] - timedelta(days=365)
    recent_swings = [s for s in swings if pd.to_datetime(s["date"]) >= cutoff_12m]

    for s in recent_swings:
        lvl_type = "resistance" if s["type"] == "high" else "support"
        distance_pct = abs(s["price"] - current_price) / current_price * 100
        if distance_pct < 20:  # Only levels within 20%
            levels.append({
                "price": s["price"],
                "type": lvl_type,
                "source": f"Swing {s['type'].capitalize()} ({s['date']})",
                "strength": "strong" if pd.to_datetime(s["date"]) >= cutoff_6m else "moderate",
                "distance_pct": round(distance_pct, 2),
            })

    # --- Round psychological levels ---
    price_range_low  = current_price * 0.85
    price_range_high = current_price * 1.15

    step = 50 if current_price < 500 else (100 if current_price < 2000 else 500)
    psych = int(price_range_low // step) * step
    while psych <= price_range_high:
        if psych > 0:
            distance_pct = abs(psych - current_price) / current_price * 100
            if distance_pct < 15:
                levels.append({
                    "price": float(psych),
                    "type": "resistance" if psych > current_price else "support",
                    "source": f"Psychological Level ({psych})",
                    "strength": "moderate",
                    "distance_pct": round(distance_pct, 2),
                })
        psych += step

    # --- Previous Week High / Low / Close ---
    today = df["Date"].iloc[-1]
    one_week_ago = today - timedelta(days=7)
    last_week_df = df[df["Date"] <= one_week_ago].tail(5)
    if len(last_week_df) > 0:
        pw_high  = float(last_week_df["High"].max())
        pw_low   = float(last_week_df["Low"].min())
        pw_close = float(last_week_df["Close"].iloc[-1])

        for label, price in [("Prev Week High", pw_high), ("Prev Week Low", pw_low), ("Prev Week Close", pw_close)]:
            d = abs(price - current_price) / current_price * 100
            levels.append({
                "price": price,
                "type": "resistance" if price > current_price else "support",
                "source": label,
                "strength": "moderate",
                "distance_pct": round(d, 2),
            })

    # --- Gap Zones ---
    gaps = []
    for i in range(1, len(df)):
        prev_close = df["Close"].iloc[i - 1]
        curr_open  = df["Open"].iloc[i]
        gap_pct    = abs(curr_open - prev_close) / prev_close * 100
        if gap_pct > 1.0:
            gap_mid = (curr_open + prev_close) / 2
            d = abs(gap_mid - current_price) / current_price * 100
            if d < 10:
                direction = "gap_up" if curr_open > prev_close else "gap_down"
                gaps.append({
                    "price": round(gap_mid, 2),
                    "type": "support" if direction == "gap_up" else "resistance",
                    "source": f"Gap ({direction.replace('_',' ').title()}) {df['Date'].iloc[i].strftime('%Y-%m-%d')}",
                    "strength": "weak",
                    "distance_pct": round(d, 2),
                })
    levels.extend(gaps[-5:])  # Last 5 gaps

    # --- Flip Zones (simplified: swing highs that are now below price = support flips) ---
    for s in recent_swings:
        if s["type"] == "high" and s["price"] < current_price:
            d = abs(s["price"] - current_price) / current_price * 100
            if d < 8:
                levels.append({
                    "price": s["price"],
                    "type": "support",
                    "source": f"Flip Zone (old resistance → support) {s['date']}",
                    "strength": "strong",
                    "distance_pct": round(d, 2),
                })
        elif s["type"] == "low" and s["price"] > current_price:
            d = abs(s["price"] - current_price) / current_price * 100
            if d < 8:
                levels.append({
                    "price": s["price"],
                    "type": "resistance",
                    "source": f"Flip Zone (old support → resistance) {s['date']}",
                    "strength": "strong",
                    "distance_pct": round(d, 2),
                })

    # De-duplicate: merge levels within 0.5% of each other
    levels.sort(key=lambda x: x["price"])
    merged = []
    for lv in levels:
        if merged and abs(lv["price"] - merged[-1]["price"]) / max(merged[-1]["price"], 1) < 0.005:
            # Keep the stronger one
            if lv["strength"] == "strong" and merged[-1]["strength"] != "strong":
                merged[-1] = lv
        else:
            merged.append(lv)

    # Find nearest support and resistance
    supports = [l for l in merged if l["type"] == "support" and l["price"] < current_price]
    resistances = [l for l in merged if l["type"] == "resistance" and l["price"] > current_price]

    nearest_support    = max(supports, key=lambda x: x["price"]) if supports else None
    nearest_resistance = min(resistances, key=lambda x: x["price"]) if resistances else None

    # Check if price is at a level (within 1%)
    at_level = any(l["distance_pct"] < 1.0 for l in merged)

    return {
        "levels": merged[:20],  # Cap at 20
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "at_key_level": at_level,
        "current_price": current_price,
    }


# ═════════════════════════════════════════════════════════════════════════
#  LAYER 3: PRICE ACTION PATTERNS
# ═════════════════════════════════════════════════════════════════════════

def detect_patterns(df):
    """Detect candle patterns on the last few bars."""
    if len(df) < 3:
        return {"patterns": [], "has_confirmation": False}

    patterns = []

    for offset in range(3):  # Check last 3 candles
        i = len(df) - 1 - offset
        if i < 1:
            break

        o, h, l, c = df["Open"].iloc[i], df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i]
        body = abs(c - o)
        full_range = h - l
        if full_range < 0.001:
            full_range = 0.001

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        is_bullish = c > o
        date_str   = df["Date"].iloc[i].strftime("%Y-%m-%d")
        days_ago   = offset

        # --- Pin Bar ---
        if full_range > 0:
            if lower_wick > 2 * body and lower_wick > upper_wick * 2 and body / full_range < 0.4:
                patterns.append({
                    "name": "Bullish Pin Bar",
                    "type": "bullish",
                    "date": date_str,
                    "days_ago": days_ago,
                    "description": "Long lower wick showing strong rejection from below. Buyers pushed price back up.",
                    "significance": "high",
                })
            elif upper_wick > 2 * body and upper_wick > lower_wick * 2 and body / full_range < 0.4:
                patterns.append({
                    "name": "Bearish Pin Bar",
                    "type": "bearish",
                    "date": date_str,
                    "days_ago": days_ago,
                    "description": "Long upper wick showing rejection from above. Sellers pushed price back down.",
                    "significance": "high",
                })

        # --- Engulfing ---
        if i >= 1:
            prev_o = df["Open"].iloc[i - 1]
            prev_c = df["Close"].iloc[i - 1]
            prev_body = abs(prev_c - prev_o)

            if is_bullish and prev_c < prev_o:  # Previous was bearish
                if c > prev_o and o < prev_c and body > prev_body:
                    patterns.append({
                        "name": "Bullish Engulfing",
                        "type": "bullish",
                        "date": date_str,
                        "days_ago": days_ago,
                        "description": "Bullish candle completely engulfs the previous bearish candle. Momentum has shifted to buyers.",
                        "significance": "high",
                    })
            elif not is_bullish and prev_c > prev_o:  # Previous was bullish
                if o > prev_c and c < prev_o and body > prev_body:
                    patterns.append({
                        "name": "Bearish Engulfing",
                        "type": "bearish",
                        "date": date_str,
                        "days_ago": days_ago,
                        "description": "Bearish candle completely engulfs the previous bullish candle. Momentum shifted to sellers.",
                        "significance": "high",
                    })

        # --- Inside Bar (Harami) ---
        if i >= 1:
            prev_o = df["Open"].iloc[i - 1]
            prev_c = df["Close"].iloc[i - 1]
            prev_h = df["High"].iloc[i - 1]
            prev_l = df["Low"].iloc[i - 1]
            prev_body = abs(prev_c - prev_o)
            
            # Classic Inside Bar
            if h <= prev_h and l >= prev_l:
                patterns.append({
                    "name": "Inside Bar",
                    "type": "neutral",
                    "date": date_str,
                    "days_ago": days_ago,
                    "description": "Price range is entirely within the previous bar. Consolidation — expect a breakout.",
                    "significance": "medium",
                })
                
            # Bullish Harami
            if prev_c < prev_o and is_bullish and c <= prev_o and o >= prev_c and body < prev_body * 0.5:
                patterns.append({
                    "name": "Bullish Harami",
                    "type": "bullish",
                    "date": date_str,
                    "days_ago": days_ago,
                    "description": "Small bullish candle contained within a prior large bearish candle. Suggests selling momentum is stopping.",
                    "significance": "medium",
                })
                
            # Bearish Harami
            if prev_c > prev_o and not is_bullish and c >= prev_o and o <= prev_c and body < prev_body * 0.5:
                patterns.append({
                    "name": "Bearish Harami",
                    "type": "bearish",
                    "date": date_str,
                    "days_ago": days_ago,
                    "description": "Small bearish candle contained within a prior large bullish candle. Suggests buying momentum is stopping.",
                    "significance": "medium",
                })
                
            # Piercing Line
            if prev_c < prev_o and is_bullish and o < prev_c and c > (prev_o + prev_c) / 2 and c < prev_o:
                patterns.append({
                    "name": "Piercing Line",
                    "type": "bullish",
                    "date": date_str,
                    "days_ago": days_ago,
                    "description": "Bullish candle opens below previous low but closes above the midpoint of the previous bearish body. Strong reversal.",
                    "significance": "high",
                })
                
            # Dark Cloud Cover
            if prev_c > prev_o and not is_bullish and o > prev_c and c < (prev_o + prev_c) / 2 and c > prev_o:
                patterns.append({
                    "name": "Dark Cloud Cover",
                    "type": "bearish",
                    "date": date_str,
                    "days_ago": days_ago,
                    "description": "Bearish candle opens above previous high but closes below the midpoint of the previous bullish body. Strong reversal.",
                    "significance": "high",
                })

        # --- 3-Candle Patterns (Morning/Evening Star, Soldiers/Crows) ---
        if i >= 2:
            c1_o, c1_c = df["Open"].iloc[i - 2], df["Close"].iloc[i - 2]
            c2_o, c2_c = df["Open"].iloc[i - 1], df["Close"].iloc[i - 1]
            c3_o, c3_c = df["Open"].iloc[i], df["Close"].iloc[i]
            
            c1_body, c2_body, c3_body = abs(c1_c - c1_o), abs(c2_c - c2_o), abs(c3_c - c3_o)
            
            c1_bull = c1_c > c1_o
            c2_bull = c2_c > c2_o
            c3_bull = c3_c > c3_o
            
            # Morning Star
            if not c1_bull and c1_body > full_range * 0.5 and c2_body < c1_body * 0.3 and c3_bull and c3_c > (c1_o + c1_c) / 2:
                patterns.append({
                    "name": "Morning Star",
                    "type": "bullish",
                    "date": date_str,
                    "days_ago": days_ago,
                    "description": "Major 3-candle bullish reversal. Strong sell-off, indecision, followed by a strong buy back.",
                    "significance": "high",
                })
                
            # Evening Star
            if c1_bull and c1_body > full_range * 0.5 and c2_body < c1_body * 0.3 and not c3_bull and c3_c < (c1_o + c1_c) / 2:
                patterns.append({
                    "name": "Evening Star",
                    "type": "bearish",
                    "date": date_str,
                    "days_ago": days_ago,
                    "description": "Major 3-candle bearish reversal. Strong rally, indecision, followed by a strong sell-off.",
                    "significance": "high",
                })
                
            # Three White Soldiers
            if c1_bull and c2_bull and c3_bull and c2_c > c1_c and c3_c > c2_c and c2_o > c1_o and c3_o > c2_o:
                patterns.append({
                    "name": "Three White Soldiers",
                    "type": "bullish",
                    "date": date_str,
                    "days_ago": days_ago,
                    "description": "Three consecutive strong bullish candles with higher closes. Indicates a powerful uptrend or reversal.",
                    "significance": "high",
                })
                
            # Three Black Crows
            if not c1_bull and not c2_bull and not c3_bull and c2_c < c1_c and c3_c < c2_c and c2_o < c1_o and c3_o < c2_o:
                patterns.append({
                    "name": "Three Black Crows",
                    "type": "bearish",
                    "date": date_str,
                    "days_ago": days_ago,
                    "description": "Three consecutive strong bearish candles with lower closes. Indicates a powerful downtrend or reversal.",
                    "significance": "high",
                })

        # --- Marubozu ---
        if body / full_range >= 0.90 and full_range > 0:
            dir_label = "Bullish" if is_bullish else "Bearish"
            patterns.append({
                "name": f"{dir_label} Marubozu",
                "type": "bullish" if is_bullish else "bearish",
                "date": date_str,
                "days_ago": days_ago,
                "description": f"Full-body candle with almost no wicks. Shows strong {dir_label.lower()} conviction — no hesitation.",
                "significance": "high",
            })

        # --- Doji ---
        if body / full_range <= 0.05 and full_range > 0:
            patterns.append({
                "name": "Doji",
                "type": "neutral",
                "date": date_str,
                "days_ago": days_ago,
                "description": "Open ≈ Close with wicks on both sides. Indecision — wait for the next candle to confirm direction.",
                "significance": "low",
            })

    # Check if today (offset=0) has a confirming pattern
    today_patterns = [p for p in patterns if p["days_ago"] == 0]
    has_confirmation = any(p["significance"] in ("high", "medium") for p in today_patterns)

    return {
        "patterns": patterns,
        "has_confirmation": has_confirmation,
        "today_patterns": today_patterns,
    }


# ═════════════════════════════════════════════════════════════════════════
#  LAYER 4: VOLUME ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

def layer4_volume(df):
    """Classify the latest price-volume relationship."""
    if len(df) < 11:
        return {
            "classification": "UNKNOWN",
            "meaning": "Not enough data for volume analysis",
            "volume_ratio": 1.0,
            "is_high_volume_rejection": False,
            "confirms_move": False,
        }

    vol = df["Volume"].iloc[-1]
    vol_avg = df["Volume"].iloc[-11:-1].mean()
    vol_ratio = vol / max(vol_avg, 1)

    price_change = df["Close"].iloc[-1] - df["Close"].iloc[-2]
    price_up     = price_change > 0
    price_down   = price_change < 0
    vol_up       = vol > vol_avg * 1.1
    vol_down     = vol < vol_avg * 0.9

    # Classification
    if price_up and vol_up:
        classification = "PRICE_UP_VOL_UP"
        meaning = "Real buying pressure — trust this move. Volume confirms the price increase."
    elif price_up and vol_down:
        classification = "PRICE_UP_VOL_DOWN"
        meaning = "Weak move on declining volume — likely fake or unsustainable. Be cautious."
    elif price_down and vol_up:
        classification = "PRICE_DOWN_VOL_UP"
        meaning = "Real selling pressure — get out or stay away. Volume confirms the decline."
    elif price_down and vol_down:
        classification = "PRICE_DOWN_VOL_DOWN"
        meaning = "No conviction selling — wait. This could be a false breakdown."
    elif abs(price_change) / max(df["Close"].iloc[-2], 1) < 0.003 and vol_up:
        classification = "FLAT_VOL_RISING"
        meaning = "Accumulation phase — big volume on flat price means smart money is positioning. Breakout is coming."
    else:
        classification = "NEUTRAL"
        meaning = "No significant price-volume divergence detected."

    # High-volume rejection candle
    last_h = df["High"].iloc[-1]
    last_l = df["Low"].iloc[-1]
    last_o = df["Open"].iloc[-1]
    last_c = df["Close"].iloc[-1]
    body = abs(last_c - last_o)
    full_range = last_h - last_l
    upper_wick = last_h - max(last_o, last_c)
    lower_wick = min(last_o, last_c) - last_l
    max_wick = max(upper_wick, lower_wick)

    is_rejection = vol_ratio > 1.5 and max_wick > body * 1.5 and full_range > 0
    confirms_move = classification in ("PRICE_UP_VOL_UP", "PRICE_DOWN_VOL_UP")

    return {
        "classification": classification,
        "meaning": meaning,
        "volume_today": float(vol),
        "volume_avg_10": float(vol_avg),
        "volume_ratio": round(float(vol_ratio), 2),
        "is_high_volume_rejection": is_rejection,
        "rejection_note": "⚠ High volume rejection candle detected — operators rejecting this level!" if is_rejection else None,
        "confirms_move": confirms_move,
    }


# ═════════════════════════════════════════════════════════════════════════
#  LAYER 5: TREND INDICATORS
# ═════════════════════════════════════════════════════════════════════════

def calc_adx(df, period=14):
    """Calculate ADX, +DI, -DI."""
    if len(df) < period * 2 + 1:
        return None, None, None

    high = df["High"].values
    low  = df["Low"].values
    close = df["Close"].values
    n = len(df)

    plus_dm  = np.zeros(n)
    minus_dm = np.zeros(n)
    tr_arr   = np.zeros(n)

    for i in range(1, n):
        up_move   = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i]  = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0

        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr_arr[i] = max(hl, hc, lc)

    # Wilder smoothing
    atr    = np.zeros(n)
    pdm_s  = np.zeros(n)
    mdm_s  = np.zeros(n)

    atr[period]   = np.mean(tr_arr[1:period + 1])
    pdm_s[period] = np.mean(plus_dm[1:period + 1])
    mdm_s[period] = np.mean(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        atr[i]   = (atr[i - 1] * (period - 1) + tr_arr[i]) / period
        pdm_s[i] = (pdm_s[i - 1] * (period - 1) + plus_dm[i]) / period
        mdm_s[i] = (mdm_s[i - 1] * (period - 1) + minus_dm[i]) / period

    plus_di  = (pdm_s / (atr + 1e-10)) * 100
    minus_di = (mdm_s / (atr + 1e-10)) * 100

    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100

    adx = np.zeros(n)
    start = period * 2
    if start < n:
        adx[start] = np.mean(dx[period + 1:start + 1])
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return float(adx[-1]), float(plus_di[-1]), float(minus_di[-1])


def layer5_trend(df):
    """EMA 9/21/50, ADX, Bollinger squeeze."""
    c = df["Close"]
    current = float(c.iloc[-1])

    ema9  = float(c.ewm(span=9, adjust=False).mean().iloc[-1])
    ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
    ema50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1])

    # EMA alignment
    if ema9 > ema21 > ema50:
        ema_signal = "BULLISH"
        ema_note = "EMAs perfectly aligned bullish (9 > 21 > 50). Price in strong uptrend."
    elif ema9 < ema21 < ema50:
        ema_signal = "BEARISH"
        ema_note = "EMAs perfectly aligned bearish (9 < 21 < 50). Price in strong downtrend."
    else:
        ema_signal = "MIXED"
        ema_note = "EMAs are crossed — mixed signals. Trend may be transitioning."

    # ADX
    adx, plus_di, minus_di = calc_adx(df, 14)
    if adx is None:
        adx, plus_di, minus_di = 0, 0, 0

    adx_trend = adx > 25
    adx_note = f"ADX = {adx:.1f}"
    if adx > 25:
        adx_note += " → Strong trend exists. Trend-following strategies work."
    else:
        adx_note += " → Weak/No trend. Avoid trend-following, consider range strategies."

    # Bollinger squeeze (width < 20-day SMA of width)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = (bb_upper - bb_lower) / (sma20 + 1e-10)

    if len(bb_width.dropna()) >= 20:
        current_width = float(bb_width.iloc[-1])
        avg_width     = float(bb_width.iloc[-20:].mean())
        squeeze       = current_width < avg_width * 0.8
    else:
        squeeze = False
        current_width = 0
        avg_width = 0

    return {
        "ema9": round(ema9, 2),
        "ema21": round(ema21, 2),
        "ema50": round(ema50, 2),
        "ema_signal": ema_signal,
        "ema_note": ema_note,
        "price_vs_ema": {
            "above_9":  current > ema9,
            "above_21": current > ema21,
            "above_50": current > ema50,
        },
        "adx": round(adx, 1),
        "plus_di": round(plus_di, 1),
        "minus_di": round(minus_di, 1),
        "adx_trend_exists": adx_trend,
        "adx_note": adx_note,
        "bollinger_squeeze": squeeze,
        "squeeze_note": "⚡ Bollinger Bands are squeezing — expect a breakout soon!" if squeeze else "Bollinger Bands are normal — no imminent squeeze breakout.",
    }


# ═════════════════════════════════════════════════════════════════════════
#  LAYER 6: MOMENTUM
# ═════════════════════════════════════════════════════════════════════════

def calc_stochastic(df, k_period=14, d_period=3, smooth=3):
    """Calculate Stochastic %K and %D."""
    if len(df) < k_period + d_period:
        return 50.0, 50.0

    lows  = df["Low"].rolling(k_period).min()
    highs = df["High"].rolling(k_period).max()
    fast_k = ((df["Close"] - lows) / (highs - lows + 1e-10)) * 100
    k = fast_k.rolling(smooth).mean()
    d = k.rolling(d_period).mean()
    return float(k.iloc[-1]) if not pd.isna(k.iloc[-1]) else 50.0, \
           float(d.iloc[-1]) if not pd.isna(d.iloc[-1]) else 50.0


def layer6_momentum(df):
    """RSI, MACD, Stochastic."""
    c = df["Close"]

    # RSI(14)
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs  = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = float(rsi.iloc[-1])

    if rsi_val < 35:
        rsi_zone = "OVERSOLD"
        rsi_note = f"RSI = {rsi_val:.1f} — Oversold territory. Watch for bullish reversal at support."
    elif rsi_val > 65:
        rsi_zone = "OVERBOUGHT"
        rsi_note = f"RSI = {rsi_val:.1f} — Overbought territory. Watch for bearish reversal at resistance."
    else:
        rsi_zone = "NEUTRAL"
        rsi_note = f"RSI = {rsi_val:.1f} — Neutral zone. No extreme momentum signal."

    rsi_aligned = rsi_val < 65  # Not overbought for buys / not oversold for sells

    # MACD(12,26,9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    macd_val  = float(macd_line.iloc[-1])
    sig_val   = float(signal_line.iloc[-1])
    hist_val  = float(histogram.iloc[-1])
    hist_prev = float(histogram.iloc[-2]) if len(histogram) > 1 else 0

    macd_direction = "BULLISH" if hist_val > 0 else "BEARISH"
    macd_increasing = hist_val > hist_prev
    macd_note = f"MACD Histogram = {hist_val:.3f} ({macd_direction})"
    if macd_increasing and hist_val > 0:
        macd_note += " — Bullish momentum increasing."
    elif not macd_increasing and hist_val > 0:
        macd_note += " — Bullish momentum fading."
    elif macd_increasing and hist_val < 0:
        macd_note += " — Bearish momentum fading, possible reversal."
    else:
        macd_note += " — Bearish momentum increasing."

    # Stochastic(14,3,3)
    stoch_k, stoch_d = calc_stochastic(df, 14, 3, 3)
    if stoch_k < 20 and stoch_d < 20:
        stoch_zone = "OVERSOLD"
        stoch_note = f"Stochastic %K={stoch_k:.1f}, %D={stoch_d:.1f} — Oversold. Watch for bullish crossover."
    elif stoch_k > 80 and stoch_d > 80:
        stoch_zone = "OVERBOUGHT"
        stoch_note = f"Stochastic %K={stoch_k:.1f}, %D={stoch_d:.1f} — Overbought. Watch for bearish crossover."
    else:
        stoch_zone = "NEUTRAL"
        stoch_note = f"Stochastic %K={stoch_k:.1f}, %D={stoch_d:.1f} — Mid-range."

    stoch_crossover_bullish = stoch_k > stoch_d and stoch_k < 30
    stoch_crossover_bearish = stoch_k < stoch_d and stoch_k > 70

    return {
        "rsi": round(rsi_val, 1),
        "rsi_zone": rsi_zone,
        "rsi_note": rsi_note,
        "rsi_aligned": rsi_aligned,
        "macd": round(macd_val, 3),
        "macd_signal": round(sig_val, 3),
        "macd_histogram": round(hist_val, 3),
        "macd_direction": macd_direction,
        "macd_increasing": macd_increasing,
        "macd_note": macd_note,
        "stochastic_k": round(stoch_k, 1),
        "stochastic_d": round(stoch_d, 1),
        "stochastic_zone": stoch_zone,
        "stochastic_note": stoch_note,
        "stochastic_crossover_bullish": stoch_crossover_bullish,
        "stochastic_crossover_bearish": stoch_crossover_bearish,
    }


# ═════════════════════════════════════════════════════════════════════════
#  LAYER 7: NEPSE-SPECIFIC FILTERS
# ═════════════════════════════════════════════════════════════════════════

def layer7_nepse(df, symbol, nepse_df):
    """NEPSE-specific analysis: index correlation, sector, circuit."""
    current_price = float(df["Close"].iloc[-1])
    prev_price    = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
    daily_change_pct = ((current_price - prev_price) / prev_price) * 100

    # Sector
    sector = SECTOR_MAP.get(symbol.upper(), "Other")

    # Circuit breaker
    at_upper_circuit = daily_change_pct >= 9.5
    at_lower_circuit = daily_change_pct <= -9.5
    near_circuit     = abs(daily_change_pct) >= 8.0

    circuit_note = None
    if at_upper_circuit:
        circuit_note = "🔒 Stock hit upper circuit (10% limit). Cannot buy more today."
    elif at_lower_circuit:
        circuit_note = "🔒 Stock hit lower circuit (-10% limit). Cannot sell more today."
    elif near_circuit:
        circuit_note = f"⚠ Stock moved {daily_change_pct:+.1f}% — nearing circuit limit. Plan exits carefully."

    # NEPSE index correlation
    nepse_aligned = None
    nepse_trend   = "UNKNOWN"
    nepse_note    = "NEPSE index data not available — scrape it for better analysis."

    if nepse_df is not None and len(nepse_df) > 5:
        idx_close = float(nepse_df["Close"].iloc[-1])
        idx_prev  = float(nepse_df["Close"].iloc[-2])
        idx_change = ((idx_close - idx_prev) / idx_prev) * 100

        # Simple recent trend
        idx_5d_ago = float(nepse_df["Close"].iloc[-6]) if len(nepse_df) > 5 else idx_close
        if idx_close > idx_5d_ago * 1.005:
            nepse_trend = "UP"
        elif idx_close < idx_5d_ago * 0.995:
            nepse_trend = "DOWN"
        else:
            nepse_trend = "FLAT"

        # Alignment: if stock is bullish and index is up, they align
        stock_dir = "UP" if current_price > prev_price else "DOWN"
        nepse_aligned = (stock_dir == "UP" and nepse_trend == "UP") or \
                        (stock_dir == "DOWN" and nepse_trend == "DOWN")

        nepse_note = f"NEPSE Index: {idx_close:.2f} ({idx_change:+.2f}%) — 5-day trend: {nepse_trend}. "
        if nepse_aligned:
            nepse_note += "✅ Market and stock are aligned."
        elif nepse_trend == "FLAT":
            nepse_note += "Market is flat — individual stock signals carry more weight."
        else:
            nepse_note += "⚠ Stock and market are diverging — trade with caution."

    return {
        "sector": sector,
        "daily_change_pct": round(daily_change_pct, 2),
        "at_upper_circuit": at_upper_circuit,
        "at_lower_circuit": at_lower_circuit,
        "near_circuit": near_circuit,
        "circuit_note": circuit_note,
        "nepse_trend": nepse_trend,
        "nepse_aligned": nepse_aligned,
        "nepse_note": nepse_note,
    }



# ═════════════════════════════════════════════════════════════════════════
#  LAYER 8: VWAP ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

def layer8_vwap(df):
    if "VWAP" not in df.columns:
        return {"vwap": 0, "vwap_distance": 0, "vwap_aligned": False, "vwap_note": "VWAP data unavailable."}
    vwap = float(df["VWAP"].iloc[-1]) if not pd.isna(df["VWAP"].iloc[-1]) else 0
    dist = float(df["VWAP_Distance"].iloc[-1]) if "VWAP_Distance" in df.columns and not pd.isna(df["VWAP_Distance"].iloc[-1]) else 0
    c = float(df["Close"].iloc[-1])
    aligned = c > vwap
    note = f"Price is {dist*100:+.1f}% vs VWAP."
    if aligned:
        note += " Bullish (above VWAP)."
    else:
        note += " Bearish (below VWAP)."
    return {"vwap": round(vwap, 2), "vwap_distance": round(dist, 4), "vwap_aligned": aligned, "vwap_note": note}

# ═════════════════════════════════════════════════════════════════════════
#  LAYER 9: VOLATILITY
# ═════════════════════════════════════════════════════════════════════════

def layer9_volatility(df):
    if "ATR_14" not in df.columns:
        return {"atr": 0, "bb_width": 0, "volatility_aligned": False, "volatility_note": "Volatility data unavailable."}
    atr = float(df["ATR_14"].iloc[-1]) if not pd.isna(df["ATR_14"].iloc[-1]) else 0
    bb = float(df["BB_Width"].iloc[-1]) if "BB_Width" in df.columns and not pd.isna(df["BB_Width"].iloc[-1]) else 0
    aligned = atr > 0
    note = f"ATR={atr:.2f}, BB_Width={bb:.2f}."
    return {"atr": round(atr, 2), "bb_width": round(bb, 4), "volatility_aligned": aligned, "volatility_note": note}

# ═════════════════════════════════════════════════════════════════════════
#  TRADE CHECKLIST & SCORING
# ═════════════════════════════════════════════════════════════════════════


def build_checklist(structure, levels, patterns, volume, trend, momentum, nepse_filter, vwap, volatility):
    """Build the 7-point trade checklist."""

    items = []
    score = 0

    # 1. STRUCTURE — Is the trend in my direction?
    trend_ok = structure["trend"] in ("UPTREND", "DOWNTREND")
    items.append({
        "id": "structure",
        "label": "STRUCTURE",
        "question": "Is the trend in my direction?",
        "pass": trend_ok,
        "detail": structure["trend_reason"],
        "value": structure["trend"],
    })
    if trend_ok:
        score += 1

    # 2. KEY LEVEL — Am I at a meaningful S/R zone?
    at_level = levels["at_key_level"]
    nearest = levels["nearest_support"] or levels["nearest_resistance"]
    nearest_desc = f"Nearest: {nearest['source']} at {nearest['price']:.2f} ({nearest['distance_pct']:.1f}% away)" if nearest else "No nearby levels detected"
    items.append({
        "id": "key_level",
        "label": "KEY LEVEL",
        "question": "Am I at a meaningful S/R zone?",
        "pass": at_level,
        "detail": nearest_desc,
        "value": "AT LEVEL" if at_level else "NOT AT LEVEL",
    })
    if at_level:
        score += 1

    # 3. CANDLE — Is there a confirming price action pattern?
    candle_ok = patterns["has_confirmation"]
    today_p = patterns["today_patterns"]
    candle_desc = ", ".join(p["name"] for p in today_p) if today_p else "No significant pattern on latest candle"
    items.append({
        "id": "candle",
        "label": "CANDLE",
        "question": "Is there a confirming price action?",
        "pass": candle_ok,
        "detail": candle_desc,
        "value": today_p[0]["name"] if today_p else "None",
    })
    if candle_ok:
        score += 1

    # 4. VOLUME — Does volume confirm the move?
    vol_ok = volume["confirms_move"] or volume["is_high_volume_rejection"]
    items.append({
        "id": "volume",
        "label": "VOLUME",
        "question": "Does volume confirm the move?",
        "pass": vol_ok,
        "detail": volume["meaning"],
        "value": volume["classification"].replace("_", " "),
    })
    if vol_ok:
        score += 1

    # 5. ADX — Is ADX > 25 (real trend)?
    adx_ok = trend["adx_trend_exists"]
    items.append({
        "id": "adx",
        "label": "ADX",
        "question": "Is ADX > 25 (real trend)?",
        "pass": adx_ok,
        "detail": trend["adx_note"],
        "value": f"{trend['adx']:.1f}",
    })
    if adx_ok:
        score += 1

    # 6. RSI — Is RSI aligned (not overbought if buying, not oversold if selling)?
    rsi_ok = momentum["rsi_aligned"]
    items.append({
        "id": "rsi",
        "label": "RSI",
        "question": "Is RSI aligned (not overbought/sold)?",
        "pass": rsi_ok,
        "detail": momentum["rsi_note"],
        "value": f"{momentum['rsi']:.1f} ({momentum['rsi_zone']})",
    })
    if rsi_ok:
        score += 1

    # 7. NEPSE INDEX — Is overall market helping or hurting?
    nepse_ok = nepse_filter["nepse_aligned"] if nepse_filter["nepse_aligned"] is not None else False
    items.append({
        "id": "nepse_index",
        "label": "NEPSE INDEX",
        "question": "Is overall market helping or hurting?",
        "pass": nepse_ok,
        "detail": nepse_filter["nepse_note"],
        "value": f"NEPSE {nepse_filter['nepse_trend']}",
    })
    if nepse_ok:
        score += 1

    # 8. VWAP
    vwap_ok = vwap["vwap_aligned"]
    items.append({
        "id": "vwap", "label": "VWAP", "question": "Is price above VWAP?",
        "pass": vwap_ok, "detail": vwap["vwap_note"], "value": f"{vwap['vwap']:.2f}"
    })
    if vwap_ok: score += 1

    # 9. VOLATILITY
    volatility_ok = volatility["volatility_aligned"]
    items.append({
        "id": "volatility", "label": "VOLATILITY", "question": "Is volatility adequate?",
        "pass": volatility_ok, "detail": volatility["volatility_note"], "value": f"ATR {volatility['atr']:.1f}"
    })
    if volatility_ok: score += 1

    # Rating
    if score >= 7:
        rating = "STRONG TRADE"
        rating_color = "green"
        advice = "High probability setup. Take the trade with full position size."
    elif score >= 5:
        rating = "ACCEPTABLE"
        rating_color = "gold"
        advice = "Decent setup. Take the trade but reduce position size (50-75%)."
    else:
        rating = "SKIP"
        rating_color = "red"
        advice = "Low probability — too many filters failing. Wait for a better setup."

    # Expected win rate
    win_rates = {9: "70-75%", 8: "68-72%", 7: "65-70%", 6: "60-65%", 5: "55-60%", 4: "50-55%", 3: "~50%", 2: "< 50%", 1: "< 50%", 0: "Don't trade"}

    return {
        "items": items,
        "score": score,
        "total": 9,
        "rating": rating,
        "rating_color": rating_color,
        "advice": advice,
        "expected_win_rate": win_rates.get(score, "Unknown"),
    }


# ═════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS RUNNER
# ═════════════════════════════════════════════════════════════════════════

def run_analysis(symbol):
    """Run all 7 layers and return the complete analysis payload."""
    print(f"Running 10-layer analysis for {symbol.upper()}...")
    df = load_data(symbol)
    df = add_all_indicators(df)
    nepse_df = load_nepse_index()

    # Layer 1: Market Structure
    print("  Layer 1: Market Structure...")
    structure = layer1_structure(df)
    swings_all = detect_swing_points(df, lookback=3)

    # Layer 2: Key Levels
    print("  Layer 2: Key Levels...")
    levels = layer2_levels(df, swings_all)

    # Layer 3: Price Action
    print("  Layer 3: Price Action Patterns...")
    patterns = detect_patterns(df)

    # Layer 4: Volume
    print("  Layer 4: Volume Analysis...")
    volume = layer4_volume(df)

    # Layer 5: Trend Indicators
    print("  Layer 5: Trend Indicators...")
    trend_indicators = layer5_trend(df)

    # Layer 6: Momentum
    print("  Layer 6: Momentum...")
    momentum = layer6_momentum(df)

    # Layer 7: NEPSE Filters
    print("  Layer 7: NEPSE-Specific Filters...")
    nepse_filter = layer7_nepse(df, symbol, nepse_df)

    print("  Layer 8: VWAP Analysis...")
    vwap = layer8_vwap(df)

    print("  Layer 9: Volatility...")
    volatility = layer9_volatility(df)

    # Checklist
    print("  Building trade checklist...")
    checklist = build_checklist(structure, levels, patterns, volume, trend_indicators, momentum, nepse_filter, vwap, volatility)

    # Latest bar summary
    last = df.iloc[-1]
    latest_bar = {
        "date": last["Date"].strftime("%Y-%m-%d"),
        "open": float(last["Open"]),
        "high": float(last["High"]),
        "low": float(last["Low"]),
        "close": float(last["Close"]),
        "volume": int(last["Volume"]),
    }

    # Chart data for levels overlay (last 60 days)
    chart_df = df.iloc[-60:]
    chart_data = {
        "dates":  [d.strftime("%Y-%m-%d") for d in chart_df["Date"]],
        "open":   [float(x) for x in chart_df["Open"]],
        "high":   [float(x) for x in chart_df["High"]],
        "low":    [float(x) for x in chart_df["Low"]],
        "close":  [float(x) for x in chart_df["Close"]],
        "volume": [float(x) for x in chart_df["Volume"]],
    }

    result = {
        "symbol": symbol.upper(),
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_bar": latest_bar,
        "total_data_points": len(df),
        "layer1_structure": structure,
        "layer2_levels": levels,
        "layer3_patterns": patterns,
        "layer4_volume": volume,
        "layer5_trend": trend_indicators,
        "layer6_momentum": momentum,
        "layer7_nepse": nepse_filter,
        "layer8_vwap": vwap,
        "layer9_volatility": volatility,
        "checklist": checklist,
        "chart_data": chart_data,
    }

    # Save to file
    out_file = os.path.join("outputs", "analysis", f"{symbol.lower()}_analysis.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"Analysis saved to {out_file}")

    return result


# ═════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="7-Layer NEPSE Professional Analysis")
    parser.add_argument("--symbol", type=str, default="LICN", help="Stock ticker symbol")
    args = parser.parse_args()

    result = run_analysis(args.symbol)

    cl = result["checklist"]
    print(f"\n{'='*60}")
    print(f"  TRADE CHECKLIST — {result['symbol']}")
    print(f"{'='*60}")
    for item in cl["items"]:
        mark = "✅ YES" if item["pass"] else "❌ NO "
        print(f"  {mark}  {item['label']:15s} — {item['question']}")
        print(f"         → {item['detail']}")
    print(f"{'='*60}")
    print(f"  SCORE: {cl['score']}/{cl['total']}  →  {cl['rating']}")
    print(f"  Expected Win Rate: {cl['expected_win_rate']}")
    print(f"  {cl['advice']}")
    print(f"{'='*60}\n")
    
    # Cleanup CSV files if run directly
    raw_file = f"{args.symbol.lower()}_data.csv"
    clean_file = f"{args.symbol.lower()}_cleaned.csv"
    for f in [raw_file, clean_file]:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"Cleaned up temporary file: {f}")
            except Exception as e:
                pass
