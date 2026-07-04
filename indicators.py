"""
indicators.py — Shared Technical Indicator Library
All 20 professional-grade indicators for NEPSE AI Predictor.

Used by both train_model.py and predict_nepse.py as a single source of truth.
Each function appends new columns to the DataFrame and returns it.
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════
#  TREND INDICATORS
# ══════════════════════════════════════════════════════════

def add_ema(df: pd.DataFrame) -> pd.DataFrame:
    """EMA 20, 50, 200 + alignment signal."""
    close = df["Close"]
    df["EMA_20"]  = close.ewm(span=20,  adjust=False).mean()
    df["EMA_50"]  = close.ewm(span=50,  adjust=False).mean()
    df["EMA_200"] = close.ewm(span=200, adjust=False).mean()
    # 1 = bullish stack (20>50>200), -1 = bearish stack, 0 = mixed
    df["EMA_Alignment"] = 0
    bullish = (df["EMA_20"] > df["EMA_50"]) & (df["EMA_50"] > df["EMA_200"])
    bearish = (df["EMA_20"] < df["EMA_50"]) & (df["EMA_50"] < df["EMA_200"])
    df.loc[bullish, "EMA_Alignment"] = 1
    df.loc[bearish, "EMA_Alignment"] = -1
    return df


def add_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend indicator. Direction: 1 = bullish, -1 = bearish."""
    atr = _true_range(df).rolling(period).mean()
    hl2 = (df["High"] + df["Low"]) / 2

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend   = pd.Series(np.nan, index=df.index)
    direction    = pd.Series(0,      index=df.index)
    final_upper  = upper_band.copy()
    final_lower  = lower_band.copy()

    for i in range(1, len(df)):
        # Upper band
        if upper_band.iloc[i] < final_upper.iloc[i - 1] or df["Close"].iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        # Lower band
        if lower_band.iloc[i] > final_lower.iloc[i - 1] or df["Close"].iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        # Direction
        if np.isnan(supertrend.iloc[i - 1]):
            direction.iloc[i] = -1
        elif supertrend.iloc[i - 1] == final_upper.iloc[i - 1]:
            direction.iloc[i] = -1 if df["Close"].iloc[i] <= final_upper.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if df["Close"].iloc[i] >= final_lower.iloc[i] else -1

        supertrend.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    df["Supertrend"]           = supertrend
    df["Supertrend_Direction"] = direction
    df["Supertrend_Distance"]  = (df["Close"] - supertrend) / df["Close"].replace(0, np.nan)
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX with +DI and -DI."""
    high, low, close = df["High"], df["Low"], df["Close"]

    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = _true_range(df)

    atr_s     = _wilder_smooth(tr,       period)
    plus_di   = 100 * _wilder_smooth(plus_dm,  period) / atr_s.replace(0, np.nan)
    minus_di  = 100 * _wilder_smooth(minus_dm, period) / atr_s.replace(0, np.nan)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = _wilder_smooth(dx, period)

    df["ADX"]      = adx
    df["Plus_DI"]  = plus_di
    df["Minus_DI"] = minus_di
    df["ADX_Signal"] = 0  # 1=trending up, -1=trending down, 0=weak
    df.loc[(adx > 25) & (plus_di > minus_di),  "ADX_Signal"] = 1
    df.loc[(adx > 25) & (minus_di > plus_di),  "ADX_Signal"] = -1
    return df


def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Ichimoku Cloud (Tenkan, Kijun, Senkou A/B, Chikou)."""
    high, low, close = df["High"], df["Low"], df["Close"]

    tenkan  = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    kijun   = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a  = ((tenkan + kijun) / 2).shift(26)
    span_b  = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou  = close.shift(-26)

    df["Ichi_Tenkan"]  = tenkan
    df["Ichi_Kijun"]   = kijun
    df["Ichi_SpanA"]   = span_a
    df["Ichi_SpanB"]   = span_b
    df["Ichi_Chikou"]  = chikou

    # Cloud signal: 1=above cloud (bullish), -1=below cloud (bearish), 0=inside
    cloud_top    = span_a.combine(span_b, max)
    cloud_bottom = span_a.combine(span_b, min)
    df["Ichi_Cloud_Signal"] = 0
    df.loc[close > cloud_top,    "Ichi_Cloud_Signal"] = 1
    df.loc[close < cloud_bottom, "Ichi_Cloud_Signal"] = -1

    # TK Cross: 1=bullish cross, -1=bearish cross
    tk_diff      = tenkan - kijun
    tk_diff_prev = tk_diff.shift(1)
    df["Ichi_TK_Cross"] = 0
    df.loc[(tk_diff > 0) & (tk_diff_prev <= 0), "Ichi_TK_Cross"] = 1
    df.loc[(tk_diff < 0) & (tk_diff_prev >= 0), "Ichi_TK_Cross"] = -1
    return df


def add_donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian Channel: upper, lower, mid, width."""
    df["Donchian_Upper"] = df["High"].rolling(period).max()
    df["Donchian_Lower"] = df["Low"].rolling(period).min()
    df["Donchian_Mid"]   = (df["Donchian_Upper"] + df["Donchian_Lower"]) / 2
    width                = df["Donchian_Upper"] - df["Donchian_Lower"]
    df["Donchian_Width"] = width / df["Close"].replace(0, np.nan)
    # Position within channel: 0=at lower, 1=at upper
    df["Donchian_Pos"]   = (df["Close"] - df["Donchian_Lower"]) / width.replace(0, np.nan)
    return df


# ══════════════════════════════════════════════════════════
#  VOLUME INDICATORS
# ══════════════════════════════════════════════════════════

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume + 20-period SMA + divergence signal."""
    if "Volume" not in df.columns or df["Volume"].isna().all():
        df["OBV"] = np.nan
        df["OBV_SMA_20"] = np.nan
        df["OBV_Divergence"] = 0
        return df

    direction = np.sign(df["Close"].diff()).fillna(0)
    obv = (direction * df["Volume"]).cumsum()
    df["OBV"]        = obv
    df["OBV_SMA_20"] = obv.rolling(20).mean()
    # Divergence: price rising but OBV falling = bearish (-1), price falling but OBV rising = bullish (1)
    price_dir = np.sign(df["Close"].diff(5))
    obv_dir   = np.sign(obv.diff(5))
    df["OBV_Divergence"] = 0
    df.loc[(price_dir > 0) & (obv_dir < 0), "OBV_Divergence"] = -1
    df.loc[(price_dir < 0) & (obv_dir > 0), "OBV_Divergence"] = 1
    return df


def add_cmf(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Chaikin Money Flow."""
    if "Volume" not in df.columns or df["Volume"].isna().all():
        df["CMF"] = np.nan
        return df

    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range
    mfv = mfm * df["Volume"]
    df["CMF"] = mfv.rolling(period).sum() / df["Volume"].rolling(period).sum().replace(0, np.nan)
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Daily VWAP proxy using daily OHLCV. Also Anchored VWAP from swing low."""
    if "Volume" not in df.columns or df["Volume"].isna().all():
        df["VWAP"] = np.nan
        df["VWAP_Distance"] = np.nan
        df["AVWAP"] = np.nan
        df["AVWAP_Distance"] = np.nan
        return df

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    tpv = typical_price * df["Volume"]
    # Rolling 20-day VWAP (institutional proxy for daily data)
    df["VWAP"] = tpv.rolling(20).sum() / df["Volume"].rolling(20).sum().replace(0, np.nan)
    df["VWAP_Distance"] = (df["Close"] - df["VWAP"]) / df["VWAP"].replace(0, np.nan)

    # Anchored VWAP: anchor to most recent 52-week swing low (auto-detected)
    lookback = min(252, len(df))
    window   = df.tail(lookback).copy()
    anchor_idx = window["Low"].idxmin()
    anchor_pos = df.index.get_loc(anchor_idx)
    anchored_tpv = tpv.iloc[anchor_pos:]
    anchored_vol = df["Volume"].iloc[anchor_pos:]
    avwap_vals   = anchored_tpv.cumsum() / anchored_vol.cumsum().replace(0, np.nan)
    df["AVWAP"]          = np.nan
    df.loc[avwap_vals.index, "AVWAP"] = avwap_vals
    df["AVWAP_Distance"] = (df["Close"] - df["AVWAP"]) / df["AVWAP"].replace(0, np.nan)
    return df


def add_volume_profile(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """Volume Profile: Point of Control, Value Area High/Low (last 60 bars)."""
    if "Volume" not in df.columns or df["Volume"].isna().all():
        df["VP_POC"]     = np.nan
        df["VP_VAH"]     = np.nan
        df["VP_VAL"]     = np.nan
        df["VP_POC_Dist"] = np.nan
        return df

    window = min(60, len(df))

    poc_vals = []
    vah_vals = []
    val_vals = []

    for i in range(len(df)):
        if i < window - 1:
            poc_vals.append(np.nan)
            vah_vals.append(np.nan)
            val_vals.append(np.nan)
            continue
        sub    = df.iloc[i - window + 1: i + 1]
        lo, hi = sub["Low"].min(), sub["High"].max()
        if hi == lo:
            poc_vals.append(sub["Close"].iloc[-1])
            vah_vals.append(hi)
            val_vals.append(lo)
            continue
        edges   = np.linspace(lo, hi, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        vol_by_bin = np.zeros(bins)
        for _, row in sub.iterrows():
            for j, (e_lo, e_hi) in enumerate(zip(edges[:-1], edges[1:])):
                if row["Low"] <= e_hi and row["High"] >= e_lo:
                    vol_by_bin[j] += row["Volume"]

        poc_idx  = np.argmax(vol_by_bin)
        poc_vals.append(centers[poc_idx])

        total_vol = vol_by_bin.sum()
        target    = total_vol * 0.70
        # Expand from POC outward to find 70% value area
        included  = {poc_idx}
        left, right = poc_idx - 1, poc_idx + 1
        running   = vol_by_bin[poc_idx]
        while running < target and (left >= 0 or right < bins):
            l_vol = vol_by_bin[left]  if left >= 0    else -1
            r_vol = vol_by_bin[right] if right < bins  else -1
            if l_vol >= r_vol and left >= 0:
                included.add(left);  running += l_vol; left  -= 1
            elif right < bins:
                included.add(right); running += r_vol; right += 1
            else:
                break
        vah_vals.append(centers[max(included)])
        val_vals.append(centers[min(included)])

    df["VP_POC"]      = poc_vals
    df["VP_VAH"]      = vah_vals
    df["VP_VAL"]      = val_vals
    df["VP_POC_Dist"] = (df["Close"] - df["VP_POC"]) / df["VP_POC"].replace(0, np.nan)
    return df


# ══════════════════════════════════════════════════════════
#  VOLATILITY
# ══════════════════════════════════════════════════════════

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range."""
    df["ATR_14"] = _true_range(df).rolling(period).mean()
    df["ATR_Pct"] = df["ATR_14"] / df["Close"].replace(0, np.nan)
    return df


def add_bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands with position and squeeze."""
    sma    = df["Close"].rolling(period).mean()
    sd     = df["Close"].rolling(period).std()
    upper  = sma + std * sd
    lower  = sma - std * sd
    width  = upper - lower
    df["BB_Middle"]   = sma
    df["BB_Upper"]    = upper
    df["BB_Lower"]    = lower
    df["BB_Width"]    = width / sma.replace(0, np.nan)
    df["BB_Position"] = (df["Close"] - lower) / width.replace(0, np.nan)
    # Squeeze: current width < 80% of 20-day avg width
    avg_width = df["BB_Width"].rolling(20).mean()
    df["BB_Squeeze"] = (df["BB_Width"] < avg_width * 0.8).astype(int)
    return df


# ══════════════════════════════════════════════════════════
#  PRICE ACTION — MARKET STRUCTURE
# ══════════════════════════════════════════════════════════

def add_market_structure(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """HH/HL/LH/LL classification + Break of Structure."""
    n     = len(df)
    highs = df["High"].values
    lows  = df["Low"].values

    struct_trend = np.zeros(n)
    bos_signal   = np.zeros(n)

    swing_highs = []
    swing_lows  = []

    for i in range(lookback, n):
        # Detect swing high
        if all(highs[i] > highs[i - j] for j in range(1, lookback + 1)):
            swing_highs.append((i, highs[i]))
        # Detect swing low
        if all(lows[i] < lows[i - j] for j in range(1, lookback + 1)):
            swing_lows.append((i, lows[i]))

        # Classify with last 2 swings
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            prev_sh, last_sh = swing_highs[-2][1], swing_highs[-1][1]
            prev_sl, last_sl = swing_lows[-2][1],  swing_lows[-1][1]
            hh = last_sh > prev_sh
            hl = last_sl > prev_sl
            lh = last_sh < prev_sh
            ll = last_sl < prev_sl
            if hh and hl:
                struct_trend[i] = 1
            elif lh and ll:
                struct_trend[i] = -1
            else:
                struct_trend[i] = 0

        # BOS: price closes beyond last swing high/low
        if swing_highs and df["Close"].iloc[i] > swing_highs[-1][1]:
            bos_signal[i] = 1   # bullish BOS
        elif swing_lows and df["Close"].iloc[i] < swing_lows[-1][1]:
            bos_signal[i] = -1  # bearish BOS

    df["Struct_Trend"] = struct_trend
    df["Struct_BOS"]   = bos_signal
    # Fill forward so every row has a trend value
    df["Struct_Trend"] = df["Struct_Trend"].replace(0, np.nan).ffill().fillna(0)
    return df


def add_order_blocks(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Order blocks: the last bearish candle before a strong bullish move (bullish OB)
    and the last bullish candle before a strong bearish move (bearish OB).
    Returns distance to nearest OB as % of price.
    """
    n = len(df)
    ob_bull_dist = np.full(n, np.nan)
    ob_bear_dist = np.full(n, np.nan)

    close = df["Close"].values
    open_ = df["Open"].values
    high  = df["High"].values
    low   = df["Low"].values

    for i in range(5, n):
        # Look back for order blocks in last 'lookback' bars
        start = max(0, i - lookback)
        bull_obs = []
        bear_obs = []
        for j in range(start, i - 2):
            # Bullish OB: bearish candle followed by 2 bullish candles with strong move
            if close[j] < open_[j]:  # bearish candle
                if close[j + 1] > open_[j + 1] and close[j + 2] > open_[j + 2]:
                    move = (close[j + 2] - close[j]) / max(close[j], 0.0001)
                    if move > 0.01:  # 1% move threshold
                        bull_obs.append((high[j] + low[j]) / 2)

            # Bearish OB: bullish candle followed by 2 bearish candles with strong move
            if close[j] > open_[j]:  # bullish candle
                if close[j + 1] < open_[j + 1] and close[j + 2] < open_[j + 2]:
                    move = (close[j] - close[j + 2]) / max(close[j], 0.0001)
                    if move > 0.01:
                        bear_obs.append((high[j] + low[j]) / 2)

        c = close[i]
        if bull_obs:
            nearest = min(bull_obs, key=lambda x: abs(c - x))
            ob_bull_dist[i] = (c - nearest) / max(c, 0.0001)
        if bear_obs:
            nearest = min(bear_obs, key=lambda x: abs(c - x))
            ob_bear_dist[i] = (c - nearest) / max(c, 0.0001)

    df["OB_Bull_Dist"] = ob_bull_dist
    df["OB_Bear_Dist"] = ob_bear_dist
    # Encode proximity as signal: near bullish OB = 1, near bearish OB = -1
    df["OB_Signal"] = 0
    df.loc[df["OB_Bull_Dist"].abs() < 0.02, "OB_Signal"] = 1
    df.loc[df["OB_Bear_Dist"].abs() < 0.02, "OB_Signal"] = -1
    return df


def add_fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    True Fair Value Gaps (3-candle imbalance).
    Bullish FVG: candle[i-2].High < candle[i].Low (gap up)
    Bearish FVG: candle[i-2].Low  > candle[i].High (gap down)
    Returns distance to nearest unfilled FVG.
    """
    n = len(df)
    fvg_bull_dist = np.full(n, np.nan)
    fvg_bear_dist = np.full(n, np.nan)

    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values

    bull_fvgs = []  # (gap_low, gap_high, bar_index)
    bear_fvgs = []

    for i in range(2, n):
        # Bullish FVG
        if high[i - 2] < low[i]:
            gap_lo = high[i - 2]
            gap_hi = low[i]
            bull_fvgs.append((gap_lo, gap_hi, i))

        # Bearish FVG
        if low[i - 2] > high[i]:
            gap_lo = high[i]
            gap_hi = low[i - 2]
            bear_fvgs.append((gap_lo, gap_hi, i))

        # Remove filled FVGs
        bull_fvgs = [(lo, hi, idx) for lo, hi, idx in bull_fvgs if close[i] > lo]
        bear_fvgs = [(lo, hi, idx) for lo, hi, idx in bear_fvgs if close[i] < hi]

        c = close[i]
        if bull_fvgs:
            nearest = min(bull_fvgs, key=lambda x: abs(c - (x[0] + x[1]) / 2))
            mid = (nearest[0] + nearest[1]) / 2
            fvg_bull_dist[i] = (c - mid) / max(c, 0.0001)
        if bear_fvgs:
            nearest = min(bear_fvgs, key=lambda x: abs(c - (x[0] + x[1]) / 2))
            mid = (nearest[0] + nearest[1]) / 2
            fvg_bear_dist[i] = (c - mid) / max(c, 0.0001)

    df["FVG_Bull_Dist"] = fvg_bull_dist
    df["FVG_Bear_Dist"] = fvg_bear_dist
    df["FVG_Signal"] = 0
    df.loc[df["FVG_Bull_Dist"].between(-0.03, 0.0), "FVG_Signal"] = 1
    df.loc[df["FVG_Bear_Dist"].between(0.0, 0.03),  "FVG_Signal"] = -1
    return df


def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite candlestick signal: checks last candle for major patterns.
    Returns Candle_Signal: +2 strong bull, +1 bull, 0 neutral, -1 bear, -2 strong bear.
    """
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    n = len(df)

    signals = np.zeros(n)

    for i in range(2, n):
        body      = abs(c[i] - o[i])
        rng       = h[i] - l[i]
        upper_wick = h[i] - max(o[i], c[i])
        lower_wick = min(o[i], c[i]) - l[i]
        body_pct   = body / rng if rng > 0 else 0
        bull       = c[i] > o[i]

        score = 0

        # Doji
        if body_pct < 0.1:
            score = 0

        # Pin Bar (strong reversal)
        elif lower_wick > 2 * body and upper_wick < body:
            score = 2  # bullish pin bar
        elif upper_wick > 2 * body and lower_wick < body:
            score = -2  # bearish pin bar

        # Marubozu
        elif body_pct > 0.85:
            score = 2 if bull else -2

        # Engulfing
        elif i > 0:
            prev_bull = c[i - 1] > o[i - 1]
            if bull and not prev_bull and c[i] > o[i - 1] and o[i] < c[i - 1]:
                score = 2
            elif not bull and prev_bull and c[i] < o[i - 1] and o[i] > c[i - 1]:
                score = -2

        # Morning / Evening Star
        elif i > 1:
            prev2_bull = c[i - 2] > o[i - 2]
            prev1_body = abs(c[i - 1] - o[i - 1]) / (h[i - 1] - l[i - 1] + 1e-8)
            if not prev2_bull and prev1_body < 0.3 and bull:
                score = 2
            elif prev2_bull and prev1_body < 0.3 and not bull:
                score = -2

        # Regular candle
        else:
            score = 1 if bull else -1

        signals[i] = np.clip(score, -2, 2)

    df["Candle_Signal"] = signals
    return df


def add_support_resistance(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Compute distance to nearest support and resistance from rolling window."""
    n = len(df)
    dist_sup = np.full(n, np.nan)
    dist_res = np.full(n, np.nan)

    for i in range(lookback, n):
        sub   = df.iloc[i - lookback: i]
        close = df["Close"].iloc[i]
        highs = sub["High"].values
        lows  = sub["Low"].values

        # Resistance = rolling max highs, support = rolling min lows
        resistance = highs.max()
        support    = lows.min()

        dist_res[i] = (resistance - close) / max(close, 0.0001)
        dist_sup[i] = (close - support)    / max(close, 0.0001)

    df["Dist_To_Resistance"] = dist_res
    df["Dist_To_Support"]    = dist_sup
    return df


def add_trendlines(df: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    """Linear regression trendline slope and distance."""
    n      = len(df)
    slopes = np.full(n, np.nan)
    dists  = np.full(n, np.nan)

    close_vals = df["Close"].values

    for i in range(lookback, n):
        y = close_vals[i - lookback: i]
        x = np.arange(lookback)
        # Linear regression
        x_mean, y_mean = x.mean(), y.mean()
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        intercept = y_mean - slope * x_mean
        trendline_val = slope * (lookback - 1) + intercept

        slopes[i] = slope / max(y_mean, 0.0001)  # normalized slope
        dists[i]  = (close_vals[i] - trendline_val) / max(close_vals[i], 0.0001)

    df["Trendline_Slope"] = slopes
    df["Trendline_Dist"]  = dists
    return df


# ══════════════════════════════════════════════════════════
#  MASTER FUNCTION
# ══════════════════════════════════════════════════════════

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 20 professional indicators and append to DataFrame.
    Input: DataFrame with columns [Date, Open, High, Low, Close, Volume (optional)]
    Output: Same DataFrame with ~50+ new indicator columns.
    """
    df = df.copy()

    # Ensure numeric types
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # 1. Trend
    df = add_ema(df)
    df = add_supertrend(df)
    df = add_adx(df)
    df = add_ichimoku(df)
    df = add_donchian(df)

    # 2. Momentum (keep existing + standardize)
    df = _add_rsi(df)
    df = _add_macd(df)
    df = add_bollinger(df)

    # 3. Volume
    df = add_obv(df)
    df = add_cmf(df)
    df = add_vwap(df)
    df = add_volume_profile(df)

    # 4. Volatility
    df = add_atr(df)

    # 5. Price Action
    df = add_market_structure(df)
    df = add_order_blocks(df)
    df = add_fair_value_gaps(df)
    df = add_candlestick_patterns(df)
    df = add_support_resistance(df)
    df = add_trendlines(df)

    return df


def get_ml_features(df: pd.DataFrame, include_volume: bool = True) -> list:
    """
    Return the list of feature column names available in df after add_all_indicators().
    Pass include_volume=False for NEPSE index where volume is unreliable.
    """
    base = [
        # Price
        "Open", "High", "Low", "Close",
        # Returns
        "Daily_Return", "Return_5", "Return_10",
        "Volatility_5", "Volatility_20",
        # EMA
        "EMA_20", "EMA_50", "EMA_200", "EMA_Alignment",
        # Supertrend
        "Supertrend_Direction", "Supertrend_Distance",
        # ADX
        "ADX", "Plus_DI", "Minus_DI", "ADX_Signal",
        # Ichimoku
        "Ichi_Cloud_Signal", "Ichi_TK_Cross",
        # Donchian
        "Donchian_Width", "Donchian_Pos",
        # RSI
        "RSI_14",
        # MACD
        "MACD", "MACD_Signal", "MACD_Hist",
        # Bollinger
        "BB_Width", "BB_Position", "BB_Squeeze",
        # ATR
        "ATR_14", "ATR_Pct",
        # Market Structure
        "Struct_Trend", "Struct_BOS",
        # Order Blocks
        "OB_Signal",
        # FVG
        "FVG_Signal",
        # Candlestick
        "Candle_Signal",
        # S/R
        "Dist_To_Resistance", "Dist_To_Support",
        # Trendlines
        "Trendline_Slope", "Trendline_Dist",
        # Lag features (added in train_model.py / predict_nepse.py)
    ]

    volume_features = [
        "OBV_Divergence",
        "CMF",
        "VWAP_Distance",
        "AVWAP_Distance",
        "VP_POC_Dist",
    ]

    if include_volume:
        base += volume_features

    # Only return features that actually exist in the DataFrame
    return [f for f in base if f in df.columns]


# ══════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ══════════════════════════════════════════════════════════

def _true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (equivalent to EMA with alpha=1/period)."""
    result = series.ewm(alpha=1 / period, adjust=False).mean()
    return result


def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    return df


def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD"]        = macd
    df["MACD_Signal"] = signal
    df["MACD_Hist"]   = macd - signal
    return df
