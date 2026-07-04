import os
import json
import argparse
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from analysis import run_analysis, NumpyEncoder
from indicators import add_all_indicators, get_ml_features

def parse_args():
    parser = argparse.ArgumentParser(description="Train stock prediction models and generate predictions.")
    parser.add_argument("--symbol", type=str, default="LICN", help="Stock ticker symbol (e.g. LICN, NABIL)")
    return parser.parse_args()

def clean_data(symbol):
    raw_file = os.path.join("data", f"{symbol.lower()}_data.csv")
    clean_file = os.path.join("data", f"{symbol.lower()}_cleaned.csv")
    
    if not os.path.exists(raw_file):
        print(f"[*] {raw_file} not found. Fetching live data for {symbol} from internet...")
        import subprocess, sys
        subprocess.run([sys.executable, "update_data.py", "--symbol", symbol, "--pages", "all"])
        if not os.path.exists(raw_file):
            raise FileNotFoundError(f"Raw data file {raw_file} not found. Scrape failed.")
        
    print(f"Loading raw data from {raw_file}...")
    df = pd.read_csv(raw_file)
    
    # Drop fully empty rows
    df = df.dropna(how="all")
    
    # Keep only valid date rows (filters garbage like headers and table pagination links)
    df = df[df["Date"].astype(str).str.contains(r"\d{4}[-/]\d{2}[-/]\d{2}", na=False)]
    
    # CRITICAL BUG FIX: Clean and strip commas from ALL numeric columns before converting
    # Only clean columns that actually exist — NepseAlpha and MeroLagani return
    # different schemas (e.g. MeroLagani includes "Change", NepseAlpha doesn't).
    cols = ["Close", "Change", "High", "Low", "Open", "Volume"]
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = df[c].astype(str).str.replace(",", "", regex=True)
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # If the source didn't provide a "Change" column (e.g. NepseAlpha), derive it
    # from Close so any downstream code expecting it still works.
    if "Change" not in df.columns:
        df["Change"] = pd.to_numeric(df["Close"], errors="coerce").diff()
        
    # Convert Date column safely
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # Remove rows where Date or essential prices failed conversion
    df = df.dropna(subset=["Date", "Close", "Open", "High", "Low"])
    
    # Sort chronologically (ascending, required for time-series feature engineering)
    df = df.sort_values("Date").reset_index(drop=True)
    
    print(f"Cleaned dataset shape: {df.shape} (Saved to {clean_file})")
    df.to_csv(clean_file, index=False)
    return df

def calculate_technical_indicators(df):
    """Compute all 20 professional indicators using the shared indicators library."""
    print("Calculating technical indicators (20-indicator suite)...")
    df = df.copy()

    # All 20 indicators via shared library
    df = add_all_indicators(df)

    # Keep classic SMAs for backward compatibility with existing saved predictions
    df["SMA_10"]  = df["Close"].rolling(window=10, min_periods=1).mean()
    df["SMA_50"]  = df["Close"].rolling(window=50, min_periods=1).mean()
    df["SMA_200"] = df["Close"].rolling(window=200, min_periods=1).mean()

    # Returns and momentum
    df["Daily_Return"]  = df["Close"].pct_change()
    df["Return_5"]      = df["Close"].pct_change(5)
    df["Return_10"]     = df["Close"].pct_change(10)
    df["Volatility_5"]  = df["Daily_Return"].rolling(5).std()
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std()

    if "Volume" in df.columns:
        df["Volume_Change"] = df["Volume"].pct_change()
        df["Volume_SMA_10"] = df["Volume"].rolling(window=10, min_periods=1).mean()

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f"Close_Lag_{lag}"]  = df["Close"].shift(lag)
        df[f"Return_Lag_{lag}"] = df["Daily_Return"].shift(lag)
        if "Volume" in df.columns:
            df[f"Volume_Lag_{lag}"] = df["Volume"].shift(lag)

    return df

def get_next_trading_day(last_date):
    # Closed Saturday (5) and Sunday (6)
    next_day = last_date + pd.Timedelta(days=1)
    while next_day.weekday() in [5, 6]:  # Saturday (5) or Sunday (6)
        next_day += pd.Timedelta(days=1)
    return next_day
def get_cross_validation_metrics(df_h, features, h_days, db):
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np

    df_clean = df_h.dropna(subset=features + ["Target_Class"])
    if len(df_clean) < 50:
        return 0.33, 0.33, 0.0

    X = df_clean[features]
    y = df_clean["Target_Class"].astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    acc_scores = []
    base_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if len(y_train) < 15 or len(y_test) < 1:
            continue

        rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1).fit(X_train, y_train)
        xgb = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1).fit(X_train, y_train)

        # Handle probability alignment across folds that might lack classes
        rf_classes = rf.classes_
        xgb_classes = xgb.classes_
        rf_raw = rf.predict_proba(X_test)
        xgb_raw = xgb.predict_proba(X_test)

        def pad_probs(probs, classes, target_len=3):
            if probs.shape[1] == target_len:
                return probs
            full_probs = np.zeros((probs.shape[0], target_len))
            for idx, c in enumerate(classes):
                if c < target_len:
                    full_probs[:, int(c)] = probs[:, idx]
            return full_probs

        rf_probs = pad_probs(rf_raw, rf_classes)
        xgb_probs = pad_probs(xgb_raw, xgb_classes)

        probs = (rf_probs + xgb_probs) / 2.0
        preds = np.argmax(probs, axis=1)

        acc = accuracy_score(y_test, preds)
        majority = int(y_train.mode()[0])
        base = (y_test == majority).mean()

        acc_scores.append(acc)
        base_scores.append(base)

    mean_acc = float(np.mean(acc_scores)) if acc_scores else 0.33
    mean_base = float(np.mean(base_scores)) if base_scores else 0.33
    skill = 1.0 - ((1.0 - mean_acc) / (1.0 - mean_base + 1e-8)) if mean_base < 1.0 else 0.0
    return mean_acc, mean_base, skill


def train_and_predict(symbol, df_features):
    """Train classification + regression models with stationary features and walk-forward cross-validation"""
    df = df_features.copy()
    has_volume = "Volume" in df.columns

    # ── STATIONARIZE / NORMALIZE FEATURES ─────────────────────────────────
    df["Close_to_SMA_20"] = (df["Close"] / df["BB_Middle"]) - 1.0
    df["Close_to_SMA_50"] = (df["Close"] / df["SMA_50"]) - 1.0 if "SMA_50" in df.columns else 0.0
    df["Close_to_SMA_200"] = (df["Close"] / df["SMA_200"]) - 1.0 if "SMA_200" in df.columns else 0.0
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Middle"] + 1e-8)
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-8)
    
    df["MACD_Norm"] = df["MACD"] / (df["Close"] + 1e-8)
    df["MACD_Signal_Norm"] = df["MACD_Signal"] / (df["Close"] + 1e-8)
    df["MACD_Hist_Norm"] = df["MACD_Hist"] / (df["Close"] + 1e-8)
    df["ATR_Norm"] = df["ATR_14"] / (df["Close"] + 1e-8)

    features = [
        "Close_to_SMA_20", "Close_to_SMA_50", "Close_to_SMA_200", 
        "BB_Width", "BB_Position", "RSI_14", 
        "MACD_Norm", "MACD_Signal_Norm", "MACD_Hist_Norm", 
        "ATR_Norm", "Volatility_20", "Volume_Change", "ADX", "Daily_Return"
    ]

    # Add lag returns only (avoid raw close prices)
    for lag in [1, 2, 3, 5, 10]:
        col = f"Return_Lag_{lag}"
        if col in df.columns and col not in features:
            features.append(col)
        if has_volume:
            col = f"Volume_Lag_{lag}"
            if col in df.columns and col not in features:
                features.append(col)

    features = [f for f in features if f in df.columns and df[f].notna().sum() > 50]
    print(f"  Using {len(features)} stationary ML features")

    horizons = {"1D": 1, "5D": 5, "20D": 20}
    predictions = {}
    metrics = {}
    
    # Track 1D models for chart predictions
    rf_reg_1d_high = None
    xgb_reg_1d_high = None
    rf_reg_1d_low = None
    xgb_reg_1d_low = None
    test_data_1d = None
    train_test_1d = None
    
    analysis_results = run_analysis(symbol)
    
    inference_row_original = df.iloc[[-1]].copy()
    today_close = float(inference_row_original["Close"].iloc[0])
    today_date = inference_row_original["Date"].iloc[0]
    
    checklist = analysis_results.get("checklist", {})
    layer_score = checklist.get("score", 0)
    
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from xgboost import XGBRegressor, XGBClassifier
    from sklearn.metrics import mean_absolute_error
    import numpy as np

    # Train for each horizon
    for h_name, h_days in horizons.items():
        print(f"\n--- Training Horizon {h_name} ({h_days} days) ---")
        
        df_h = df.copy()
        df_h["Target_Return"] = df_h["Close"].pct_change(periods=h_days).shift(-h_days)
        df_h["Target_High_Return"] = (df_h["High"].shift(-h_days) - df_h["Close"]) / df_h["Close"]
        df_h["Target_Low_Return"] = (df_h["Low"].shift(-h_days) - df_h["Close"]) / df_h["Close"]
        
        # Discretize targets: 0 = DOWN, 1 = FLAT, 2 = UP
        db = { "1D": 0.005, "5D": 0.015, "20D": 0.040 }[h_name]
        
        def assign_class(r):
            if pd.isna(r):
                return np.nan
            if r > db: return 2
            if r < -db: return 0
            return 1
            
        df_h["Target_Class"] = df_h["Target_Return"].apply(assign_class)
        df_h = df_h.reset_index(drop=True)
        
        inference_row = df_h.iloc[[-1]].copy()
        train_test_data = df_h.iloc[:-h_days].copy()
        train_test_data = train_test_data.dropna(subset=features + ["Target_Return", "Target_High_Return", "Target_Low_Return", "Target_Class"])
        
        if len(train_test_data) == 0:
            train_test_data = df_h.iloc[:-h_days].copy()
            train_test_data[features] = train_test_data[features].ffill().bfill()
            train_test_data = train_test_data.dropna(subset=["Target_Return", "Target_High_Return", "Target_Low_Return", "Target_Class"])
            
        # Get Cross Validation Skill Scores
        cv_acc, cv_base, cv_skill = get_cross_validation_metrics(df_h, features, h_days, db)
        print(f"  [{h_name}] Walk-Forward Skill Score: {cv_skill:.4f} (Accuracy: {cv_acc:.2%}, Baseline: {cv_base:.2%})")

        split_idx = int(len(train_test_data) * 0.8)
        train_data = train_test_data.iloc[:split_idx]
        test_data = train_test_data.iloc[split_idx:]
        
        if len(train_data) < 15 or len(test_data) < 1:
            print(f"Not enough data for {h_name}")
            continue
            
        X_train = train_data[features]
        y_train_return = train_data["Target_Return"]
        y_train_high = train_data["Target_High_Return"]
        y_train_low = train_data["Target_Low_Return"]
        y_train_class = train_data["Target_Class"].astype(int)
        
        X_test = test_data[features]
        y_test_return = test_data["Target_Return"]
        y_test_high = test_data["Target_High_Return"]
        y_test_low = test_data["Target_Low_Return"]
        y_test_class = test_data["Target_Class"].astype(int)
        
        # 1. Regressor (predicts High/Low return percentage without price bias)
        rf_high_reg = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1).fit(X_train, y_train_high)
        xgb_high_reg = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1).fit(X_train, y_train_high)
        
        rf_low_reg = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1).fit(X_train, y_train_low)
        xgb_low_reg = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1).fit(X_train, y_train_low)
        
        ensemble_high_pred = (rf_high_reg.predict(X_test) + xgb_high_reg.predict(X_test)) / 2.0
        ensemble_low_pred = (rf_low_reg.predict(X_test) + xgb_low_reg.predict(X_test)) / 2.0
        ensemble_reg_pred = (ensemble_high_pred + ensemble_low_pred) / 2.0
        
        mae = mean_absolute_error(y_test_return, ensemble_reg_pred)
        test_closes = test_data["Close"].values
        pred_closes = test_closes * (1.0 + ensemble_reg_pred)
        actual_closes = test_closes * (1.0 + y_test_return.values)
        mape = float(np.mean(np.abs((actual_closes - pred_closes) / (actual_closes + 1e-8))) * 100)
        
        if h_name == '1D':
            rf_reg_1d_high = rf_high_reg
            xgb_reg_1d_high = xgb_high_reg
            rf_reg_1d_low = rf_low_reg
            xgb_reg_1d_low = xgb_low_reg
            test_data_1d = test_data.copy()
            train_test_1d = train_test_data.copy()
            
        # 2. Classifier (predicts probabilities of DOWN, FLAT, UP)
        rf_clf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1).fit(X_train, y_train_class)
        xgb_clf = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1).fit(X_train, y_train_class)
        
        # 3. Inference
        X_inf = inference_row[features]
        pred_high_return = (rf_high_reg.predict(X_inf)[0] + xgb_high_reg.predict(X_inf)[0]) / 2.0
        pred_low_return = (rf_low_reg.predict(X_inf)[0] + xgb_low_reg.predict(X_inf)[0]) / 2.0
        
        pred_high = today_close * (1.0 + pred_high_return)
        pred_low = today_close * (1.0 + pred_low_return)
        pred_close = (pred_high + pred_low) / 2.0
        pred_return = (pred_close - today_close) / today_close
        
        # Calculate class probabilities
        rf_probs = rf_clf.predict_proba(X_inf)
        xgb_probs = xgb_clf.predict_proba(X_inf)
        
        # Safe class matching (make sure classes 0, 1, 2 are padded)
        def align_classes(probs, classes, target_len=3):
            p_aligned = np.zeros(target_len)
            for idx, c in enumerate(classes):
                if c < target_len:
                    p_aligned[int(c)] = probs[0, idx]
            return p_aligned
            
        probs_aligned = (align_classes(rf_probs, rf_clf.classes_) + align_classes(xgb_probs, xgb_clf.classes_)) / 2.0
        predicted_class = int(np.argmax(probs_aligned))
        direction_confidence = float(probs_aligned[predicted_class])
        
        # Map class to direction
        direction_labels = {0: "DOWN", 1: "FLAT", 2: "UP"}
        predicted_dir = direction_labels[predicted_class]
        
        # Signal based on prediction class and 7-layer score
        if predicted_class == 2 and layer_score >= 4:
            signal = "BUY"
        elif predicted_class == 0 and layer_score <= 3:
            signal = "SELL"
        else:
            signal = "HOLD"
            
        target_date = today_date
        for _ in range(h_days):
            target_date = get_next_trading_day(target_date)
            
        pred_change = pred_close - today_close
        pred_change_pct = (pred_change / today_close) * 100
        pred_range_pct = ((pred_high - pred_low) / today_close) * 100 / 2.0
        
        predictions[h_name] = {
            "date": target_date.strftime("%Y-%m-%d"),
            "predicted_high": float(pred_high),
            "predicted_low": float(pred_low),
            "predicted_range_percent": float(pred_range_pct),
            "predicted_close": float(pred_close),
            "predicted_change": float(pred_change),
            "predicted_change_percent": float(pred_change_pct),
            "direction": predicted_dir,
            "probability": float(direction_confidence),
            "signal": signal,
            "reasons": [
                f"Classification Probabilities: DOWN {probs_aligned[0]:.1%}, FLAT {probs_aligned[1]:.1%}, UP {probs_aligned[2]:.1%}",
                f"Regression Range: {pred_low:.1f} to {pred_high:.1f}, 7-Layer Score: {layer_score}/7"
            ]
        }
        
        metrics[h_name] = {
            "mae": float(mae),
            "mape_percent": float(mape),
            "accuracy_percent": float(cv_acc * 100),
            "baseline_accuracy_percent": float(cv_base * 100),
            "skill_score": float(cv_skill)
        }

    chart_df = df.iloc[-60:].copy()
    chart_dates = [d.strftime("%Y-%m-%d") for d in chart_df["Date"]]
    chart_actuals = [float(x) for x in chart_df["Close"]]
    chart_sma20 = [float(x) if not pd.isna(x) else None for x in chart_df["BB_Middle"]]
    chart_bb_upper = [float(x) if not pd.isna(x) else None for x in chart_df["BB_Upper"]]
    chart_bb_lower = [float(x) if not pd.isna(x) else None for x in chart_df["BB_Lower"]]
    chart_volumes = [int(x) if not pd.isna(x) else 0 for x in chart_df["Volume"]]
    
    # ── Generate AI Model vs Actual (back-test on 1D test set) ─────────
    # Build a lookup: date -> predicted close for test set dates
    chart_predicted = [None] * len(chart_dates)
    if rf_reg_1d_high is not None and test_data_1d is not None and len(test_data_1d) > 0:
        try:
            bt_data = test_data_1d.copy()
            bt_X = bt_data[features]
            bt_rf_high  = rf_reg_1d_high.predict(bt_X)
            bt_xgb_high = xgb_reg_1d_high.predict(bt_X)
            bt_rf_low  = rf_reg_1d_low.predict(bt_X)
            bt_xgb_low = xgb_reg_1d_low.predict(bt_X)
            bt_pred_returns = ((bt_rf_high + bt_xgb_high) / 2.0 + (bt_rf_low + bt_xgb_low) / 2.0) / 2.0

            # Convert predicted return -> predicted next-day close
            # The prediction for row i is: "what will close be the next trading day?"
            # So we store it under (date[i] + 1 trading day)
            pred_map = {}
            bt_date_list = list(bt_data["Date"])
            for i, dt in enumerate(bt_date_list):
                row_close = float(bt_data.iloc[i]["Close"])
                pred_c = row_close * (1.0 + float(bt_pred_returns[i]))
                # Advance to next trading day (skip weekend)
                next_dt = dt + pd.Timedelta(days=1)
                while next_dt.weekday() in [5, 6]:
                    next_dt += pd.Timedelta(days=1)
                pred_map[next_dt.strftime("%Y-%m-%d")] = round(pred_c, 2)

            # Fill chart_predicted aligned with chart_dates
            for i, cd in enumerate(chart_dates):
                if cd in pred_map:
                    chart_predicted[i] = pred_map[cd]
        except Exception as e:
            print(f"[warn] chart predicted generation failed: {e}")
    
    def sfloat(col, default=0.0):
        if col in inference_row_original.columns:
            val = inference_row_original[col].iloc[0]
            if not pd.isna(val): return float(val)
        return default

    results = {
        "symbol": symbol.upper(),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_data": {
            "date": today_date.strftime("%Y-%m-%d"),
            "open": float(inference_row_original["Open"].iloc[0]),
            "high": float(inference_row_original["High"].iloc[0]),
            "low": float(inference_row_original["Low"].iloc[0]),
            "close": float(today_close),
            "volume": int(inference_row_original["Volume"].iloc[0]) if "Volume" in inference_row_original.columns else 0
        },
        "predictions": predictions,
        "metrics": metrics,
        "indicators": {
            "rsi_14":         sfloat("RSI_14", 50.0),
            "macd":           sfloat("MACD", 0.0),
            "macd_signal":    sfloat("MACD_Signal", 0.0),
            "macd_hist":      sfloat("MACD_Hist", 0.0),
            "atr_14":         sfloat("ATR_14", 0.0),
            "bb_middle":      sfloat("BB_Middle", 0.0),
            "bb_upper":       sfloat("BB_Upper", 0.0),
            "bb_lower":       sfloat("BB_Lower", 0.0),
            "bb_width":       sfloat("BB_Width", 0.0),
            "adx":            sfloat("ADX", 0.0),
            "plus_di":        sfloat("Plus_DI", 0.0),
            "minus_di":       sfloat("Minus_DI", 0.0),
            "ema_20":         sfloat("EMA_20", 0.0),
            "ema_50":         sfloat("EMA_50", 0.0),
            "ema_200":        sfloat("EMA_200", 0.0),
            "supertrend":     sfloat("Supertrend", 0.0),
            "supertrend_dir": sfloat("Supertrend_Direction", 0.0),
            "vwap":           sfloat("VWAP", 0.0),
            "cmf":            sfloat("CMF", 0.0),
            "obv":            sfloat("OBV", 0.0),
        },
        "chart_data": {
            "dates":     chart_dates,
            "actual":    chart_actuals,
            "predicted": chart_predicted,
            "sma20":     chart_sma20,
            "bb_upper":  chart_bb_upper,
            "bb_lower":  chart_bb_lower,
            "volume":    chart_volumes
        },
        "7_layer_analysis": analysis_results
    }
    
    HISTORY_FILE = os.path.join("outputs", "history", "prediction_history.json")
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []
        
    for h_name in predictions:
        pred = predictions[h_name]
        new_entry = {
            "id": f"{symbol.lower()}_{pred['date']}_{h_name}",
            "type": "stock",
            "symbol": symbol.upper(),
            "timeframe": h_name,
            "made_on": results["latest_data"]["date"],
            "prediction_date": pred["date"],
            "prev_close": results["latest_data"]["close"],
            "predicted_close": pred["predicted_close"],
            "predicted_high": pred.get("predicted_high"),
            "predicted_low": pred.get("predicted_low"),
            "predicted_range_percent": pred.get("predicted_range_percent"),
            "predicted_change": pred["predicted_change"],
            "predicted_change_pct": pred["predicted_change_percent"],
            "predicted_direction": pred["direction"],
            "confidence": pred["probability"],
            "signal": pred["signal"],
            "deep_analysis": pred.get("deep_analysis"),
            "status": "pending",
            "label": "⏳ Pending",
            "actual_close": None,
            "diff_points": None,
            "diff_percent": None,
            "direction_correct": None
        }
        # Preserve grading data from any existing entry for the same prediction target
        # (same symbol + timeframe + prediction_date, regardless of made_on time)
        existing = next(
            (e for e in history if
             e.get("symbol") == new_entry["symbol"] and
             e.get("timeframe") == new_entry["timeframe"] and
             e.get("prediction_date") == new_entry["prediction_date"]),
            None
        )
        if existing and existing.get("status") != "pending":
            # Already graded — preserve the grading results but update prediction values
            new_entry["status"]           = existing["status"]
            new_entry["label"]            = existing["label"]
            new_entry["actual_close"]     = existing["actual_close"]
            new_entry["diff_points"]      = existing["diff_points"]
            new_entry["diff_percent"]     = existing["diff_percent"]
            new_entry["direction_correct"]= existing["direction_correct"]

        # Remove ALL prior entries for same symbol+timeframe+prediction_date (upsert)
        history = [
            e for e in history if not (
                e.get("symbol") == new_entry["symbol"] and
                e.get("timeframe") == new_entry["timeframe"] and
                e.get("prediction_date") == new_entry["prediction_date"]
            )
        ]
        history.insert(0, new_entry)
        
    history = history[:1000] # Increased size
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, cls=NumpyEncoder)
        
    json_file = os.path.join("outputs", "predictions", "predictions.json")
    symbol_json_file = os.path.join("outputs", "predictions", f"{symbol.lower()}_predictions.json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    with open(symbol_json_file, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
        
    return results

def main():
    args = parse_args()
    symbol = args.symbol.upper()
    
    # 1. Clean raw CSV
    try:
        df_cleaned = clean_data(symbol)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        # Check if a cleaned CSV already exists as fallback
        clean_file = os.path.join("data", f"{symbol.lower()}_cleaned.csv")
        if os.path.exists(clean_file):
            print(f"Fallback: Found cleaned file {clean_file}. Loading existing data...")
            df_cleaned = pd.read_csv(clean_file)
            df_cleaned["Date"] = pd.to_datetime(df_cleaned["Date"])
        else:
            sys.exit(1)
            
    # 2. Calculate Indicators
    df_features = calculate_technical_indicators(df_cleaned)
    df_features = df_features.reset_index(drop=True)  # Ensure contiguous integer index
    
    # 3. Train & Predict
    results = train_and_predict(symbol, df_features)
    
    
    print("\n==========================================")
    print(f"PREDICTION SUMMARY FOR {symbol.upper()}")
    print(f"Current Price: {results['latest_data']['close']:.2f}")
    for h_name, p in results['predictions'].items():
        print(f"[{h_name}] -> {p['date']}: {p['predicted_close']:.2f} ({p['predicted_change_percent']:+.2f}%) | Dir: {p['direction']} (Conf: {p['probability']*100:.1f}%) | Sig: {p['signal']}")
    print("==========================================")
    
    # ---------------------------
    # Deep Contextual AI Analysis
    # ---------------------------
    from deep_analyst import generate_executive_summary
    print("Generating Deep Contextual Summary via LLM...")
    try:
        deep_res = generate_executive_summary(symbol, f"outputs/analysis/{symbol.lower()}_analysis.json")
        if deep_res:
            print("  [✓] Deep Analysis generated successfully.")
            for h_name in results['predictions']:
                results['predictions'][h_name]["deep_analysis"] = deep_res
        else:
            print("  [-] Deep Analysis skipped.")
    except Exception as e:
        print(f"  [x] Deep Analysis failed: {e}")

    # Generate final analysis JSON output
    # (Re-dumping results after adding deep_analysis)
    json_file = os.path.join("outputs", "predictions", "predictions.json")
    symbol_json_file = os.path.join("outputs", "predictions", f"{symbol.lower()}_predictions.json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    with open(symbol_json_file, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)


if __name__ == "__main__":
    main()