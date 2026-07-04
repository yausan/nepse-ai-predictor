import re

with open("train_model.py", "r") as f:
    content = f.read()

# We need to replace train_and_predict to loop over horizons
new_train_and_predict = """
def train_and_predict(symbol, df_features):
    \"\"\"Train models and make predictions for 1D, 5D, 20D horizons\"\"\"
    features = [
        "SMA_20", "SMA_50", "SMA_200", "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Upper", "BB_Middle", "BB_Lower", "ATR_14", "Volatility_20", "Volume_Change",
        "OBV", "ADX", "Daily_Return"
    ]
    has_volume = "Volume" in df_features.columns
    
    # Add lag columns
    df = df_features.copy()
    for lag in [1, 2, 3, 5, 10]:
        for prefix in ["Close_Lag", "Return_Lag"]:
            col = f"{prefix}_{lag}"
            if col in df.columns and col not in features:
                features.append(col)
        if has_volume:
            col = f"Volume_Lag_{lag}"
            if col in df.columns and col not in features:
                features.append(col)

    features = [f for f in features if f in df.columns and df[f].notna().sum() > 50]
    print(f"  Using {len(features)} ML features")

    horizons = {"1D": 1, "5D": 5, "20D": 20}
    predictions = {}
    metrics = {}
    
    analysis_results = run_analysis(symbol)
    
    inference_row_original = df.iloc[[-1]].copy()
    today_close = float(inference_row_original["Close"].iloc[0])
    today_date = inference_row_original["Date"].iloc[0]
    
    # Extract 7-layer score
    checklist = analysis_results.get("checklist", {})
    layer_score = checklist.get("score", 0)
    
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from xgboost import XGBRegressor, XGBClassifier
    from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score
    import numpy as np

    # Run for each horizon
    for h_name, h_days in horizons.items():
        print(f"\\n--- Training Horizon {h_name} ({h_days} days) ---")
        
        # Set targets for this horizon
        df_h = df.copy()
        # Return over next h_days
        df_h["Target_Return"] = df_h["Close"].pct_change(periods=h_days).shift(-h_days)
        df_h["Target_Direction"] = (df_h["Target_Return"] > 0).astype(int)
        
        df_h = df_h.reset_index(drop=True)
        inference_row = df_h.iloc[[-1]].copy()
        train_test_data = df_h.iloc[:-h_days].copy() # Drop last h_days because target is NaN
        
        train_test_data = train_test_data.dropna(subset=features + ["Target_Return", "Target_Direction"])
        
        if len(train_test_data) == 0:
            train_test_data = df_h.iloc[:-h_days].copy()
            train_test_data[features] = train_test_data[features].ffill().bfill()
            train_test_data = train_test_data.dropna(subset=["Target_Return", "Target_Direction"])
            
        split_idx = int(len(train_test_data) * 0.8)
        train_data = train_test_data.iloc[:split_idx]
        test_data = train_test_data.iloc[split_idx:]
        
        if len(train_data) < 10 or len(test_data) < 1:
            print(f"Not enough data for {h_name}")
            continue
            
        X_train = train_data[features]
        y_train_return = train_data["Target_Return"]
        y_train_dir = train_data["Target_Direction"]
        X_test = test_data[features]
        y_test_return = test_data["Target_Return"]
        y_test_dir = test_data["Target_Direction"]
        
        tscv = TimeSeriesSplit(n_splits=3)
        rf_param_grid = {'n_estimators': [50], 'max_depth': [5, 10], 'min_samples_split': [5]}
        xgb_param_grid = {'n_estimators': [50], 'max_depth': [3, 6], 'learning_rate': [0.1]}
        
        # Regression
        rf_reg = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1).fit(X_train, y_train_return)
        xgb_reg = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1).fit(X_train, y_train_return)
        
        rf_reg_pred = rf_reg.predict(X_test)
        xgb_reg_pred = xgb_reg.predict(X_test)
        ensemble_reg_pred = (rf_reg_pred + xgb_reg_pred) / 2.0
        mae = mean_absolute_error(y_test_return, ensemble_reg_pred)
        
        # Classification
        rf_clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1).fit(X_train, y_train_dir)
        xgb_clf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1).fit(X_train, y_train_dir)
        
        rf_clf_prob = rf_clf.predict_proba(X_test)[:, 1]
        xgb_clf_prob = xgb_clf.predict_proba(X_test)[:, 1]
        ensemble_clf_prob = (rf_clf_prob + xgb_clf_prob) / 2.0
        ensemble_clf_pred = (ensemble_clf_prob >= 0.5).astype(int)
        
        acc = accuracy_score(y_test_dir, ensemble_clf_pred)
        prec = precision_score(y_test_dir, ensemble_clf_pred, zero_division=0)
        rec = recall_score(y_test_dir, ensemble_clf_pred, zero_division=0)
        
        # Inference
        X_inf = inference_row[features]
        pred_return = (rf_reg.predict(X_inf)[0] + xgb_reg.predict(X_inf)[0]) / 2.0
        
        # Calculate target date by adding trading days
        target_date = today_date
        for _ in range(h_days):
            target_date = get_next_trading_day(target_date)
            
        pred_close = today_close * (1.0 + pred_return)
        
        bullish_prob = (rf_clf.predict_proba(X_inf)[0, 1] + xgb_clf.predict_proba(X_inf)[0, 1]) / 2.0
        direction_confidence = bullish_prob if bullish_prob >= 0.50 else (1 - bullish_prob)
        
        # Simple signal assignment based on 7-layer score
        # The larger the timeframe, the larger the return should be to trigger a BUY
        thresh = { "1D": 0.5, "5D": 1.5, "20D": 4.0 }[h_name]
        
        pred_change = pred_close - today_close
        pred_change_pct = (pred_change / today_close) * 100
        
        if pred_change_pct > thresh and layer_score >= 4:
            signal = "BUY"
        elif pred_change_pct < -thresh and layer_score <= 3:
            signal = "SELL"
        else:
            signal = "HOLD"
            
        predicted_dir = "UP" if pred_close > today_close else "DOWN"
        
        predictions[h_name] = {
            "date": target_date.strftime("%Y-%m-%d"),
            "predicted_close": float(pred_close),
            "predicted_change": float(pred_change),
            "predicted_change_percent": float(pred_change_pct),
            "direction": predicted_dir,
            "probability": float(direction_confidence),
            "signal": signal,
            "reasons": [f"ML Ensemble Return: {pred_return*100:.2f}%, 7-Layer Score: {layer_score}/7"]
        }
        metrics[h_name] = {
            "mae": float(mae),
            "accuracy_percent": float(acc * 100),
            "precision_percent": float(prec * 100),
            "recall_percent": float(rec * 100)
        }

    # Chart data for 1D test
    chart_df = df.iloc[-60:].copy()
    chart_dates = [d.strftime("%Y-%m-%d") for d in chart_df["Date"]]
    chart_actuals = [float(x) for x in chart_df["Close"]]
    chart_sma20 = [float(x) if not pd.isna(x) else None for x in chart_df["BB_Middle"]]
    chart_bb_upper = [float(x) if not pd.isna(x) else None for x in chart_df["BB_Upper"]]
    chart_bb_lower = [float(x) if not pd.isna(x) else None for x in chart_df["BB_Lower"]]
    chart_volumes = [int(x) if not pd.isna(x) else 0 for x in chart_df["Volume"]]
    
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
            "rsi_14": float(inference_row_original["RSI_14"].iloc[0]),
            "macd": float(inference_row_original["MACD"].iloc[0]),
            "macd_signal": float(inference_row_original["MACD_Signal"].iloc[0]),
            "bb_middle": float(inference_row_original["BB_Middle"].iloc[0])
        },
        "chart_data": {
            "dates": chart_dates,
            "actual": chart_actuals,
            "sma20": chart_sma20,
            "bb_upper": chart_bb_upper,
            "bb_lower": chart_bb_lower,
            "volume": chart_volumes
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
            "predicted_change": pred["predicted_change"],
            "predicted_change_pct": pred["predicted_change_percent"],
            "predicted_direction": pred["direction"],
            "confidence": pred["probability"],
            "signal": pred["signal"],
            "status": "pending",
            "label": "⏳ Pending",
            "actual_close": None,
            "diff_points": None,
            "diff_percent": None,
            "direction_correct": None
        }
        # Replace existing
        history = [e for e in history if e.get("id") != new_entry["id"]]
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
"""

# Replace train_and_predict block using regex
import re
new_content = re.sub(r'def train_and_predict.*?def main\(\):', new_train_and_predict + '\ndef main():', content, flags=re.DOTALL)

# Also fix the printing in main to show multi-timeframe
main_print_replacement = """
    print("\\n==========================================")
    print(f"PREDICTION SUMMARY FOR {symbol.upper()}")
    print(f"Current Price: {results['latest_data']['close']:.2f}")
    for h_name, p in results['predictions'].items():
        print(f"[{h_name}] -> {p['date']}: {p['predicted_close']:.2f} ({p['predicted_change_percent']:+.2f}%) | Dir: {p['direction']} (Conf: {p['probability']*100:.1f}%) | Sig: {p['signal']}")
    print("==========================================\\n")
"""
new_content = re.sub(r'print\("\\n=========================================="\).*?print\("==========================================\\n"\)', main_print_replacement, new_content, flags=re.DOTALL)

with open("train_model.py", "w") as f:
    f.write(new_content)

print("train_model.py successfully refactored!")
