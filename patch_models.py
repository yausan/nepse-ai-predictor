import re

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 1. Track 1D models
    content = content.replace('rf_reg_1d = None\n    xgb_reg_1d = None',
                              'rf_reg_1d_high = None\n    xgb_reg_1d_high = None\n    rf_reg_1d_low = None\n    xgb_reg_1d_low = None')
    
    # 2. Target preparation
    target_prep_orig = 'df_h["Target_Return"] = df_h["Close"].pct_change(periods=h_days).shift(-h_days)'
    target_prep_new = (target_prep_orig + '\n' +
                       '        df_h["Target_High_Return"] = (df_h["High"].shift(-h_days) - df_h["Close"]) / df_h["Close"]\n' +
                       '        df_h["Target_Low_Return"] = (df_h["Low"].shift(-h_days) - df_h["Close"]) / df_h["Close"]')
    content = content.replace(target_prep_orig, target_prep_new)
    
    # 3. Dropna
    content = content.replace('["Target_Return", "Target_Class"]', '["Target_Return", "Target_High_Return", "Target_Low_Return", "Target_Class"]')
    
    # 4. Y train/test variables
    y_vars_orig = 'y_train_return = train_data["Target_Return"]\n        y_train_class = train_data["Target_Class"].astype(int)'
    y_vars_new = (y_vars_orig.replace('y_train_class', 'y_train_high = train_data["Target_High_Return"]\n        y_train_low = train_data["Target_Low_Return"]\n        y_train_class'))
    content = content.replace(y_vars_orig, y_vars_new)
    
    y_test_orig = 'y_test_return = test_data["Target_Return"]\n        y_test_class = test_data["Target_Class"].astype(int)'
    y_test_new = (y_test_orig.replace('y_test_class', 'y_test_high = test_data["Target_High_Return"]\n        y_test_low = test_data["Target_Low_Return"]\n        y_test_class'))
    content = content.replace(y_test_orig, y_test_new)
    
    # 5. Regressor Training
    reg_orig = '''        # 1. Regressor (predicts return percentage without price bias)
        rf_reg = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1).fit(X_train, y_train_return)
        xgb_reg = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1).fit(X_train, y_train_return)
        
        rf_reg_pred = rf_reg.predict(X_test)
        xgb_reg_pred = xgb_reg.predict(X_test)
        ensemble_reg_pred = (rf_reg_pred + xgb_reg_pred) / 2.0'''
        
    reg_new = '''        # 1. Regressor (predicts High/Low return percentage without price bias)
        rf_high_reg = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1).fit(X_train, y_train_high)
        xgb_high_reg = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1).fit(X_train, y_train_high)
        
        rf_low_reg = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1).fit(X_train, y_train_low)
        xgb_low_reg = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1).fit(X_train, y_train_low)
        
        ensemble_high_pred = (rf_high_reg.predict(X_test) + xgb_high_reg.predict(X_test)) / 2.0
        ensemble_low_pred = (rf_low_reg.predict(X_test) + xgb_low_reg.predict(X_test)) / 2.0
        ensemble_reg_pred = (ensemble_high_pred + ensemble_low_pred) / 2.0'''
    content = content.replace(reg_orig, reg_new)
    
    # 6. Saving 1D models
    save_1d_orig = '''            rf_reg_1d = rf_reg
            xgb_reg_1d = xgb_reg'''
    save_1d_new = '''            rf_reg_1d_high = rf_high_reg
            xgb_reg_1d_high = xgb_high_reg
            rf_reg_1d_low = rf_low_reg
            xgb_reg_1d_low = xgb_low_reg'''
    content = content.replace(save_1d_orig, save_1d_new)
    
    # 7. Inference
    inf_orig = '''        pred_return = (rf_reg.predict(X_inf)[0] + xgb_reg.predict(X_inf)[0]) / 2.0
        pred_close = today_close * (1.0 + pred_return)'''
    inf_new = '''        pred_high_return = (rf_high_reg.predict(X_inf)[0] + xgb_high_reg.predict(X_inf)[0]) / 2.0
        pred_low_return = (rf_low_reg.predict(X_inf)[0] + xgb_low_reg.predict(X_inf)[0]) / 2.0
        
        pred_high = today_close * (1.0 + pred_high_return)
        pred_low = today_close * (1.0 + pred_low_return)
        pred_close = (pred_high + pred_low) / 2.0
        pred_return = (pred_close - today_close) / today_close'''
    content = content.replace(inf_orig, inf_new)
    
    # 8. Chart Inference
    chart_inf_orig = '''            bt_rf  = rf_reg_1d.predict(bt_X)
            bt_xgb = xgb_reg_1d.predict(bt_X)
            bt_pred_returns = (bt_rf + bt_xgb) / 2.0'''
    chart_inf_new = '''            bt_rf_high  = rf_reg_1d_high.predict(bt_X)
            bt_xgb_high = xgb_reg_1d_high.predict(bt_X)
            bt_rf_low  = rf_reg_1d_low.predict(bt_X)
            bt_xgb_low = xgb_reg_1d_low.predict(bt_X)
            bt_pred_returns = ((bt_rf_high + bt_xgb_high) / 2.0 + (bt_rf_low + bt_xgb_low) / 2.0) / 2.0'''
    content = content.replace(chart_inf_orig, chart_inf_new)
    content = content.replace('if rf_reg_1d is not None and', 'if rf_reg_1d_high is not None and')
    
    # 9. Predictions dict
    pred_dict_orig = '''        pred_change_pct = (pred_change / today_close) * 100
        
        predictions[h_name] = {
            "date": target_date.strftime("%Y-%m-%d"),
            "predicted_close": float(pred_close),'''
    pred_dict_new = '''        pred_change_pct = (pred_change / today_close) * 100
        pred_range_pct = ((pred_high - pred_low) / today_close) * 100 / 2.0
        
        predictions[h_name] = {
            "date": target_date.strftime("%Y-%m-%d"),
            "predicted_high": float(pred_high),
            "predicted_low": float(pred_low),
            "predicted_range_percent": float(pred_range_pct),
            "predicted_close": float(pred_close),'''
    content = content.replace(pred_dict_orig, pred_dict_new)
    
    reasons_orig = 'f"Regression Est Return: {pred_return*100:+.2f}%, 7-Layer Score: {layer_score}/7"'
    reasons_new = 'f"Regression Range: {pred_low:.1f} to {pred_high:.1f}, 7-Layer Score: {layer_score}/7"'
    content = content.replace(reasons_orig, reasons_new)

    # 10. History Entry
    hist_orig = '''            "prev_close": float(today_close),
            "predicted_close": float(pred['predicted_close']),
            "predicted_direction": pred['direction'],'''
    hist_new = '''            "prev_close": float(today_close),
            "predicted_close": float(pred['predicted_close']),
            "predicted_high": float(pred['predicted_high']),
            "predicted_low": float(pred['predicted_low']),
            "predicted_range_percent": float(pred.get('predicted_range_percent', 0.0)),
            "predicted_direction": pred['direction'],'''
    content = content.replace(hist_orig, hist_new)

    with open(filepath, 'w') as f:
        f.write(content)

patch_file("train_model.py")
patch_file("predict_nepse.py")
