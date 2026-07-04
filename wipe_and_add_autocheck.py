import json
import os
import re

# 1. Wipe History and reset to pending
HISTORY_FILE = "outputs/history/prediction_history.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    for entry in history:
        entry["status"] = "pending"
        entry["label"] = "⏳ Pending"
        entry["actual_close"] = None
        entry["diff_points"] = None
        entry["diff_percent"] = None
        entry["direction_correct"] = None
        entry["resolved_on"] = None
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Wiped {len(history)} entries back to pending.")

# 2. Add autocheck to server.py
with open("server.py", "r") as f:
    server_code = f.read()

autocheck_code = """
    # ── History: Auto-check all pending ─────────────────────────────────
    def handle_history_autocheck(self, payload):
        import pandas as pd
        HISTORY_FILE = os.path.join("outputs", "history", "prediction_history.json")
        if not os.path.exists(HISTORY_FILE):
            self.send_json_response(200, {"status": "success", "resolved": 0})
            return
            
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
            
        from datetime import datetime
        resolved_count = 0
        today = datetime.now().strftime("%Y-%m-%d")
        
        for entry in history:
            if entry["status"] != "pending":
                continue
            if entry["prediction_date"] > today:
                continue
                
            symbol = entry["symbol"]
            pred_date = entry["prediction_date"]
            
            # Load CSV to check actual close
            if symbol == "NEPSE":
                csv_file = os.path.join("data", "nepse_data.csv")
            else:
                csv_file = os.path.join("data", f"{symbol.lower()}_cleaned.csv")
                
            if not os.path.exists(csv_file):
                continue
                
            df = pd.read_csv(csv_file)
            df = df.sort_values(by="Date", ascending=True)
            # Find the row on or immediately after the prediction_date
            future_df = df[df["Date"] >= pred_date]
            if len(future_df) == 0:
                continue
            
            actual_row = future_df.iloc[0]
            actual_close = float(actual_row["Close"])
            actual_date = actual_row["Date"]
            
            # Assign grade
            pred_close = float(entry["predicted_close"])
            prev_close = float(entry["prev_close"])
            diff_pts = actual_close - prev_close
            
            actual_dir = "UP" if actual_close > prev_close else "DOWN"
            pred_dir = entry["predicted_direction"]
            
            if entry["timeframe"] == "1D" and actual_date != pred_date:
                # If market was closed on pred_date and 1D target, we accept the next day
                pass
                
            is_correct = (actual_dir == pred_dir)
            
            diff_percent = ((actual_close - pred_close) / pred_close) * 100
            
            entry["actual_close"] = actual_close
            entry["diff_points"] = diff_pts
            entry["diff_percent"] = diff_percent
            entry["direction_correct"] = is_correct
            entry["resolved_on"] = actual_date
            
            if abs(diff_percent) < 0.5:
                entry["status"] = "exact"
                entry["label"] = "✅ Exact"
            elif is_correct:
                entry["status"] = "correct"
                entry["label"] = "✓ Correct"
            else:
                entry["status"] = "wrong"
                entry["label"] = "❌ Wrong"
                
            resolved_count += 1
            
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
            
        self.send_json_response(200, {"status": "success", "resolved": resolved_count})
"""

# Replace handle_history_resolve with handle_history_autocheck
server_code = re.sub(
    r'    # ── History: Resolve a pending entry ────────────────────────────────.*?(?=    # ── History: Delete an entry ─────────────────────────────────────────)',
    autocheck_code + '\n',
    server_code,
    flags=re.DOTALL
)

# Wire the endpoint in do_POST
server_code = server_code.replace(
    'if path == "/api/history/resolve":\n            self.handle_history_resolve(payload)',
    'if path == "/api/history/autocheck":\n            self.handle_history_autocheck(payload)'
)

with open("server.py", "w") as f:
    f.write(server_code)
    
print("Added autocheck to server.py")
