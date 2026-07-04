import os, glob

# 1. analysis.py
with open("analysis.py", "r") as f: content = f.read()
content = content.replace('f"{symbol.lower()}_data.csv"', 'os.path.join("data", f"{symbol.lower()}_data.csv")')
content = content.replace('f"{symbol.lower()}_cleaned.csv"', 'os.path.join("data", f"{symbol.lower()}_cleaned.csv")')
content = content.replace('"nepse_index_data.csv"', 'os.path.join("data", "nepse_index_data.csv")')
content = content.replace('f"{symbol.lower()}_analysis.json"', 'os.path.join("outputs", "analysis", f"{symbol.lower()}_analysis.json")')
with open("analysis.py", "w") as f: f.write(content)

# 2. train_model.py
with open("train_model.py", "r") as f: content = f.read()
content = content.replace('f"{symbol.lower()}_data.csv"', 'os.path.join("data", f"{symbol.lower()}_data.csv")')
content = content.replace('f"{symbol.lower()}_cleaned.csv"', 'os.path.join("data", f"{symbol.lower()}_cleaned.csv")')
content = content.replace('"predictions.json"', 'os.path.join("outputs", "predictions", "predictions.json")')
content = content.replace('f"{symbol.lower()}_predictions.json"', 'os.path.join("outputs", "predictions", f"{symbol.lower()}_predictions.json")')
content = content.replace('HISTORY_FILE = "prediction_history.json"', 'HISTORY_FILE = os.path.join("outputs", "history", "prediction_history.json")')
with open("train_model.py", "w") as f: f.write(content)

# 3. predict_nepse.py
with open("predict_nepse.py", "r") as f: content = f.read()
content = content.replace('"nepse_index_data.csv"', 'os.path.join("data", "nepse_index_data.csv")')
content = content.replace('PRED_DIR = "predictions"', 'PRED_DIR = os.path.join("outputs", "predictions")')
content = content.replace('HISTORY_FILE = "prediction_history.json"', 'HISTORY_FILE = os.path.join("outputs", "history", "prediction_history.json")')
content = content.replace('NEPSE_JSON = "nepse_predictions.json"', 'NEPSE_JSON = os.path.join("outputs", "predictions", "nepse_predictions.json")')
content = content.replace('history_file = os.path.join(PRED_DIR, "nepse_history.json")', 'history_file = os.path.join("outputs", "history", "nepse_history.json")')
with open("predict_nepse.py", "w") as f: f.write(content)

# 4. update_data.py
with open("update_data.py", "r") as f: content = f.read()
content = content.replace('csv_filename = f"{symbol.lower()}_data.csv"', 'csv_filename = os.path.join("data", f"{symbol.lower()}_data.csv")')
content = content.replace('csv_filename = "nepse_index_data.csv"', 'csv_filename = os.path.join("data", "nepse_index_data.csv")')
with open("update_data.py", "w") as f: f.write(content)

# 5. nepsealpha_scraper.py
with open("nepsealpha_scraper.py", "r") as f: content = f.read()
content = content.replace('f"{symbol.lower()}_data.csv"', 'os.path.join("data", f"{symbol.lower()}_data.csv")')
content = content.replace('"nepse_index_data.csv"', 'os.path.join("data", "nepse_index_data.csv")')
with open("nepsealpha_scraper.py", "w") as f: f.write(content)

# 6. server.py
with open("server.py", "r") as f: content = f.read()
content = content.replace('pred_file = f"{sym_lower}_predictions.json"', 'pred_file = os.path.join("outputs", "predictions", f"{sym_lower}_predictions.json")')
content = content.replace('pred_file_alt = os.path.join("predictions", f"{sym_lower}_history.json")', 'pred_file_alt = os.path.join("outputs", "history", f"{sym_lower}_history.json")')
content = content.replace('nepse_pred_exists = os.path.exists("nepse_predictions.json")', 'nepse_pred_exists = os.path.exists(os.path.join("outputs", "predictions", "nepse_predictions.json"))')
content = content.replace('nepse_history_alt = os.path.join("predictions", "nepse_history.json")', 'nepse_history_alt = os.path.join("outputs", "history", "nepse_history.json")')
content = content.replace('nepse_ts_file = "nepse_predictions.json" if nepse_pred_exists', 'nepse_ts_file = os.path.join("outputs", "predictions", "nepse_predictions.json") if nepse_pred_exists')
content = content.replace('pred_file = f"{symbol.lower()}_predictions.json"', 'pred_file = os.path.join("outputs", "predictions", f"{symbol.lower()}_predictions.json")')
content = content.replace('os.path.exists("nepse_predictions.json")', 'os.path.exists(os.path.join("outputs", "predictions", "nepse_predictions.json"))')
content = content.replace('with open("nepse_predictions.json")', 'with open(os.path.join("outputs", "predictions", "nepse_predictions.json"))')
content = content.replace('history_file = os.path.join("predictions", f"{symbol.lower()}_history.json")', 'history_file = os.path.join("outputs", "history", f"{symbol.lower()}_history.json")')
content = content.replace('analysis_file = f"{symbol.lower()}_analysis.json"', 'analysis_file = os.path.join("outputs", "analysis", f"{symbol.lower()}_analysis.json")')
content = content.replace('HISTORY_FILE = "prediction_history.json"', 'HISTORY_FILE = os.path.join("outputs", "history", "prediction_history.json")')
with open("server.py", "w") as f: f.write(content)

print("Paths fixed successfully!")
