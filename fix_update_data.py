import os

with open("update_data.py", "r") as f: content = f.read()
content = content.replace('outfile = "nepse_index_data.csv" if symbol == "NEPSE" else f"{symbol.lower()}_data.csv"', 'outfile = os.path.join("data", "nepse_index_data.csv") if symbol == "NEPSE" else os.path.join("data", f"{symbol.lower()}_data.csv")')
with open("update_data.py", "w") as f: f.write(content)
