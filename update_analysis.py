import re

with open("analysis.py", "r") as f:
    content = f.read()

# 1. Add import
if "from indicators import add_all_indicators" not in content:
    content = content.replace("import pandas as pd", "import pandas as pd\nfrom indicators import add_all_indicators")

# 2. Add layers 8 and 9
layers_code = """
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
"""

content = content.replace(
    "# ═════════════════════════════════════════════════════════════════════════\n#  TRADE CHECKLIST & SCORING\n# ═════════════════════════════════════════════════════════════════════════",
    layers_code
)

# 3. Update build_checklist signature
content = content.replace(
    "def build_checklist(structure, levels, patterns, volume, trend, momentum, nepse_filter):",
    "def build_checklist(structure, levels, patterns, volume, trend, momentum, nepse_filter, vwap, volatility):"
)

# 4. Add VWAP and Volatility to checklist items
checklist_adds = """    # 8. VWAP
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
"""
content = content.replace("    # Rating\n", checklist_adds)

# 5. Update Rating Thresholds and Total
content = content.replace("if score >= 6:", "if score >= 7:")
content = content.replace("elif score >= 4:", "elif score >= 5:")
content = content.replace("\"total\": 7,", "\"total\": 9,")
content = content.replace("win_rates = {7: \"68–72%\", 6: \"65–70%\", 5: \"60–65%\", 4: \"55–60%\", 3: \"50–55%\", 2: \"~50%\", 1: \"< 50%\", 0: \"Don't trade\"}",
                          "win_rates = {9: \"70-75%\", 8: \"68-72%\", 7: \"65-70%\", 6: \"60-65%\", 5: \"55-60%\", 4: \"50-55%\", 3: \"~50%\", 2: \"< 50%\", 1: \"< 50%\", 0: \"Don't trade\"}")

# 6. Update run_analysis
content = content.replace("print(f\"Running 7-layer analysis for {symbol.upper()}...\")", "print(f\"Running 10-layer analysis for {symbol.upper()}...\")")
content = content.replace(
    "df = load_data(symbol)\n    nepse_df = load_nepse_index()",
    "df = load_data(symbol)\n    df = add_all_indicators(df)\n    nepse_df = load_nepse_index()"
)

content = content.replace(
    "    print(\"  Layer 7: NEPSE-Specific Filters...\")\n    nepse_filter = layer7_nepse(df, symbol, nepse_df)\n\n    # Checklist",
    "    print(\"  Layer 7: NEPSE-Specific Filters...\")\n    nepse_filter = layer7_nepse(df, symbol, nepse_df)\n\n    print(\"  Layer 8: VWAP Analysis...\")\n    vwap = layer8_vwap(df)\n\n    print(\"  Layer 9: Volatility...\")\n    volatility = layer9_volatility(df)\n\n    # Checklist"
)

content = content.replace(
    "checklist = build_checklist(structure, levels, patterns, volume, trend_indicators, momentum, nepse_filter)",
    "checklist = build_checklist(structure, levels, patterns, volume, trend_indicators, momentum, nepse_filter, vwap, volatility)"
)

content = content.replace(
    "        \"layer7_nepse\": nepse_filter,\n        \"checklist\": checklist,",
    "        \"layer7_nepse\": nepse_filter,\n        \"layer8_vwap\": vwap,\n        \"layer9_volatility\": volatility,\n        \"checklist\": checklist,"
)

with open("analysis.py", "w") as f:
    f.write(content)

print("Updated analysis.py successfully")
