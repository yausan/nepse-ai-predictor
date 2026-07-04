import re

with open("predict_nepse.py", "r") as f:
    content = f.read()
    
# Replace the old print block in the main execution
main_block_replacement = """
    print(f"NEPSE Index Prediction for Multi-Timeframe")
    print(f"Current Level: {result['latest_data']['close']:.2f}")
    
    for h_name, p in result['predictions'].items():
        print(f"[{h_name}] -> {p['date']}: {p['predicted_close']:.2f} ({p['predicted_change_percent']:+.2f}%) | Dir: {p['direction']} (Conf: {p['probability']*100:.1f}%) | Sig: {p['signal']}")
        
    print("==================================================")
"""

content = re.sub(
    r'    print\(f"NEPSE Index Prediction for \{result\[\'prediction\'\]\[\'date\'\]\}"\).*?print\("=================================================="\)',
    main_block_replacement,
    content,
    flags=re.DOTALL
)

with open("predict_nepse.py", "w") as f:
    f.write(content)
