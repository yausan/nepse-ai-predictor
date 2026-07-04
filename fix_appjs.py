with open("app.js", "r") as f: content = f.read()

# Update fetch paths for JSON files
content = content.replace("fetch('/nepse_predictions.json')", "fetch('/outputs/predictions/nepse_predictions.json')")
content = content.replace("fetch(`/${sym.toLowerCase()}_predictions.json`)", "fetch(`/outputs/predictions/${sym.toLowerCase()}_predictions.json`)")
content = content.replace("fetch(`/${sym.toLowerCase()}_analysis.json`)", "fetch(`/outputs/analysis/${sym.toLowerCase()}_analysis.json`)")

with open("app.js", "w") as f: f.write(content)
print("app.js paths fixed successfully")
