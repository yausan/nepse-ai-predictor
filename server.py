import os
import sys
import json
import time
import subprocess
import http.server
import socketserver
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

load_dotenv()

PORT = int(os.environ.get("PORT", 8080))

class NEPSEPredictorHandler(http.server.SimpleHTTPRequestHandler):

    def log_message(self, format, *args):
        sys.stdout.write(f"[{self.log_date_time_string()}] {format % args}\n")

    def do_GET(self):
        parsed_url  = urlparse(self.path)
        path        = parsed_url.path
        query_params = parse_qs(parsed_url.query)

        routes = {
            "/api/status":          self.handle_status,
            "/api/predict":         self.handle_predict,
            "/api/update":          self.handle_update,
            "/api/nepse/update":    self.handle_nepse_update,
            "/api/nepse/predict":   self.handle_nepse_predict,
            "/api/history":         self.handle_history,
            "/api/history/stats":   self.handle_history_stats,
            "/api/stock/history":   self.handle_stock_history,
            "/api/analysis":        self.handle_analysis,
        }

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] RECEIVED GET: {path}")
        if path in routes:
            routes[path](query_params)
        else:
            super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_POST(self):
        parsed_url   = urlparse(self.path)
        path         = parsed_url.path
        length       = int(self.headers.get("Content-Length", 0))
        body         = self.rfile.read(length)
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        if path == "/api/history/autocheck":
            self.handle_history_autocheck(payload)
        elif path == "/api/history/delete":
            self.handle_history_delete(payload)
        elif path == "/api/chat":
            self.handle_chat(payload)
        else:
            self.send_json_response(404, {"error": "Not Found"})

    def send_json_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    # ── Stock Symbol Status ──────────────────────────────────────────────
    def handle_status(self, qp):
        symbol    = qp.get("symbol", ["LICN"])[0].upper()
        sym_lower = symbol.lower()
        
        # Check root folder first, then predictions/ subfolder
        pred_file = os.path.join("outputs", "predictions", f"{sym_lower}_predictions.json")
        pred_file_alt = os.path.join("outputs", "history", f"{sym_lower}_history.json")
        pred_exists = os.path.exists(pred_file) or os.path.exists(pred_file_alt)
        pred_ts_file = pred_file if os.path.exists(pred_file) else (pred_file_alt if os.path.exists(pred_file_alt) else None)
        
        nepse_pred_exists = os.path.exists(os.path.join("outputs", "predictions", "nepse_predictions.json"))
        nepse_history_alt = os.path.join("outputs", "history", "nepse_history.json")
        nepse_exists = nepse_pred_exists or os.path.exists(nepse_history_alt)
        nepse_ts_file = os.path.join("outputs", "predictions", "nepse_predictions.json") if nepse_pred_exists else (nepse_history_alt if os.path.exists(nepse_history_alt) else None)
        
        self.send_json_response(200, {
            "symbol":              symbol,
            "raw_data_exists":     pred_exists,
            "cleaned_data_exists": pred_exists,
            "predictions_exist":   pred_exists,
            "last_scraped":        time_modified_str(pred_ts_file) if pred_ts_file else "Never",
            "last_predicted":      time_modified_str(pred_ts_file) if pred_ts_file else "Never",
            "nepse_data_exists":   nepse_exists,
            "nepse_last_scraped":  time_modified_str(nepse_ts_file) if nepse_ts_file else "Never",
            "nepse_last_predicted": time_modified_str(nepse_ts_file) if nepse_ts_file else "Never",
        })

    # ── Stock Predict ────────────────────────────────────────────────────
    def handle_predict(self, qp):
        symbol = qp.get("symbol", ["LICN"])[0].upper()
        print(f"\n[API] Retraining stock model for {symbol}...")
        result = self._run_script(["train_model.py", "--symbol", symbol], timeout=180)
        if result["ok"]:
            pred_file = os.path.join("outputs", "predictions", f"{symbol.lower()}_predictions.json")
            if os.path.exists(pred_file):
                with open(pred_file) as f:
                    self.send_json_response(200, {"status": "success", "data": json.load(f)})
            else:
                self.send_json_response(500, {"status": "error", "message": "Prediction file not generated."})
        else:
            self.send_json_response(500, {"status": "error", "message": result["stderr"]})

    # ── Stock Update + Predict ───────────────────────────────────────────
    def handle_update(self, qp):
        symbol    = qp.get("symbol", ["LICN"])[0].upper()
        full_sync = qp.get("full", ["false"])[0].lower() == "true"
        pages     = "all" if full_sync else "1"
        print(f"\n[API] Scraping {symbol} (pages={pages}) then retraining...")

        self._run_script(["update_data.py", "--symbol", symbol, "--pages", pages], timeout=300)
        result = self._run_script(["train_model.py", "--symbol", symbol], timeout=180)

        if result["ok"]:
            pred_file = os.path.join("outputs", "predictions", f"{symbol.lower()}_predictions.json")
            if os.path.exists(pred_file):
                with open(pred_file) as f:
                    self.send_json_response(200, {"status": "success", "data": json.load(f)})
            else:
                self.send_json_response(500, {"status": "error", "message": "Prediction file not generated."})
        else:
            self.send_json_response(500, {"status": "error", "message": result["stderr"]})

    # ── NEPSE Index: Scrape + Predict ────────────────────────────────────
    def handle_nepse_update(self, qp):
        pages = qp.get("pages", ["3"])[0]
        print(f"\n[API] Scraping NEPSE Index (pages={pages})...")
        self._run_script(["scrape_nepse.py", "--pages", pages], timeout=360)

        print("[API] Training NEPSE index model...")
        result = self._run_script(["predict_nepse.py"], timeout=180)

        if result["ok"] and os.path.exists(os.path.join("outputs", "predictions", "nepse_predictions.json")):
            with open(os.path.join("outputs", "predictions", "nepse_predictions.json")) as f:
                self.send_json_response(200, {"status": "success", "data": json.load(f)})
        else:
            # Try to just run predict on whatever data exists
            result2 = self._run_script(["predict_nepse.py"], timeout=180)
            if result2["ok"] and os.path.exists(os.path.join("outputs", "predictions", "nepse_predictions.json")):
                with open(os.path.join("outputs", "predictions", "nepse_predictions.json")) as f:
                    self.send_json_response(200, {"status": "success", "data": json.load(f)})
            else:
                self.send_json_response(500, {
                    "status": "error",
                    "message": result["stderr"] or "NEPSE data not available. Please scrape data first."
                })

    # ── NEPSE Index: Predict only ────────────────────────────────────────
    def handle_nepse_predict(self, _qp):
        print("\n[API] Running NEPSE Index prediction (no scrape)...")
        result = self._run_script(["predict_nepse.py"], timeout=180)
        if result["ok"] and os.path.exists(os.path.join("outputs", "predictions", "nepse_predictions.json")):
            with open(os.path.join("outputs", "predictions", "nepse_predictions.json")) as f:
                self.send_json_response(200, {"status": "success", "data": json.load(f)})
        else:
            self.send_json_response(500, {"status": "error", "message": result["stderr"] or "No NEPSE data found."})

    # ── History: Get all NEPSE history ──────────────────────────────────
    def handle_history(self, qp):
        history = _load_history()
        limit   = int(qp.get("limit", [str(len(history))])[0])
        self.send_json_response(200, {"history": history[:limit]})

    # ── History: Get stock prediction history ───────────────────────────
    def handle_stock_history(self, qp):
        symbol = qp.get("symbol", ["LICN"])[0].upper()
        history_file = os.path.join("outputs", "history", f"{symbol.lower()}_history.json")
        if os.path.exists(history_file):
            with open(history_file) as f:
                history = json.load(f)
        else:
            history = []
        limit = int(qp.get("limit", [str(max(len(history), 1))])[0])
        self.send_json_response(200, {"symbol": symbol, "history": history[:limit], "total": len(history)})

    # ── History: Stats ───────────────────────────────────────────────────
    def handle_history_stats(self, _qp):
        from predict_nepse import get_accuracy_stats
        history = _load_history()
        stats   = get_accuracy_stats(history)
        self.send_json_response(200, stats)


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
                
            symbol = entry.get("symbol")
            if not symbol:
                # Legacy entries: derive from type or id
                if entry.get("type") == "nepse_index":
                    symbol = "NEPSE"
                else:
                    symbol = entry.get("id", "").split("_")[0].upper()
            if not symbol:
                continue
            pred_date = entry["prediction_date"]
            
            # Load CSV to check actual close
            if symbol == "NEPSE":
                csv_file = os.path.join("data", "nepse_index_data.csv")
            else:
                csv_file = os.path.join("data", f"{symbol.lower()}_cleaned.csv")
                if not os.path.exists(csv_file):
                    csv_file = os.path.join("data", f"{symbol.lower()}_data.csv")
                
            if not os.path.exists(csv_file):
                continue
                
            df = pd.read_csv(csv_file)
            # Normalize date format: CSV may use YYYY/MM/DD, predictions use YYYY-MM-DD
            df["Date"] = df["Date"].str.replace("/", "-")
            df = df.sort_values(by="Date", ascending=True)
            # Find the row on or immediately after the prediction_date
            future_df = df[df["Date"] >= pred_date]
            if len(future_df) == 0:
                continue
            
            actual_row = future_df.iloc[0]
            actual_close = float(actual_row.get("Close", 0))
            actual_high = float(actual_row.get("High", actual_close))
            actual_low = float(actual_row.get("Low", actual_close))
            actual_date = str(actual_row["Date"]).replace("/", "-")
            
            pred_close = float(entry.get("predicted_close", 0))
            pred_high = float(entry.get("predicted_high", 0))
            pred_low = float(entry.get("predicted_low", 0))
            prev_close = float(entry["prev_close"])
            diff_pts = actual_close - prev_close
            
            actual_dir = "UP" if actual_close > prev_close else "DOWN"
            pred_dir = entry["predicted_direction"]
            
            if entry.get("timeframe", "1D") == "1D" and actual_date != pred_date:
                pass
                
            is_correct = False
            dead_band = abs(actual_close - prev_close) / (prev_close + 1e-8)
            if pred_dir == 'FLAT':
                is_correct = dead_band <= 0.005
            else:
                is_correct = (actual_dir == pred_dir)
            
            diff_percent = ((actual_close - pred_close) / (pred_close + 1e-8)) * 100 if pred_close else 0.0
            
            entry["actual_close"] = actual_close
            entry["actual_high"] = actual_high
            entry["actual_low"] = actual_low
            entry["diff_points"] = diff_pts
            entry["diff_percent"] = diff_percent
            entry["direction_correct"] = is_correct
            entry["resolved_on"] = actual_date
            
            if pred_high > 0 and pred_low > 0:
                high_hit = actual_high >= pred_high
                low_hit = actual_low <= pred_low
                range_contained_close = (actual_close <= pred_high and actual_close >= pred_low)
                entry["high_hit"] = bool(high_hit)
                entry["low_hit"] = bool(low_hit)
                entry["range_contained_close"] = bool(range_contained_close)
                
                if high_hit or low_hit:
                    entry["status"] = "exact"
                    entry["label"] = "✅ Exact"
                elif range_contained_close:
                    entry["status"] = "close"
                    entry["label"] = "🟡 Close"
                else:
                    entry["status"] = "wrong"
                    entry["label"] = "❌ Wrong"
            else:
                if abs(diff_percent) < 0.5:
                    entry["status"] = "exact"
                    entry["label"] = "✅ Exact"
                elif abs(diff_percent) < 2.0:
                    entry["status"] = "close"
                    entry["label"] = "🟡 Close"
                else:
                    entry["status"] = "wrong"
                    entry["label"] = "❌ Wrong"
                
                if not is_correct:
                    entry["status"] = "wrong"
                    entry["label"] = "❌ Wrong"
                
            resolved_count += 1
            
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
            
        self.send_json_response(200, {"status": "success", "resolved": resolved_count})

    # ── History: Delete an entry ─────────────────────────────────────────
    def handle_history_delete(self, payload):
        entry_id = payload.get("id", "")
        history  = _load_history()
        before   = len(history)
        history  = [e for e in history if e.get("id") != entry_id]
        _save_history(history)
        self.send_json_response(200, {
            "success": True,
            "deleted": before - len(history),
            "remaining": len(history)
        })

    # ── Analysis: 7-Layer Engine ──────────────────────────────────────────
    def handle_analysis(self, qp):
        symbol = qp.get("symbol", ["LICN"])[0].upper()
        print(f"\n[API] Running 7-layer analysis for {symbol}...")
        result = self._run_script(["analysis.py", "--symbol", symbol], timeout=120)
        if result["ok"]:
            analysis_file = os.path.join("outputs", "analysis", f"{symbol.lower()}_analysis.json")
            if os.path.exists(analysis_file):
                with open(analysis_file) as f:
                    self.send_json_response(200, {"status": "success", "data": json.load(f)})
            else:
                self.send_json_response(500, {"status": "error", "message": "Analysis file not generated."})
        else:
            self.send_json_response(500, {"status": "error", "message": result["stderr"] or "Analysis failed."})

    # ── Chat: Live AI Assistant ──────────────────────────────────────────
    def handle_chat(self, payload):
        message = payload.get("message", "")
        symbol = payload.get("symbol", "").upper()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.send_json_response(200, {
                "reply": "⚠️ **API Key Missing**\n\nPlease add your `GEMINI_API_KEY` to the `.env` file in the StockPrediction folder to enable the Live AI Assistant."
            })
            return
            
        if not symbol:
            self.send_json_response(200, {"reply": "Please specify a stock symbol for me to analyze."})
            return
            
        try:
            import google.genai as genai
            from google.genai import types
            from nepsealpha_scraper import fetch_nepsealpha
            from merolagani_scraper import fetch_stock_merolagani, fetch_nepse_merolagani
            
            # 1. Fetch live intraday data
            print(f"[API] Fetching live data for {symbol}...")
            df = fetch_nepsealpha(symbol, frame=10, provided_fsk="55CwQUJF") # Last 10 days
            
            if df.empty:
                print(f"[API] NepseAlpha failed, falling back to Merolagani for {symbol}...")
                if symbol == "NEPSE":
                    df = fetch_nepse_merolagani()
                else:
                    df = fetch_stock_merolagani(symbol, pages_to_scrape=1)
            
            if df.empty:
                self.send_json_response(200, {"reply": f"Sorry, I couldn't fetch live data for {symbol}. The market might be closed or the symbol is invalid."})
                return
                
            # 2. Format context
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None
            
            change = latest['Close'] - prev['Close'] if prev is not None else 0
            change_pct = (change / prev['Close'] * 100) if prev is not None and prev['Close'] > 0 else 0
            direction = "UP 🟢" if change > 0 else "DOWN 🔴"
            
            context = f"You are a professional quantitative trading assistant. Analyze the following LIVE market data for {symbol}.\n\n"
            context += f"CURRENT STATUS:\n"
            context += f"- Last Close: {latest['Close']}\n"
            context += f"- Today's Open: {latest['Open']}\n"
            context += f"- Today's High: {latest['High']}\n"
            context += f"- Today's Low: {latest['Low']}\n"
            context += f"- Volume: {latest['Volume']}\n"
            context += f"- Daily Change: {change:.2f} ({change_pct:.2f}%) {direction}\n\n"
            context += f"RECENT HISTORY (Last 5 Days):\n"
            context += df.tail(5)[['Date', 'Close', 'Volume']].to_string(index=False)
            context += f"\n\nUser Question: {message}\n"
            context += "\nKeep your response concise, professional, and focus on immediate market structure, momentum, and actionable insight."
            
            # 3. Call Gemini
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=context,
            )
            
            self.send_json_response(200, {"reply": response.text})
            
        except Exception as e:
            print(f"[ERROR] Chat failed: {e}")
            self.send_json_response(500, {"reply": f"An error occurred while analyzing the market: {str(e)}"})

    # ── Internal helper ──────────────────────────────────────────────────
    def _run_script(self, args, timeout=180):
        try:
            r = subprocess.run(
                [sys.executable] + args,
                capture_output=True, text=True, timeout=timeout
            )
            print(r.stdout[-2000:] if r.stdout else "")
            if r.returncode != 0:
                print(f"[SCRIPT ERROR] {r.stderr[-1000:]}")
            return {"ok": r.returncode == 0, "stdout": r.stdout, "stderr": r.stderr}
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": "Script timed out."}
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": str(e)}


# ── Helpers ──────────────────────────────────────────────────────────────
HISTORY_FILE = os.path.join("outputs", "history", "prediction_history.json")

def _load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def _save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def time_modified_str(filepath):
    if not os.path.exists(filepath):
        return "Never"
    return datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M:%S")


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ThreadedTCPServer.allow_reuse_address = True

    with ThreadedTCPServer(("0.0.0.0", PORT), NEPSEPredictorHandler) as httpd:
        print(f"\n{'='*55}")
        print(f"  NEPSE AI Prediction Dashboard — Server Active")
        print(f"  Port    : {PORT}")
        print(f"  URL     : http://localhost:{PORT}/index.html")
        print(f"  Press Ctrl+C to stop")
        print(f"{'='*55}\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.server_close()
            sys.exit(0)


if __name__ == "__main__":
    main()
