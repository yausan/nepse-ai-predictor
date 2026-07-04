import os
import subprocess
import time
from analysis import SECTOR_MAP

def run_command(cmd, desc):
    print(f"\n[+] {desc}")
    print(f"    Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [!] Error in {desc}:\n{result.stderr[-500:]}")
        return False
    return True

def main():
    stocks = list(SECTOR_MAP.keys())
    total = len(stocks)
    print(f"Starting bulk training for {total} stocks...")
    
    # First, always update and train NEPSE index
    run_command(["python3", "predict_nepse.py"], "Updating NEPSE Index")
    
    success_count = 0
    fail_count = 0
    failed_stocks = []
    
    for i, symbol in enumerate(stocks, 1):
        print(f"\n{'='*50}")
        print(f" Processing {symbol} ({i}/{total})")
        print(f"{'='*50}")
        
        # 1. Train Model (handles scraping if needed)
        ok_train = run_command(["python3", "train_model.py", "--symbol", symbol], f"Training {symbol}")
        
        # 2. Run Analysis
        ok_analysis = run_command(["python3", "analysis.py", "--symbol", symbol], f"Analysis for {symbol}")
        
        if ok_train and ok_analysis:
            print(f"[✓] {symbol} completed successfully.")
            success_count += 1
        else:
            print(f"[✗] {symbol} failed.")
            fail_count += 1
            failed_stocks.append(symbol)
            
        time.sleep(2)
        
    print("\n" + "="*50)
    print("BULK TRAINING COMPLETE")
    print(f"Successful: {success_count}/{total}")
    print(f"Failed: {fail_count}/{total}")
    if failed_stocks:
        print(f"Failed stocks: {', '.join(failed_stocks)}")
    print("="*50)

if __name__ == "__main__":
    main()
