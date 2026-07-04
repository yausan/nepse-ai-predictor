import os
import subprocess
import time

FAILED_STOCKS = ["GDBL", "GRDBL", "JBBL", "KSBBL", "MLBL", "MNBBL", "NABBC", "SADBL", "SAPDBL", "SHINE", "CBBL", "DDBL", "FMDBL", "FOWAD", "GBLBS", "JBLB", "KLBSL", "LLBS", "MLBSL", "MMLBS", "NSLB", "RSDC", "SLBS", "SMATA", "SWBBL", "VLBS"]

def run_command(cmd, desc):
    print(f"\n[+] {desc}")
    print(f"    Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [!] Error in {desc}:\n{result.stderr[-500:]}")
        return False
    return True

def main():
    total = len(FAILED_STOCKS)
    print(f"Retrying bulk training for {total} previously failed stocks...")
    
    success_count = 0
    fail_count = 0
    still_failed = []
    
    for i, symbol in enumerate(FAILED_STOCKS, 1):
        print(f"\n{'='*50}")
        print(f" Processing {symbol} ({i}/{total})")
        print(f"{'='*50}")
        
        ok_train = run_command(["python3", "train_model.py", "--symbol", symbol], f"Training {symbol}")
        ok_analysis = run_command(["python3", "analysis.py", "--symbol", symbol], f"Analysis for {symbol}")
        
        if ok_train and ok_analysis:
            print(f"[✓] {symbol} completed successfully.")
            success_count += 1
        else:
            print(f"[✗] {symbol} failed.")
            fail_count += 1
            still_failed.append(symbol)
            
        time.sleep(3)
        
    print("\n" + "="*50)
    print("RETRY COMPLETE")
    print(f"Successful: {success_count}/{total}")
    print(f"Failed: {fail_count}/{total}")
    if still_failed:
        print(f"Still failed stocks: {', '.join(still_failed)}")
    print("="*50)

if __name__ == "__main__":
    main()
