import os
import sys

def main():
    args = " ".join(sys.argv[1:])
    # scrape_nepse is now just an alias that fetches NEPSE via update_data.py
    cmd = f"python3 update_data.py --symbol NEPSE {args}"
    os.system(cmd)

if __name__ == "__main__":
    main()
