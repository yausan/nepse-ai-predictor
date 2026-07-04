import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import random

def scrape_news(symbol, max_items=5):
    """
    Scrapes recent news related to the given NEPSE symbol using Google News RSS.
    Returns a list of dictionaries with 'title', 'link', 'pubDate'.
    """
    query = f"{symbol} NEPSE Nepal"
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    
    news_items = []
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        
        for item in root.findall('./channel/item')[:max_items]:
            title = item.find('title').text if item.find('title') is not None else ""
            link = item.find('link').text if item.find('link') is not None else ""
            pubDate = item.find('pubDate').text if item.find('pubDate') is not None else ""
            
            news_items.append({
                "title": title,
                "link": link,
                "pubDate": pubDate
            })
            
    except Exception as e:
        print(f"Error scraping news for {symbol}: {e}")
        
    return news_items

def scrape_fundamentals(symbol):
    """
    Returns fundamental data for a given NEPSE symbol.
    In a fully productionized system, this would scrape ShareSansar or use an API like NepseAlpha.
    For this implementation, we simulate realistic fundamental data extraction.
    """
    # Deterministic randomness based on symbol length and characters to maintain consistency
    seed = sum(ord(c) for c in symbol)
    random.seed(seed)
    
    eps = round(random.uniform(5.0, 50.0), 2)
    pe = round(random.uniform(10.0, 40.0), 2)
    pbv = round(random.uniform(1.0, 5.0), 2)
    roe = round(random.uniform(5.0, 25.0), 2)
    dividend = round(random.uniform(0.0, 20.0), 2)
    
    sector_pe = round(pe * random.uniform(0.8, 1.2), 2)
    
    return {
        "EPS": eps,
        "PE_Ratio": pe,
        "Sector_PE": sector_pe,
        "PBV": pbv,
        "ROE_Percent": roe,
        "Dividend_Yield_Percent": dividend,
        "Market_Cap_Cr": round(random.uniform(1000, 50000), 2),
        "Debt_to_Equity": round(random.uniform(0.1, 3.0), 2)
    }

if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "NABIL"
    print(f"--- Fundamentals for {sym} ---")
    print(scrape_fundamentals(sym))
    print(f"\n--- Recent News for {sym} ---")
    news = scrape_news(sym)
    for n in news:
        print(f"- {n['title']} ({n['pubDate']})")
