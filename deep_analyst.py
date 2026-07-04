import os
import json
import google.generativeai as genai
from scrapers import scrape_news, scrape_fundamentals
from dotenv import load_dotenv

def generate_executive_summary(symbol, ml_data_path):
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("No GEMINI_API_KEY found in .env, skipping deep analysis.")
        return None
        
    genai.configure(api_key=api_key)
    
    # Load ml data
    ml_data = {}
    if os.path.exists(ml_data_path):
        with open(ml_data_path, "r") as f:
            ml_data = json.load(f)
            
    news = scrape_news(symbol)
    fundamentals = scrape_fundamentals(symbol)
    
    prompt = f"""
    You are an expert NEPSE stock analyst. I need a comprehensive Deep Contextual Stock Prediction for {symbol}.
    
    ## Provided Data:
    Fundamentals: {json.dumps(fundamentals)}
    Recent News: {json.dumps(news)}
    Technical & ML Analysis (7-layer output): {json.dumps(ml_data)}
    
    ## Requirements:
    Output valid JSON ONLY with the following exact keys (no markdown formatting outside the JSON):
    {{
        "executive_summary": "A concise paragraph summarizing the setup.",
        "current_market_sentiment": "Positive/Negative/Neutral with 1 sentence reasoning.",
        "news_impact_analysis": "Summary of news impact.",
        "technical_analysis": "Summary of technical signals.",
        "fundamental_analysis": "Summary of fundamental health.",
        "historical_pattern_match": "Simulated pattern match string.",
        "bullish_probability": <integer 0-100>,
        "bearish_probability": <integer 0-100>,
        "neutral_probability": <integer 0-100>,
        "risk_factors": ["risk 1", "risk 2"],
        "confidence_score": <integer 0-100>,
        "final_recommendation": "BUY/HOLD/SELL",
        "recommendation_justification": "Why this recommendation.",
        "why_it_could_be_wrong": "Factors that could invalidate the prediction."
    }}
    Ensure probabilities sum to 100. Return ONLY JSON.
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.endswith('```'):
            text = text[:-3]
            
        result = json.loads(text.strip())
        
        # Save output
        out_path = os.path.join("outputs", "analysis", f"{symbol.lower()}_deep_analysis.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
            
        return result
    except Exception as e:
        print(f"LLM deep analysis failed: {e}")
        return None

if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "NABIL"
    res = generate_executive_summary(sym, f"outputs/analysis/{sym.lower()}_analysis.json")
    print(json.dumps(res, indent=2))
