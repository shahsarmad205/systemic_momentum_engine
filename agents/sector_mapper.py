import yfinance as yf
import json, os, time
from collections import Counter

SECTOR_CACHE_PATH = "output/cache/ticker_info"
CACHE_TTL_DAYS = 30

SECTOR_NORMALIZE = {
    'Technology': 'Tech',
    'Communication Services': 'Tech',
    'Consumer Cyclical': 'Consumer',
    'Consumer Defensive': 'Consumer',
    'Financial Services': 'Finance',
    'Healthcare': 'Healthcare',
    'Energy': 'Energy',
    'Industrials': 'Industrial',
    'Basic Materials': 'Materials',
    'Real Estate': 'RealEstate',
    'Utilities': 'Utilities',
}

ETF_OVERRIDES = {
    'SPY':'ETF','QQQ':'ETF','IWM':'ETF','DIA':'ETF',
    'ARKK':'ETF','VTI':'ETF','XLK':'ETF',
}

def get_ticker_info(ticker):
    os.makedirs(SECTOR_CACHE_PATH, exist_ok=True)
    cache_file = f"{SECTOR_CACHE_PATH}/{ticker}.json"
    if os.path.exists(cache_file):
        age = (time.time() - os.path.getmtime(cache_file))/86400
        if age < CACHE_TTL_DAYS:
            with open(cache_file) as f:
                return json.load(f)
    try:
        info = yf.Ticker(ticker).info
        with open(cache_file, 'w') as f:
            json.dump(info, f)
        return info
    except Exception as e:
        print(f"  [SectorMapper] WARNING: {ticker}: {e}")
        return {}

def get_sector(ticker):
    if ticker in ETF_OVERRIDES:
        return ETF_OVERRIDES[ticker]
    info = get_ticker_info(ticker)
    raw = info.get('sector', 'Unknown')
    return SECTOR_NORMALIZE.get(raw, raw)

def build_sector_map(tickers):
    return {t: get_sector(t) for t in tickers}

def validate_sector_balance(sector_map, max_pct=0.50):
    counts = Counter(sector_map.values())
    total = len(sector_map)
    print("[SectorMapper] Sector distribution:")
    for sector, count in counts.most_common():
        pct = count/total*100
        flag = " ⚠️" if count/total > max_pct else ""
        print(f"  {sector:15s}: {count:3d} ({pct:.0f}%){flag}")
