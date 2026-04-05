import pandas as pd
import requests
import os
from datetime import datetime, timedelta

def get_sp500_tickers(cache_path="config/sp500_tickers.txt", max_age_days=7):
    """
    Fetch S&P 500 tickers with local cache fallback.
    Refreshes cache if older than max_age_days.
    Pillar 29: Institutional Dynamic Universe.
    """
    # 1. Check if fresh cache exists
    if os.path.exists(cache_path):
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
        if age < timedelta(days=max_age_days):
            try:
                with open(cache_path, "r") as f:
                    tickers = f.read().splitlines()
                if tickers:
                    print(f"Loaded {len(tickers)} tickers from cache (Age: {age.days} days)")
                    return tickers
            except Exception as e:
                print(f"Cache read error: {e}")

    # 2. Fetch fresh from Wikipedia
    print("Fetching fresh S&P 500 constituents from Wikipedia...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        # Pillar 29: mimick browser to avoid 403 Forbidden
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        tables = pd.read_html(response.text)
        sp500 = tables[0]
        tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()
        
        # Pillar 29 Condition 2: Survivorship Bias Warning
        print("\n" + "="*60)
        print("WARNING: Wikipedia S&P 500 list reflects CURRENT constituents.")
        print("Backtest results may be slightly optimistic due to survivorship bias.")
        print("For production research, use a point-in-time universe (CRSP/Compustat).")
        print("="*60 + "\n")

        # Pillar 29: Initial Sanity Check
        exclude = ["SPY", "XYZ", "BF-B", "BRK-B"]
        tickers = [t for t in tickers if t not in exclude]
        
        # 3. Save Cache for reproducibility (Pillar 29 Condition 3)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            f.write("\n".join(tickers))
        print(f"Fetched and cached {len(tickers)} tickers to {cache_path}")
        return tickers

    except Exception as e:
        print(f"Wikipedia fetch failed: {e}")
        # Fall back to cache even if stale (Pillar 29 Condition 1)
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                tickers = f.read().splitlines()
            if tickers:
                print(f"Using stale cache fallback: {len(tickers)} tickers")
                return tickers
        
        print("Pillar 29: Falling back to hardcoded institutional tickers...")
        return [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA",
            "JPM", "V", "JNJ", "WMT", "PG", "XOM", "UNH", "HD", "AVGO"
        ]

def load_universe(config):
    """
    Dynamically resolve symbols from config for run_backtest.py or run_model_selection.py.
    Pillar 29: Dynamic Resolver.
    """
    if not config:
        return []

    # Pillar 29: Robust access (support both DotDict/Namespace and standard dict)
    if isinstance(config, dict):
        universe_cfg = config.get('universe', {})
        tickers_fallback = config.get('tickers', [])
    else:
        universe_cfg = getattr(config, 'universe', {})
        tickers_fallback = getattr(config, 'tickers', [])

    if not isinstance(universe_cfg, dict):
        universe_cfg = vars(universe_cfg) if hasattr(universe_cfg, '__dict__') else {}

    mode = universe_cfg.get('mode', 'custom')
    cache_path = universe_cfg.get('cache_path', 'config/sp500_tickers.txt')
    cache_age = universe_cfg.get('cache_max_age_days', 7)
    
    if mode == 'sp500':
        tickers = get_sp500_tickers(cache_path=cache_path, max_age_days=cache_age)
    elif mode == 'file':
        path = universe_cfg.get('file_path', 'config/tickers.txt')
        if os.path.exists(path):
            with open(path, "r") as f:
                tickers = f.read().splitlines()
        else:
            print(f"Universe file {path} not found, falling back to tickers list")
            tickers = tickers_fallback
    else:
        tickers = tickers_fallback
    
    # 4. Filter and Deduplicate
    exclude = universe_cfg.get('exclude', [])
    max_t = universe_cfg.get('max_tickers', 500)
    
    # Deduplicate preserving order
    unique_tickers = list(dict.fromkeys(tickers))
    filtered_tickers = [t for t in unique_tickers if t not in exclude][:max_t]
    
    if not filtered_tickers:
        print("WARNING: load_universe resolved zero tickers. Check config/universe block.")
        
    return filtered_tickers
