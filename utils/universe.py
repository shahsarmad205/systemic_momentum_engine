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

def _load_wrds_universe(config, universe_cfg: dict) -> list[str]:
    """
    Build a point-in-time S&P 500 universe from CRSP via WRDS and return
    tickers for the existing ticker-addressed pipeline.

    Requires WRDS_USERNAME env var and the ``wrds`` Python package.
    Returns [] on any failure so the caller can fall back gracefully.
    """
    import os
    wrds_user = os.environ.get("WRDS_USERNAME")
    if not wrds_user:
        return []

    try:
        from utils.wrds_universe import WRDSUniverse, build_backtest_universe, connect_wrds

        start_date = getattr(config, "start_date", None) or (
            config.get("start_date") if isinstance(config, dict) else None
        )
        if not start_date:
            return []

        min_price = universe_cfg.get("min_price", 10.0)
        # accept both key spellings from YAML
        min_dollar_vol = universe_cfg.get(
            "min_dollar_vol", universe_cfg.get("min_dollar_volume", 1e8)
        )

        db = connect_wrds(wrds_user)
        universe = WRDSUniverse(db)

        # Build investable universe at the backtest start date
        permnos = build_backtest_universe(
            db,
            date=start_date,
            min_price=float(min_price),
            min_dollar_vol=float(min_dollar_vol),
        )
        if not permnos:
            return []

        # Map PERMNOs → tickers (point-in-time)
        permno_to_tick = universe.permno_to_ticker_map(permnos, start_date)
        tickers = [permno_to_tick[p] for p in permnos if p in permno_to_tick]
        print(
            f"WRDS universe at {start_date}: {len(permnos)} PERMNOs → {len(tickers)} tickers "
            f"(price≥${min_price}, dvol≥${min_dollar_vol/1e6:.0f}M, no delistings)"
        )
        return tickers

    except Exception as exc:
        print(f"_load_wrds_universe error: {exc}")
        return []


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

    if mode == 'wrds':
        tickers = _load_wrds_universe(config, universe_cfg)
        if tickers:
            # Apply exclude + max_tickers and return immediately (already deduped)
            exclude = universe_cfg.get('exclude', [])
            max_t = universe_cfg.get('max_tickers', 500)
            return [t for t in tickers if t not in exclude][:max_t]
        # Fall through to sp500 mode on WRDS failure
        print("WRDS universe load failed — falling back to Wikipedia S&P 500 list.")
        mode = 'sp500'

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
