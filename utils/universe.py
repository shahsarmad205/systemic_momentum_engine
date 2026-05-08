import pandas as pd
import requests
import os
from datetime import datetime, timedelta


def _set_config_attr(config, key: str, value) -> None:
    """Set config metadata on either a dict-like config or BacktestConfig object."""
    if isinstance(config, dict):
        config[key] = value
    else:
        setattr(config, key, value)


def _get_config_attr(config, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_backtest_attr(config, key: str, default=None):
    """Resolve backtest-scoped config while preserving legacy top-level keys."""
    if isinstance(config, dict):
        if key in config:
            return config.get(key, default)
        bt = config.get("backtest", {}) or {}
        return bt.get(key, default) if isinstance(bt, dict) else default
    if hasattr(config, key):
        return getattr(config, key)
    bt = getattr(config, "backtest", None)
    if isinstance(bt, dict):
        return bt.get(key, default)
    return getattr(bt, key, default) if bt is not None else default


def _is_missing_wrds_username(raw: str | None) -> bool:
    user = str(raw or "").strip().lower()
    return user in {"", "your_wrds_username", "wrds_username", "username"}


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
        
        raise RuntimeError(
            "Unable to resolve S&P 500 universe: Wikipedia fetch failed and no cached ticker list exists. "
            "For production research, set universe.mode='wrds' to use a point-in-time CRSP universe. "
            "For research-only fallback, pre-populate config/sp500_tickers.txt or set universe.mode='file'."
        )

def _load_wrds_universe(config, universe_cfg: dict) -> list[str]:
    """
    Build a point-in-time S&P 500 universe from CRSP via WRDS and return
    tickers for the existing ticker-addressed pipeline.

    Requires WRDS_USERNAME env var and the ``wrds`` Python package.
    Returns [] on any failure so the caller can fall back gracefully.
    """
    import os
    wrds_user = os.environ.get("WRDS_USERNAME")
    if _is_missing_wrds_username(wrds_user):
        print("WRDS_USERNAME is missing or still set to a placeholder.")
        return []

    try:
        from utils.wrds_universe import WRDSUniverse, build_backtest_universe, connect_wrds

        start_date = _get_backtest_attr(config, "start_date", None)
        end_date = _get_backtest_attr(config, "end_date", start_date)
        if not start_date:
            return []

        min_price = universe_cfg.get("min_price", 10.0)
        # accept both key spellings from YAML
        min_dollar_vol = universe_cfg.get(
            "min_dollar_vol", universe_cfg.get("min_dollar_volume", 1e8)
        )

        db = connect_wrds(wrds_user)
        universe = WRDSUniverse(db)

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) if end_date else start_ts

        # Build the day-0 investable universe for immediate liquidity sanity.
        # The ticker-addressed research pipeline should not silently re-admit
        # low-liquidity historical members after the configured investability
        # screen. The full PIT membership panel remains attached for audit and
        # downstream eligibility checks.
        start_permnos = build_backtest_universe(
            db,
            date=start_date,
            min_price=float(min_price),
            min_dollar_vol=float(min_dollar_vol),
        )
        panel = universe.get_sp500_panel(start_ts, end_ts)
        if panel.empty and not start_permnos:
            return []

        panel_permnos = sorted(panel["permno"].astype(int).unique().tolist()) if not panel.empty else []
        include_full_membership_panel = bool(universe_cfg.get("include_full_membership_panel", False))
        if include_full_membership_panel:
            all_permnos = sorted(set(int(p) for p in start_permnos) | set(panel_permnos))
            permno_policy = "full_membership_with_start_liquidity_context"
        else:
            all_permnos = sorted(set(int(p) for p in start_permnos))
            permno_policy = "start_date_liquid"
        if not all_permnos:
            return []
        permno_to_tick = universe.permno_to_ticker_map(all_permnos, end_ts)
        if not permno_to_tick and start_permnos:
            permno_to_tick = universe.permno_to_ticker_map(start_permnos, start_ts)
        selected_permnos = set(all_permnos)
        permno_to_tick = {
            int(p): str(t)
            for p, t in permno_to_tick.items()
            if int(p) in selected_permnos
        }

        membership_ranges: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
        if not panel.empty:
            for _, row in panel.iterrows():
                permno = int(row["permno"])
                ticker = permno_to_tick.get(permno)
                if not ticker:
                    continue
                rng = membership_ranges.setdefault(ticker, [])
                rng.append(
                    (
                        pd.Timestamp(row["effective_start"]).normalize(),
                        pd.Timestamp(row["effective_end"]).normalize(),
                    )
                )

        # Stable ticker universe keyed by the end-date identifier for each PERMNO.
        tickers = [permno_to_tick[p] for p in all_permnos if p in permno_to_tick]
        ticker_to_permno = {t: p for p, t in permno_to_tick.items() if p in selected_permnos}
        _set_config_attr(config, "wrds_ticker_to_permno", ticker_to_permno)
        _set_config_attr(config, "pit_membership_ranges", membership_ranges)
        _set_config_attr(config, "pit_universe_mode", "wrds")
        _set_config_attr(config, "pit_universe_window", {"start_date": str(start_ts.date()), "end_date": str(end_ts.date())})
        print(
            f"WRDS PIT universe {start_ts.date()}→{end_ts.date()}: "
            f"{len(all_permnos)} PERMNOs → {len(tickers)} tradeable symbols "
            f"(membership records={len(panel)}, start-date liquid={len(start_permnos)}, "
            f"permno_policy={permno_policy})"
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
        if not bool(universe_cfg.get("allow_fallback", False)):
            raise RuntimeError(
                "WRDS universe mode requested but no point-in-time WRDS universe could be loaded. "
                "Check WRDS_USERNAME/WRDS access, CRSP dsp500list availability, and data.cache_dir. "
                "Set universe.allow_fallback: true only for non-production research fallbacks."
            )
        # Explicit research-only fallback.
        print("WRDS universe load failed — falling back to Wikipedia S&P 500 list because universe.allow_fallback=true.")
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
