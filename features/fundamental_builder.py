"""
Fundamental Feature Builder — SEC EDGAR XBRL
=============================================
Fetches Piotroski F-score components and accruals from the SEC EDGAR XBRL API.

Why EDGAR over yfinance:
  - yfinance quarterly financials only provides the last 4-5 quarters (2024+)
  - EDGAR provides full history back to ~2009 for most S&P 500 companies
  - EDGAR uses the actual `filed` date for each report → true point-in-time,
    eliminating lookahead bias without a fixed filing-lag assumption

Point-in-time safety:
  Each value becomes available on the `filed` date from EDGAR (the actual SEC
  submission date), NOT the period end date. No fixed lag needed.

Features produced:
  f_score          (0–5)  Piotroski score; LOW = weak = short candidate
  accruals_ratio   (float) (Net Income - Op CF) / Assets; HIGH = inflated earnings
  roa              (float) Net Income / Assets; NEGATIVE = unprofitable
  delta_roa        (float) QoQ ROA change; NEGATIVE = deteriorating
  delta_leverage   (float) QoQ LT Debt / Assets change; POSITIVE = rising debt

EDGAR XBRL concepts used (us-gaap namespace):
  NetIncomeLoss
  Assets
  NetCashProvidedByUsedInOperatingActivities
  LongTermDebt (or LongTermDebtNoncurrent)
  AssetsCurrent
  LiabilitiesCurrent

Cache:
  Ticker → CIK:   data/cache/fundamentals/cik_map.json  (refreshed weekly)
  Company facts:  data/cache/fundamentals/{CIK}.json     (refreshed monthly)
  Daily features: data/cache/fundamentals/{ticker}.parquet (rebuilt from above)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/cache/fundamentals")
CIK_CACHE_TTL_DAYS = 7
FACTS_CACHE_TTL_DAYS = 30
FEATURE_CACHE_TTL_DAYS = 30

_ZERO_COLS = ["f_score", "accruals_ratio", "roa", "delta_roa", "delta_leverage"]
_EDGAR_UA = "trend-signal-engine/1.0 research@example.com"   # SEC requires a User-Agent

# XBRL concept fallback chains (us-gaap namespace)
_CONCEPTS = {
    "net_income":    ["NetIncomeLoss", "NetIncome", "ProfitLoss"],
    "total_assets":  ["Assets"],
    "operating_cf":  ["NetCashProvidedByUsedInOperatingActivities",
                      "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "lt_debt":       ["LongTermDebtNoncurrent", "LongTermDebt",
                      "LongTermDebtAndCapitalLeaseObligations"],
    "current_assets": ["AssetsCurrent"],
    "current_liab":  ["LiabilitiesCurrent"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zeros(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=dates, columns=_ZERO_COLS)


def _fetch_json(url: str, retries: int = 3) -> dict:
    import urllib.request
    headers = {"User-Agent": _EDGAR_UA, "Accept": "application/json"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.0 * (attempt + 1))
    return {}


def _cache_fresh(path: Path, ttl_days: int) -> bool:
    if not path.exists():
        return False
    age = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).days
    return age < ttl_days


# ---------------------------------------------------------------------------
# Ticker → CIK mapping
# ---------------------------------------------------------------------------

def _get_cik(ticker: str) -> str | None:
    """Return zero-padded 10-digit CIK for a ticker, or None if not found."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    map_path = CACHE_DIR / "cik_map.json"

    cik_map: dict[str, str] = {}
    if _cache_fresh(map_path, CIK_CACHE_TTL_DAYS):
        try:
            cik_map = json.loads(map_path.read_text())
        except Exception:
            pass

    if ticker.upper() in cik_map:
        return cik_map[ticker.upper()]

    # Try the direct company_tickers endpoint
    try:
        tickers_data = _fetch_json("https://www.sec.gov/files/company_tickers_exchange.json")
        fields = tickers_data.get("fields", [])
        rows = tickers_data.get("data", [])
        if "ticker" in fields and "cik" in fields:
            ticker_idx = fields.index("ticker")
            cik_idx = fields.index("cik")
            for row in rows:
                t = str(row[ticker_idx]).upper()
                c = str(row[cik_idx]).zfill(10)
                cik_map[t] = c
    except Exception:
        pass

    # Fallback: try EDGAR full-text search
    if ticker.upper() not in cik_map:
        try:
            search = _fetch_json(
                f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K"
            )
            hits = search.get("hits", {}).get("hits", [])
            for hit in hits:
                src = hit.get("_source", {})
                if src.get("period_of_report") and src.get("entity_id"):
                    cik = str(src.get("entity_id", "")).zfill(10)
                    # Check display name contains ticker
                    if ticker.upper() in src.get("display_names", "").upper():
                        cik_map[ticker.upper()] = cik
                        break
        except Exception:
            pass

    # Persist updated map
    try:
        map_path.write_text(json.dumps(cik_map))
    except Exception:
        pass

    return cik_map.get(ticker.upper())


# ---------------------------------------------------------------------------
# EDGAR company facts
# ---------------------------------------------------------------------------

def _get_company_facts(cik: str) -> dict:
    """Fetch and cache all XBRL facts for a CIK."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    facts_path = CACHE_DIR / f"facts_{cik}.json"

    if _cache_fresh(facts_path, FACTS_CACHE_TTL_DAYS):
        try:
            return json.loads(facts_path.read_text())
        except Exception:
            pass

    padded = cik.zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{padded}.json"
    data = _fetch_json(url)

    try:
        facts_path.write_text(json.dumps(data))
    except Exception:
        pass

    return data


def _extract_quarterly_series(facts: dict, concept_names: list[str]) -> pd.DataFrame:
    """
    Extract standalone quarterly FLOW values (≈90d period) for income/cash-flow concepts.
    Returns DataFrame with columns: period_end, filed, value.

    Only keeps entries where the reported period is 50–130 days (true quarterly).
    Excludes: YTD H1 (≈180d), YTD 9-month (≈270d), annual (≈365d).
    Entries without a `start` field (balance-sheet snapshots) are skipped here.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for concept in concept_names:
        if concept not in us_gaap:
            continue
        units = us_gaap[concept].get("units", {})
        usd_entries = units.get("USD", [])
        if not usd_entries:
            continue

        rows = []
        for entry in usd_entries:
            form = entry.get("form", "")
            if form not in ("10-Q", "10-K"):
                continue
            start = entry.get("start")
            end = entry.get("end")
            filed = entry.get("filed")
            val = entry.get("val")
            if not start or not end or not filed or val is None:
                continue
            try:
                period_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
            except Exception:
                continue
            if not (50 < period_days < 130):
                continue
            rows.append({
                "period_end": pd.Timestamp(end),
                "filed": pd.Timestamp(filed),
                "value": float(val),
            })

        if not rows:
            continue

        df = pd.DataFrame(rows)
        # Deduplicate: keep the earliest filing for each period_end
        df = (df
              .sort_values("filed")
              .drop_duplicates(subset="period_end", keep="first")
              .sort_values("period_end"))
        return df

    return pd.DataFrame(columns=["period_end", "filed", "value"])


def _extract_balance_sheet_series(facts: dict, concept_names: list[str]) -> pd.DataFrame:
    """
    Extract balance-sheet snapshot values for a list of concept fallbacks.
    Balance-sheet items have only an `end` date (no `start`) in EDGAR.
    Returns DataFrame with columns: period_end, filed, value.

    Keeps the earliest filing for each period_end across 10-Q and 10-K forms.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for concept in concept_names:
        if concept not in us_gaap:
            continue
        units = us_gaap[concept].get("units", {})
        usd_entries = units.get("USD", [])
        if not usd_entries:
            continue

        rows = []
        for entry in usd_entries:
            form = entry.get("form", "")
            if form not in ("10-Q", "10-K"):
                continue
            end = entry.get("end")
            filed = entry.get("filed")
            val = entry.get("val")
            if not end or not filed or val is None:
                continue
            rows.append({
                "period_end": pd.Timestamp(end),
                "filed": pd.Timestamp(filed),
                "value": float(val),
            })

        if not rows:
            continue

        df = pd.DataFrame(rows)
        # Deduplicate: keep the earliest filing for each period_end
        df = (df
              .sort_values("filed")
              .drop_duplicates(subset="period_end", keep="first")
              .sort_values("period_end"))
        return df

    return pd.DataFrame(columns=["period_end", "filed", "value"])


def _series_to_daily(
    pit_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    value_col: str = "value",
) -> pd.Series:
    """
    Map point-in-time quarterly values onto a daily date index.
    Value is available from `filed` date onward (true point-in-time).
    Forward-filled until the next filing becomes available.
    """
    if pit_df.empty:
        return pd.Series(np.nan, index=dates)

    # Set index to filed date, value = financial figure
    s = pit_df.set_index("filed")[value_col].sort_index()

    # Reindex onto a dense daily range and forward-fill
    full_range = pd.date_range(
        min(s.index.min(), dates.min()),
        max(s.index.max(), dates.max()),
        freq="D",
    )
    daily = s.reindex(full_range).ffill()
    return daily.reindex(dates)


# ---------------------------------------------------------------------------
# Piotroski computation
# ---------------------------------------------------------------------------

def _compute_piotroski(quarterly_data: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Compute 5-factor Piotroski score from quarterly filing-dated Series.
    Input dict keys: net_income, total_assets, operating_cf, lt_debt,
                     current_assets, current_liab
    Each Series is indexed by filed date, value is the quarterly figure.
    Returns DataFrame indexed by filed date with f_score and components.
    """
    # Align all series to their filing dates (deduplicate each series first)
    all_dates = sorted(set().union(*[s.index for s in quarterly_data.values() if not s.empty]))
    if not all_dates:
        return pd.DataFrame()

    df = pd.DataFrame(index=all_dates)
    for key, s in quarterly_data.items():
        if not s.empty:
            # Deduplicate index before reindex to avoid ValueError
            s_dedup = s[~s.index.duplicated(keep="first")]
            df[key] = s_dedup.reindex(all_dates).ffill()
        else:
            df[key] = np.nan

    df = df.sort_index().ffill()

    ta = df["total_assets"].replace(0, np.nan)

    roa = (df["net_income"] / ta).clip(-5.0, 5.0)
    delta_roa = roa.diff()

    operating_cf = df.get("operating_cf", pd.Series(np.nan, index=df.index))
    accruals = ((df["net_income"] - operating_cf) / ta).clip(-2.0, 2.0)

    lt_debt = df.get("lt_debt", pd.Series(0.0, index=df.index)).fillna(0.0)
    leverage = (lt_debt / ta).clip(0.0, 10.0)
    delta_leverage = leverage.diff()

    curr_assets = df.get("current_assets", pd.Series(np.nan, index=df.index))
    curr_liab = df.get("current_liab", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    current_ratio = (curr_assets / curr_liab).clip(0.0, 20.0)
    delta_liquidity = current_ratio.diff()

    f1 = (roa > 0).astype(float)
    f2 = (delta_roa > 0).astype(float)
    f3 = (accruals < 0).astype(float)
    f4 = (delta_leverage < 0).astype(float)
    f5 = (delta_liquidity > 0).astype(float)
    f_score = (f1 + f2 + f3 + f4 + f5).clip(0, 5)

    return pd.DataFrame({
        "f_score": f_score,
        "accruals_ratio": accruals,
        "roa": roa,
        "delta_roa": delta_roa,
        "delta_leverage": delta_leverage,
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_fundamental_features(
    ticker: str,
    dates: pd.DatetimeIndex,
    cache_dir: Path | None = None,
    cache_ttl_days: int = FEATURE_CACHE_TTL_DAYS,
) -> pd.DataFrame:
    """
    Return a DataFrame (indexed by `dates`) with fundamental features.
    Columns: f_score, accruals_ratio, roa, delta_roa, delta_leverage
    Returns 0.0 when data unavailable (pre-2009 or fetch failure).
    """
    global CACHE_DIR
    if cache_dir:
        CACHE_DIR = cache_dir
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker}.parquet"

    if _cache_fresh(cache_path, cache_ttl_days):
        try:
            cached = pd.read_parquet(cache_path)
            if all(c in cached.columns for c in _ZERO_COLS):
                return cached.reindex(dates, method="ffill").fillna(0.0)[_ZERO_COLS]
        except Exception:
            pass

    try:
        result = _fetch_and_compute(ticker, dates)
    except Exception:
        result = _zeros(dates)

    try:
        result.to_parquet(cache_path)
    except Exception:
        pass

    return result


def _fetch_and_compute(ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Fetch from EDGAR and compute features for all `dates`."""
    cik = _get_cik(ticker)
    if not cik:
        return _zeros(dates)

    facts = _get_company_facts(cik)
    if not facts:
        return _zeros(dates)

    # Balance-sheet items are point-in-time snapshots (no start date in EDGAR)
    _BALANCE_SHEET_FIELDS = {"total_assets", "lt_debt", "current_assets", "current_liab"}

    # Extract quarterly series for each concept group
    quarterly: dict[str, pd.Series] = {}
    for field, concepts in _CONCEPTS.items():
        if field in _BALANCE_SHEET_FIELDS:
            pit_df = _extract_balance_sheet_series(facts, concepts)
        else:
            pit_df = _extract_quarterly_series(facts, concepts)
        if not pit_df.empty:
            s = pit_df.set_index("filed")["value"].sort_index()
            # Deduplicate filed-date index (same date can appear in multiple filings)
            s = s[~s.index.duplicated(keep="first")]
            quarterly[field] = s
        else:
            quarterly[field] = pd.Series(dtype=float)

    if "net_income" not in quarterly or quarterly["net_income"].empty:
        return _zeros(dates)
    if "total_assets" not in quarterly or quarterly["total_assets"].empty:
        return _zeros(dates)

    # Compute Piotroski scores on filing-date-indexed series
    piotroski = _compute_piotroski(quarterly)
    if piotroski.empty:
        return _zeros(dates)

    # Map each column to daily dates using the filing date as availability date
    result = pd.DataFrame(index=dates)
    for col in _ZERO_COLS:
        if col in piotroski.columns:
            s = piotroski[col].sort_index()
            full_range = pd.date_range(
                min(s.index.min(), dates.min()),
                max(s.index.max(), dates.max()),
                freq="D",
            )
            daily = s.reindex(full_range).ffill()
            result[col] = daily.reindex(dates).values
        else:
            result[col] = 0.0

    return result.fillna(0.0)
