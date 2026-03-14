"""
FastAPI application for the quant trading platform.

- CORS: localhost:3000 and https://*.vercel.app
- Env: .env via python-dotenv
- Routes: api/routes/signals, market, portfolio
- GET /health: { status, date }
- Startup: warn if today's signals cache is missing
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path so "api" package is found (run from any cwd)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import market, portfolio, signals

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warn if today's signals cache is missing. No shutdown logic."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache_dir = os.environ.get("SIGNALS_CACHE_DIR", "output/cache")
    path = os.path.join(cache_dir, f"signals_{today}.json")
    if not os.path.isfile(path):
        logger.warning("Today's cache file not found: %s (run pipeline.daily_runner to generate)", path)
    yield


app = FastAPI(
    title="Trend Signal Engine API",
    description="Quant trading platform: signals, market data, portfolio",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: development origins + optional production FRONTEND_URL; Vercel via regex
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
FRONTEND_URL = os.getenv("FRONTEND_URL")
if FRONTEND_URL:
    ALLOWED_ORIGINS.append(FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
app.include_router(market.router, prefix="/api/market", tags=["market"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])


@app.get("/ping")
def ping():
    """Simple liveness check; useful for Railway health checks."""
    return {"pong": True}


@app.get("/health")
def health():
    """Health check; returns status and today's date."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return {"status": "ok", "date": today}


# Root-level aliases so /signal/AAPL and /market/overview work (same as /api/signals/signal/..., /api/market/...)
@app.get("/signal/{ticker}", include_in_schema=False)
def root_signal(ticker: str):
    """Same response as GET /api/signals/signal/{ticker}."""
    return signals.get_signal(ticker)


@app.get("/market/overview", include_in_schema=False)
def root_market_overview():
    """Same response as GET /api/market/overview."""
    return market.get_market_overview()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
