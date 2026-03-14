# Frontend / static assets (legacy)

The **Flask backend** (`app.py`, `api_helpers.py`) has been removed. The platform now uses a **FastAPI** API.

## API (new)

Run the API from **project root**:

```bash
pip install fastapi uvicorn python-dotenv
python api/main.py
# Or: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Base URL:** http://localhost:8000  
- **Docs:** http://localhost:8000/docs  
- **Health:** GET http://localhost:8000/health  
- **Signals:** GET http://localhost:8000/api/signals?date=YYYY-MM-DD  
- **Market:** GET http://localhost:8000/api/market/history/{ticker}  
- **Portfolio:** GET http://localhost:8000/api/portfolio  

CORS is enabled for `http://localhost:3000` and `https://*.vercel.app` so a separate frontend (e.g. React on port 3000 or deployed on Vercel) can call the API.

## This folder

- `static/` — legacy HTML/CSS/JS for the old Flask UI; can be reused or replaced by a new app that talks to the FastAPI API.
- `mock_data/` — sample tickers and signals for testing.

To build a new UI, point it at `http://localhost:8000` (or your deployed API) and use the routes above.
