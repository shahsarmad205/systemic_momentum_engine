# Live monitoring dashboard (React)

Small **Create React App** UI that polls the Flask API under `trend_signal_engine/api/server.py`.

## Prerequisites

- Node.js 18+ and npm
- Python 3 with `flask`, `flask-cors`, `pandas` (see `../requirements.txt`)

## Configuration

```bash
cp .env.example .env
# Edit .env — set REACT_APP_API_URL to your API (default http://localhost:5001)
```

Use a **full URL** with scheme (`http://` or `https://`), no quotes, no spaces. Example:

`REACT_APP_API_URL=http://localhost:5001`

If you change `.env`, **restart** `npm start` (CRA reads env at startup). If the API is down, the UI now shows a clear error instead of a bare `SyntaxError` from parsing non‑JSON HTML.

## Install & run (React)

From this `dashboard/` directory:

```bash
npm install
npm start
```

The app opens at [http://localhost:3000](http://localhost:3000) and refreshes data every **30 seconds**.

## Run the Flask API

From the **`trend_signal_engine/`** directory (parent of `api/` and `dashboard/`):

```bash
pip install flask flask-cors pandas
python api/server.py
```

The API listens on **port 5001** by default. Override with:

```bash
PORT=5002 python api/server.py
```

## Production build

```bash
npm run build
```

Serve the `build/` folder with any static host; ensure `REACT_APP_API_URL` points at the deployed API when you run `npm run build`.
