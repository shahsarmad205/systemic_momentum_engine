FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/trend_signal_engine

# Install dependencies first for better caching.
COPY requirements-lock.txt requirements-lock.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements-lock.txt

# Copy the application code.
COPY . .

# Default to running the batch pipeline (safe to override in scheduler).
CMD ["python", "run_daily_pipeline.py", "--dry-run"]

