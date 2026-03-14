#!/usr/bin/env bash
# Run the FastAPI server from the project root so the 'api' package is found.
cd "$(dirname "$0")"
python -m api.main
