#!/usr/bin/env bash
# Start the API in the background, then the Streamlit UI on the Spaces port (7860).
set -euo pipefail

uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &

exec streamlit run ui/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true
