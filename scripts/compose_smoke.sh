#!/usr/bin/env bash
# Local compose smoke: build api, wait for /health, tear down.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Validating compose"
docker compose config >/dev/null
docker compose --profile pgvector config >/dev/null
docker compose --profile ollama config >/dev/null

echo "==> Starting api"
docker compose up -d --build api

cleanup() {
  docker compose down >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> Waiting for /health"
for i in $(seq 1 36); do
  if curl -fsS http://localhost:8000/health | grep -q '"status"'; then
    echo "OK: $(curl -fsS http://localhost:8000/health)"
    exit 0
  fi
  sleep 5
done

echo "FAIL: API never became healthy"
docker compose logs api | tail -n 80
exit 1
