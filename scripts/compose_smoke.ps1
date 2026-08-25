# Local compose smoke (Windows): build api, wait for /health, tear down.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "==> Validating compose"
docker compose config | Out-Null
docker compose --profile pgvector config | Out-Null
docker compose --profile ollama config | Out-Null

Write-Host "==> Starting api"
docker compose up -d --build api

try {
    Write-Host "==> Waiting for /health"
    for ($i = 1; $i -le 36; $i++) {
        try {
            $body = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 3
            if ($body.status -eq "ok") {
                Write-Host "OK:" ($body | ConvertTo-Json -Compress)
                exit 0
            }
        } catch {
            Start-Sleep -Seconds 5
        }
    }
    Write-Host "FAIL: API never became healthy"
    docker compose logs api
    exit 1
}
finally {
    docker compose down | Out-Null
}
