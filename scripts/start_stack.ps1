# Startup script for ChatREL v4 Stack (Redis + Web + Celery)

Write-Host "Starting ChatREL v4 Stack..." -ForegroundColor Cyan

# Check if Docker is running
docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker is not running. Please start Docker."
    exit 1
}

# Start Redis
Write-Host "Starting Redis..." -ForegroundColor Yellow
docker-compose up -d redis

# Wait for Redis
Write-Host "Waiting for Redis to be ready..." -ForegroundColor Yellow
$redisReady = $false
for ($i = 1; $i -le 30; $i++) {
    $output = docker-compose exec redis redis-cli ping 2>&1
    if ($output -match "PONG") {
        Write-Host "Redis is ready!" -ForegroundColor Green
        $redisReady = $true
        break
    }
    Start-Sleep -Seconds 1
}

if (-not $redisReady) {
    Write-Error "Redis failed to start."
    exit 1
}

# Start Web and Celery
Write-Host "Starting Web and Celery services..." -ForegroundColor Yellow
docker-compose up -d web celery_worker celery_beat

Write-Host "Stack started successfully!" -ForegroundColor Green
Write-Host "Web UI: http://localhost:5000"
Write-Host "Health Check: http://localhost:5000/health"
