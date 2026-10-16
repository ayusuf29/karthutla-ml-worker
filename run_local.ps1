# PowerShell runner for ML Worker
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Starting Nusantara Fire Sentinel ML Worker (CPU Local)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Check protobuf stubs
if (-not (Test-Path "fire_risk_service_pb2.py") -or -not (Test-Path "fire_risk_service_pb2_grpc.py")) {
    Write-Host "Compiling protobuf stubs..." -ForegroundColor Yellow
    python compile_proto.py
}

$env:GRPC_PORT = if ($env:GRPC_PORT) { $env:GRPC_PORT } else { "50051" }
Write-Host "Running gRPC server on port $env:GRPC_PORT..." -ForegroundColor Green
python grpc_server.py
