@echo off
cd /d "%~dp0"
echo ==========================================================
echo Starting Nusantara Fire Sentinel ML Worker (CPU Local)
echo ==========================================================

if not exist fire_risk_service_pb2.py (
    echo Compiling protobuf stubs...
    python compile_proto.py
)

if not defined GRPC_PORT set GRPC_PORT=50051
echo Running gRPC server on port %GRPC_PORT%...
python grpc_server.py
