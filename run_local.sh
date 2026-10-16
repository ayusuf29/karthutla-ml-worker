#!/usr/bin/env bash
set -e

# Navigate to ml-worker directory
cd "$(dirname "$0")"

echo "=========================================================="
echo "Starting Nusantara Fire Sentinel ML Worker (CPU Local)"
echo "=========================================================="

# Check Python version
python3 --version || python --version

# Compile protobuf if needed
if [ ! -f "fire_risk_service_pb2.py" ] || [ ! -f "fire_risk_service_pb2_grpc.py" ]; then
    echo "Compiling protobuf stubs..."
    python3 compile_proto.py || python compile_proto.py
fi

# Run gRPC server
export GRPC_PORT=${GRPC_PORT:-50051}
echo "Running gRPC server on port ${GRPC_PORT}..."
python3 grpc_server.py || python grpc_server.py
