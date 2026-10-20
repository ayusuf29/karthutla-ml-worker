# Nusantara Fire Sentinel — AMD Cloud ML Worker (ROCm)

[🇮🇩 Baca dalam Bahasa Indonesia](README-IN.md) • [📜 MIT License](LICENSE) • [🛡️ Contribution Guidelines](CONTRIBUTING.md)

Stateless Heavy-Compute GPU/CPU Machine Learning Inference Engine for Indonesia's Peatland Wildfire (*Karhutla*) Prevention and Early Warning System.

---

## 🚀 Role & Architecture

This repository contains the specialized machine learning inference worker of the Nusantara Fire Sentinel architecture, engineered for deployment on **AMD Cloud accelerators** (AMD Instinct MI300X, MI250, MI210, or AMD Radeon Pro / EPYC workstations) with full CPU-mode local compatibility:

* **High-Throughput gRPC Service (Port 50051)**: Receives batched geospatial spatial inference requests serialized via Protocol Buffers from the GCP Control Plane backend.
* **Peatland XGBoost Risk Engine**: Computes high-precision peatland ignition risk scores (*Ignition Risk Score*) adhering strictly to Indonesian peatland regulation **PP 57/2016** (incorporating Ground Water Level / GWL, rainfall deficit, peat depth, soil moisture, and canal proximity).
* **SAR Sentinel-1 Radar Anomaly Detector**: Detects pre-ignition land-clearing and slash-and-burn activities using radar backscatter ratio analysis (*C-Band Sigma0 VV/VH cross-polarization ratio*), penetrating cloud and haze covers before thermal hotspots appear.

---

## 📋 Prerequisites

For local development and testing:

* **Python**: Version 3.10 or 3.11
* **pip & virtualenv**
* **Docker & Docker Compose** (Optional, for containerized local execution)
* *(Optional for GPU acceleration)*: AMD ROCm 6.x toolkit and supported AMD GPU driver.

---

## 🛠️ Running Locally (CPU Mode)

You can run the ML Worker directly on your local workstation without requiring an AMD GPU:

### Method 1: Python Virtual Environment (Fastest)

```bash
# 1. Navigate to the ML worker directory
cd karhutla-ml-worker-amd

# 2. Create and activate a Python virtual environment
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
python -m venv venv
.\venv\Scripts\activate.bat

# 3. Install required Python packages
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Recompile Protocol Buffers if you modified proto definitions
python compile_proto.py

# 5. Start the gRPC server
python grpc_server.py
```

The gRPC server will start and listen on port `50051`:
```text
[INFO] Starting Nusantara Fire Sentinel ML gRPC Worker...
[INFO] Loaded Peatland XGBoost model successfully.
[INFO] gRPC server active on [::]:50051 (stateless workers: 10)
```

> **Helper scripts**: On Windows, you can also simply run `.\run_local.bat` or `.\run_local.ps1`. On Linux/macOS, execute `./run_local.sh`.

---

### Method 2: Local Docker Container

If you prefer to run inside an isolated Docker container without configuring local Python environments:

```bash
# Build and run the local CPU container
docker compose -f docker-compose.local.yml up -d --build

# View container status and logs
docker compose -f docker-compose.local.yml ps
docker compose -f docker-compose.local.yml logs -f
```

The container exposes port `50051` to `localhost:50051`.

---

## 🧪 Testing the gRPC Endpoint

You can verify that the local ML Worker is functioning properly using the built-in test client:

```bash
# Run client test against the local gRPC server
python test_client.py --endpoint localhost:50051
```

Expected output:
* Submits synthetic peatland spatiotemporal features (GWL, rainfall, peat thickness).
* Receives inferred risk categories (`LOW`, `MODERATE`, `HIGH`, or `EXTREME`) with millisecond-level response times.
* Validates SAR Sentinel-1 anomaly detection flag evaluation.

---

## 🐳 Production Deployment on AMD Cloud (ROCm GPU Passthrough)

When deploying to production on AMD Cloud instances equipped with ROCm GPU accelerators:

```bash
# Start container with ROCm GPU passthrough
docker compose up -d
```

Or deploy directly via Docker CLI with KFD/DRI device passthrough:
```bash
docker run -d \
  --name sentinel-ml-worker-amd \
  --restart unless-stopped \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --group-add render \
  -p 50051:50051 \
  -e GRPC_PORT=50051 \
  ghcr.io/<OWNER>/karhutla-ml-worker-amd:latest
```

For complete instructions on VM provisioning, ROCm 6.x driver configuration, Nginx gRPC TLS reverse proxying (Port 443), and automated GitHub Actions CI/CD, refer to [AMD_SETUP.md](./AMD_SETUP.md).

---

## 📜 License & Open Source Rights

This project is licensed under the **[MIT License](LICENSE)**.

```
Copyright (c) 2026 Achmad Yusuf Ramadhan (https://linkedin.com/in/cupu2903)
```

You are completely free to **Fork** this repository, train alternative models, adapt the feature engineering pipelines, and deploy custom instances for academic, governmental, or private purposes under the MIT License terms.

---

## 🛡️ Contribution Policy

The upstream repository operates under a **Curated and Selective Contribution Model**. 
Direct Pull Requests, Issue creation, and Merge permissions on the official upstream repository are restricted to approved maintainers and collaborators to safeguard mission-critical model stability and hardware driver compatibility.

If you have developed enhancements on your fork and wish to collaborate upstream, please read our **[Contribution Guidelines](CONTRIBUTING.md)** and contact:
* **Maintainer**: Achmad Yusuf Ramadhan
* **LinkedIn**: [https://linkedin.com/in/cupu2903](https://linkedin.com/in/cupu2903)
* **GitHub**: [@ayusuf29](https://github.com/ayusuf29)
