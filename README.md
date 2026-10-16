# Nusantara Fire Sentinel — AMD Cloud ML Worker (ROCm)

Stateless Heavy-Compute GPU Inference Engine untuk sistem pencegahan dan penanggulangan kebakaran hutan & lahan gambut (Karhutla) Indonesia.

## 🚀 Peran & Arsitektur
Repositori ini dirancang khusus untuk dideploy pada infrastruktur **AMD Cloud** (AMD Instinct MI300X, MI250, MI210, atau AMD Radeon Pro / EPYC):
* **gRPC Service (Port 50051):** Menerima request inferensi spasial dari Control Plane di Google Cloud Platform (GCP).
* **Peatland XGBoost Engine:** Menghitung indeks risiko penyalaan gambut (*Ignition Risk Score*) berdasarkan PP 57/2016 (*Ground Water Level*, curah hujan, kedalaman gambut, kanal).
* **SAR Sentinel-1 Anomaly Detection:** Deteksi anomali *land-clearing* / tebas-bakar pra-penyalaan api menggunakan perbandingan rasio hamburan balik radar (*C-Band Sigma0 VV/VH*).

## 🛠️ Menjalankan Secara Lokal (CPU Mode)
```bash
# 1. Buat Virtual Environment
python -m venv venv
source venv/bin/activate  # atau venv\Scripts\activate di Windows

# 2. Install Dependensi
pip install -r requirements.txt

# 3. Jalankan gRPC Server
python grpc_server.py
```

## 🧪 Menguji Endpoint gRPC
```bash
python test_client.py --endpoint localhost:50051
```

## 🐳 Menjalankan dengan Docker di AMD Cloud (ROCm GPU Passthrough)
```bash
docker compose up -d
```
Atau jalankan kontainer secara langsung:
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

## 📖 Panduan Lengkap Hosting AMD Cloud
Lihat [AMD_SETUP.md](./AMD_SETUP.md) untuk panduan provisioning VM, setup driver ROCm 6.x, konfigurasi Nginx gRPC TLS (Port 443), dan aktivasi CI/CD GitHub Actions.
