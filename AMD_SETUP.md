# Panduan Lengkap Setup & Hosting di AMD Cloud
## Nusantara Fire Sentinel — Heavy-Compute GPU ML Inference Island
**Versi Dokumen:** 1.0.0  
**Target Komponen:** `sentinel-ml-worker` (Stateless GPU Inference Engine)  
**Lingkungan:** AMD Instinct / ROCm 6.x Compute Node  

---

## 1. Ringkasan & Ruang Lingkup (AMD Cloud Only)

Dalam arsitektur hybrid multi-cloud Nusantara Fire Sentinel:
* **Hanya komponen heavy compute / ML inference** yang ditempatkan di **AMD Cloud**.
* Komponen ini bertindak sebagai **Stateless High-Performance GPU Island** yang mengekspos endpoint **gRPC** pada port `50051` (atau port `443` dengan reverse proxy TLS).
* Seluruh komponen Control Plane (Frontend Next.js, Backend Java Spring Boot, Cloud SQL PostGIS, Memorystore Redis, dan Cloud Storage) **tetap berada di Google Cloud Platform (GCP)**.

```
+-----------------------------------------------------------------------------------+
| GOOGLE CLOUD PLATFORM (Control Plane)                                             |
| Java 21 / Spring Boot Core Gateway & Next.js Frontend                             |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          | gRPC (HTTP/2) mTLS / TLS (Port 443/50051)
                                          v
+-----------------------------------------------------------------------------------+
| AMD CLOUD (Heavy-Compute GPU Inference Island)                                    |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   | Host VM: Ubuntu 22.04 LTS + AMD ROCm 6.x Driver                           |   |
|   | AMD Instinct MI210 / MI250 / MI300X (atau AMD Radeon Pro / EPYC CPU)      |   |
|   +-------------------------------------+-------------------------------------+   |
|                                         |                                         |
|                 +-----------------------v-----------------------+                 |
|                 | Docker Container: sentinel-ml-worker-amd      |                 |
|                 | - Base: rocm/pytorch:rocm6.1_ubuntu22.04_py3.10 |               |
|                 | - Python gRPC Server (Dual Protobuf / JSON)   |                 |
|                 | - Model XGBoost GPU Hist + SAR Sentinel-1     |                 |
|                 | - Device nodes: /dev/kfd, /dev/dri            |                 |
|                 +-----------------------------------------------+                 |
+-----------------------------------------------------------------------------------+
```

---

## 2. Strategi Pemisahan Repository GitHub

Untuk efisiensi build dan isolasi dependensi runtime C++/ROCm berukuran besar (image ROCm ~10GB–15GB), modul worker ML direkomendasikan dipisahkan ke repositori GitHub tersendiri, misalnya:
`karhutla-ml-worker-amd`

### A. Struktur File di Repository Baru (`karhutla-ml-worker-amd`)
```
karhutla-ml-worker-amd/
├── .github/
│   └── workflows/
│       └── deploy-amd.yml        # CI/CD Auto-Deploy via GitHub Actions
├── deploy/
│   ├── docker-compose.amd.yml    # Konfigurasi container di host AMD
│   └── nginx-grpc.conf           # Konfigurasi Nginx gRPC TLS (Port 443)
├── models/
│   └── karhutla_xgb_v1.json      # Model terlatih XGBoost Karhutla
├── proto/
│   └── fire_risk_service.proto   # Kontrak interface Protobuf gRPC
├── training/                     # Dataset, feature engineering & script training
├── Dockerfile                    # Dockerfile target ROCm 6.x
├── requirements.txt              # Dependensi Python
├── compile_proto.py              # Generator Protobuf Python
├── grpc_server.py                # gRPC Server Daemon
├── risk_model.py                 # Evaluator risiko gambut & GWL
└── test_client.py                # Script validasi / healthcheck inferensi
```

### B. Cara Mengekstrak Modul `ml-worker` ke Repo Baru
Jalankan langkah berikut dari komputer lokal Anda:

```bash
# 1. Buat folder baru untuk repo AMD Worker
mkdir -p /path/to/karhutla-ml-worker-amd
cd /path/to/karhutla-ml-worker-amd

# 2. Inisialisasi Git
git init
git branch -M main

# 3. Salin file-file dari project utama:
cp -r /path/to/karhutla-prevention-system/ml-worker/* .
cp -r /path/to/karhutla-prevention-system/proto ./proto
mkdir -p deploy
cp /path/to/karhutla-prevention-system/deploy/docker/Dockerfile.amd ./Dockerfile
cp /path/to/karhutla-prevention-system/deploy/amd/* ./deploy/

# 4. Tambahkan GitHub Actions workflow
mkdir -p .github/workflows
cp /path/to/karhutla-prevention-system/.github/workflows/deploy-amd.yml .github/workflows/

# 5. Commit dan push ke repository baru di GitHub
git remote add origin https://github.com/<username-atau-org>/karhutla-ml-worker-amd.git
git add .
git commit -m "feat: initial commit for AMD Cloud ML Worker standalone repository"
git push -u origin main
```

---

## 3. Prasyarat & Persiapan Akun AMD Cloud

### A. Pilihan Penyedia Instance AMD Cloud
Anda dapat menggunakan penyedia cloud yang menyediakan compute accelerator AMD (Instinct MI300X, MI250, MI210, Radeon Pro W7900, atau AMD EPYC High-Compute):
* **AMD Developer Cloud** (Direct Accelerator Cloud dari AMD)
* **TensorWave** (Spesialis AMD Instinct MI300X / ROCm Cloud)
* **Lamini AI / Crusoe Cloud / Nscale** (Cloud provider dengan cluster AMD Instinct)
* **Custom Bare-metal / Virtual Machine** dengan GPU AMD yang terpasang

### B. Spesifikasi Instance yang Direkomendasikan
* **OS:** Ubuntu 22.04 LTS (x86_64) 64-bit
* **GPU:** 1x AMD Instinct MI210 / MI250 / MI300X (atau Radeon Pro W7900)
  *(Catatan: Worker juga dapat berjalan dalam CPU High-Throughput fallback mode jika GPU sedang offline)*
* **RAM:** Minimal 16 GB (Direkomendasikan 32 GB+)
* **Penyimpanan:** Minimal 60 GB SSD (karena image Docker ROCm memerlukan ~15 GB disk space)
* **Jaringan:** Alamat IPv4 Publik Statis atau Domain DNS (misal: `ml.sentinel.domain.id`)

---

## 4. Persiapan Host Server VM di AMD Cloud

Jalankan perintah berikut melalui SSH di VM AMD Cloud Anda sebelum melakukan deployment otomatis pertama kali:

### Langkah 1: Verifikasi Driver ROCm & GPU
Pastikan driver AMD ROCm telah terinstall dan GPU dikenali:
```bash
# Cek ketersediaan perintah rocm-smi
rocm-smi

# Cek device node AMD GPU kernel
ls -la /dev/kfd /dev/dri
```
*Output yang valid:* `/dev/kfd` harus ada dan `/dev/dri/renderD128` (atau nomor render lain) harus dapat diakses.

### Langkah 2: Install Docker Engine di Ubuntu 22.04
Jika Docker belum terpasang:
```bash
# Update repository
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Tambahkan GPG key resmi Docker
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Tambahkan Docker repo
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker dan Docker Compose plugin
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### Langkah 3: Konfigurasi Akses GPU untuk Pengguna Docker
Beri hak akses grup `video` dan `render` kepada user Anda agar kontainer Docker dapat mengakses akselerator ROCm tanpa hambatan:
```bash
sudo usermod -aG docker $USER
sudo usermod -aG video,render $USER
newgrp docker
```

Uji coba kemampuan Docker membaca GPU AMD:
```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined rocm/pytorch:rocm6.1_ubuntu22.04_py3.10 rocm-smi
```

### Langkah 4: Konfigurasi Firewall Server (UFW / Security Group)
Buka port yang dibutuhkan untuk komunikasi dari GCP:
```bash
# SSH Management
sudo ufw allow 22/tcp

# Direct gRPC Port
sudo ufw allow 50051/tcp

# Jika menggunakan Nginx TLS Reverse Proxy
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

sudo ufw enable
```

---

## 5. Konfigurasi Domain & TLS (Rekomendasi Produksi)

gRPC menggunakan protokol **HTTP/2**. Agar backend di GCP dapat menghubungi worker secara aman عبر internet publik tanpa VPN murni:

### Opsi A: Direct gRPC Port 50051 (Plaintext atau mTLS)
* Cocok untuk koneksi awal / staging atau jika Anda menggunakan VPN WireGuard / Tailscale antar GCP dan AMD Cloud.
* URL di backend GCP: `<IP_PUBLIK_AMD>:50051`.

### Opsi B: Nginx Reverse Proxy dengan SSL/TLS Port 443 (Sangat Direkomendasikan)
1. Arahkan DNS Subdomain (misal `ml.sentinel.domain.id`) ke IP Publik VM AMD Cloud.
2. Install Certbot untuk membuat sertifikat SSL gratis:
   ```bash
   sudo apt-get install -y certbot
   sudo certbot certonly --standalone -d ml.sentinel.domain.id
   ```
3. Gunakan konfigurasi Nginx gRPC yang disediakan di `deploy/amd/nginx-grpc.conf`:
   ```bash
   # Jalankan Nginx container via docker-compose
   docker compose -f deploy/amd/docker-compose.amd.yml up -d
   ```
4. URL di backend GCP cukup menjadi: `dns:///ml.sentinel.domain.id:443` dengan flag `USE_TLS=true`.

---

## 6. GitHub Actions: Auto-Deploy on Main Branch

File workflow telah dibuat di [.github/workflows/deploy-amd.yml](file:///c:/project/karhutla-prevention-system/.github/workflows/deploy-amd.yml).

### A. Alur Kerja CI/CD
1. Setiap ada push ke branch `main` pada modul ML / AMD.
2. GitHub Runner membangun image container menggunakan base image AMD ROCm 6.1.
3. Image dipush ke **GitHub Container Registry (GHCR)** (`ghcr.io/<org>/karhutla-ml-worker:latest`).
4. GitHub Actions menghubungkan diri ke VM AMD Cloud melalui **SSH Key**.
5. Di VM AMD Cloud, container lama di-stop, image terbaru di-pull, lalu kontainer baru dijalankan ulang dengan pemetaan device `/dev/kfd` dan `/dev/dri`.
6. Healthcheck otomatis memverifikasi bahwa port `50051` sudah menerima koneksi TCP/gRPC.

### B. Daftar GitHub Secrets yang Harus Disiapkan
Masuk ke repository GitHub Anda -> **Settings** -> **Secrets and variables** -> **Actions** -> **New repository secret**:

| Nama Secret | Deskripsi | Contoh Nilai |
|---|---|---|
| `AMD_SSH_HOST` | IP Publik atau domain dari server AMD Cloud | `198.51.100.45` atau `amd-node.domain.id` |
| `AMD_SSH_USER` | Username login SSH ke server AMD Cloud | `ubuntu` atau `root` |
| `AMD_SSH_PRIVATE_KEY` | Isi private key SSH (format OpenSSH) untuk login ke VM | `-----BEGIN OPENSSH PRIVATE KEY----- ...` |
| `AMD_SSH_PORT` | Port SSH server (opsional, default 22) | `22` |

> [!TIP]
> **Cara Membuat SSH Key Khusus untuk GitHub Actions:**
> ```bash
> ssh-keygen -t ed25519 -C "github-actions-deploy@amd-cloud" -f ./amd_deploy_key
> # Masukkan isi file amd_deploy_key ke secret AMD_SSH_PRIVATE_KEY
> # Tambahkan isi file amd_deploy_key.pub ke ~/.ssh/authorized_keys di VM AMD Cloud
> ```

---

## 7. Validasi & Pengujian End-to-End

Setelah auto-deploy berhasil di GitHub Actions, lakukan pengujian untuk memastikan service inferensi siap melayani backend GCP:

### A. Pengujian Lokal dari VM AMD Cloud
```bash
# Masuk ke container atau jalankan test_client.py
cd /path/to/karhutla-ml-worker-amd
python test_client.py --endpoint localhost:50051
```
*Output yang diharapkan:*
```
[INFO] Sending BatchPredictIgnitionRisk request with 5 H3 cells...
[SUCCESS] Received BatchRiskResponse:
  - Total items: 5
  - Compute time: 14 ms
  - Accelerator Hardware: AMD Instinct MI300X (ROCm 6.1 HipBLAS)
```

### B. Pengujian Eksternal dari Komputer Lokal / Server Luar
```bash
python test_client.py --endpoint ml.sentinel.domain.id:443 --use-tls
```

### C. Monitoring Beban GPU AMD
Pantau utilisasi memori VRAM dan compute core AMD saat request masuk:
```bash
watch -n 1 rocm-smi --showuse --showmeminfo vram
```

---

## 8. Troubleshooting Masalah Umum

| Gejala | Penyebab Umum | Solusi |
|---|---|---|
| `docker: Error response: error gathering device information: /dev/kfd` | Driver AMD ROCm belum aktif di host VM. | Jalankan `sudo modprobe amdgpu` dan pastikan package `amdgpu-dkms` terinstall di host. |
| `Permission denied` saat akses `/dev/kfd` di container | User docker belum tergabung di grup `video` dan `render`. | Jalankan `sudo usermod -aG video,render $USER` lalu restart daemon Docker (`sudo systemctl restart docker`). |
| `gRPC Unavailable: Connection refused` | Firewall belum mengizinkan port 50051 atau container crash. | Periksa log container dengan `docker logs -f sentinel-ml-worker-amd` dan cek `sudo ufw status`. |
| `HSA_STATUS_ERROR_DEVICE_NOT_FOUND` pada kartu RDNA3 non-server | ID GFX belum dikenali secara default oleh HIP runtime. | Tambahkan variabel lingkungan `HSA_OVERRIDE_GFX_VERSION="11.0.0"` pada file konfigurasi container. |
