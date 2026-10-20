# Nusantara Fire Sentinel — ML Worker AMD Cloud (ROCm)

[🇬🇧 Read in English](README.md) • [📜 Lisensi MIT](LICENSE) • [🛡️ Panduan Kontribusi](CONTRIBUTING-IN.md)

Stateless Heavy-Compute GPU/CPU Machine Learning Inference Engine untuk Sistem Pencegahan dan Penanggulangan Kebakaran Hutan & Lahan Gambut (Karhutla) Indonesia.

---

## 🚀 Peran & Arsitektur

Repositori ini memuat *inference worker* kecerdasan buatan khusus untuk platform Nusantara Fire Sentinel, yang dirancang khusus untuk deployment pada akselerator komputasi **AMD Cloud** (AMD Instinct MI300X, MI250, MI210, atau workstation AMD Radeon Pro / EPYC) dengan kompatibilitas penuh untuk dijalankan di mode CPU lokal:

* **Layanan gRPC Throughput Tinggi (Port 50051)**: Menerima request inferensi spasial batch berkecepatan tinggi yang diserialisasi melalui Protocol Buffers dari backend Control Plane GCP.
* **Peatland XGBoost Risk Engine**: Menghitung indeks risiko penyalaan gambut (*Ignition Risk Score*) secara presisi tinggi berdasarkan regulasi pemerintah **PP 57/2016** (memperhitungkan Tinggi Muka Air Tanah / GWL, defisit curah hujan, kedalaman gambut, kelembaban tanah, dan jarak ke kanal).
* **Deteksi Anomali Radar SAR Sentinel-1**: Mendeteksi aktivitas tebas-bakar (*slash-and-burn*) dan pembukaan lahan ilegal pra-penyalaan api menggunakan analisis rasio hamburan balik radar (*C-Band Sigma0 VV/VH cross-polarization*), mampu menembus tutupan awan dan asap tebal sebelum titik panas termal (hotspot) terdeteksi.

---

## 📋 Prasyarat Sistem

Untuk pengembangan dan pengujian lokal:

* **Python**: Versi 3.10 atau 3.11
* **pip & virtualenv**
* **Docker & Docker Compose** (Opsional, untuk menjalankan secara terisolasi via kontainer)
* *(Opsional untuk akselerasi GPU)*: AMD ROCm 6.x toolkit dan driver kartu grafis AMD yang didukung.

---

## 🛠️ Menjalankan Secara Lokal (Mode CPU)

Anda dapat menjalankan ML Worker secara langsung di komputer lokal tanpa memerlukan GPU AMD:

### Metode 1: Python Virtual Environment (Paling Cepat)

```bash
# 1. Masuk ke direktori ML worker
cd karhutla-ml-worker-amd

# 2. Buat dan aktifkan Virtual Environment Python
# Di Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# Di Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# Di Windows (Command Prompt):
python -m venv venv
.\venv\Scripts\activate.bat

# 3. Pasang paket dependensi Python
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Opsional) Kompilasi ulang Protocol Buffers jika Anda mengubah file proto
python compile_proto.py

# 5. Jalankan server gRPC
python grpc_server.py
```

Server gRPC akan aktif dan mendengarkan request pada port `50051`:
```text
[INFO] Starting Nusantara Fire Sentinel ML gRPC Worker...
[INFO] Loaded Peatland XGBoost model successfully.
[INFO] gRPC server active on [::]:50051 (stateless workers: 10)
```

> **Skrip Otomatis**: Pada Windows, Anda juga bisa langsung menjalankan `.\run_local.bat` atau `.\run_local.ps1`. Pada Linux/macOS, jalankan `./run_local.sh`.

---

### Metode 2: Container Docker Lokal

Jika Anda lebih memilih menjalankan server dalam kontainer Docker yang bersih dan terisolasi:

```bash
# Build dan jalankan container CPU lokal
docker compose -f docker-compose.local.yml up -d --build

# Pantau status dan log container
docker compose -f docker-compose.local.yml ps
docker compose -f docker-compose.local.yml logs -f
```

Kontainer akan mengekspos port `50051` ke `localhost:50051`.

---

## 🧪 Menguji Endpoint gRPC

Anda dapat memastikan bahwa server inferensi gRPC lokal berfungsi dengan benar menggunakan skrip klien pengujian bawaan:

```bash
# Jalankan test client ke endpoint lokal
python test_client.py --endpoint localhost:50051
```

Hasil yang diharapkan:
* Mengirimkan data spatiotemporal simulasi lahan gambut (GWL, curah hujan, ketebalan gambut).
* Menerima respon kategori risiko (`LOW`, `MODERATE`, `HIGH`, atau `EXTREME`) dengan latensi dalam hitungan milidetik.
* Memvalidasi evaluasi deteksi anomali radar SAR Sentinel-1.

---

## 🐳 Menjalankan di AMD Cloud (ROCm GPU Passthrough)

Untuk deployment di lingkungan server produksi AMD Cloud dengan akselerasi GPU AMD ROCm:

```bash
# Jalankan container dengan ROCm GPU passthrough
docker compose up -d
```

Atau jalankan kontainer secara langsung menggunakan perintah Docker:
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

Untuk panduan mendalam mengenai *provisioning VM*, konfigurasi driver ROCm 6.x, reverse proxy Nginx gRPC TLS (Port 443), dan aktivasi CI/CD GitHub Actions, silakan buka dokumen [AMD_SETUP.md](./AMD_SETUP.md).

---

## 📜 Lisensi & Hak Penggunaan Open Source

Repositori ini dilisensikan di bawah **[MIT License](LICENSE)**.

```
Copyright (c) 2026 Achmad Yusuf Ramadhan (https://linkedin.com/in/cupu2903)
```

Siapa pun memiliki hak penuh untuk me-**Fork** repositori ini, melatih arsitektur model alternatif, memodifikasi pipeline *feature engineering*, dan mendeploy instalasi mandiri untuk keperluan akademik, instansi, maupun privat di bawah ketentuan Lisensi MIT.

---

## 🛡️ Kebijakan Kontribusi

Repositori upstream resmi dikelola menggunakan **Model Kontribusi Terkurasi / Terpilih (Curated Contribution Model)**. 
Pembuatan Pull Request langsung, pembukaan Issue, dan hak Merge di repositori upstream dibatasi hanya untuk maintainer dan kolaborator resmi yang telah disetujui demi menjaga determinisme inferensi model kebencanaan kritis dan stabilitas driver komputasi perangkat keras.

Jika Anda telah mengembangkan perbaikan atau model baru pada fork Anda dan ingin mengajukannya ke repositori resmi, silakan baca **[Panduan Kontribusi](CONTRIBUTING-IN.md)** dan hubungi:
* **Maintainer**: Achmad Yusuf Ramadhan
* **LinkedIn**: [https://linkedin.com/in/cupu2903](https://linkedin.com/in/cupu2903)
* **GitHub**: [@ayusuf29](https://github.com/ayusuf29)
