# Panduan Kontribusi — Nusantara Fire Sentinel (ML Worker)

Terima kasih atas ketertarikan Anda pada repositori **Nusantara Fire Sentinel** Heavy-Compute ML Worker!

[🇬🇧 Read Contribution Guidelines in English](CONTRIBUTING.md)

---

## 📜 Hak Open Source & Bebas Fork (Lisensi MIT)

Repositori ini dilisensikan di bawah **[MIT License](LICENSE)** yang sangat terbuka dan permisif:

* **Sangat Bebas untuk Melakukan Fork**: Siapa pun—baik peneliti data science, machine learning engineer, akademisi, instansi kebencanaan, maupun komunitas pengembang—berhak sepenuhnya untuk melakukan *Fork* repositori ini, melatih ulang bobot model, memodifikasi ekstraksi fitur spasial, dan menjalankan server inferensi mandiri di bawah lisensi MIT.
* **Penggunaan Komersial maupun Non-Komersial**: Anda diperbolehkan menggunakan, mengembangkan, dan mendistribusikan kode ini pada fork Anda sendiri tanpa kewajiban royalti, selama menyertakan atribusi hak cipta dan teks lisensi asli.

---

## 🛡️ Kebijakan Repositori Upstream: Model Kontribusi Selektif (Terkurasi)

Meskipun repositori ini bebas di-fork dan dimodifikasi secara mandiri oleh siapa saja, **repositori upstream resmi** (`ayusuf29/karthutla-ml-worker`) dikelola menggunakan **Model Kontribusi Terkurasi / Terpilih (Curated & Invitation-Only)**:

### Mengapa kontribusi langsung di repositori upstream dibatasi?
1. **Keandalan Inferensi Model Kritis**: ML Worker bertugas menghitung skor risiko kebakaran gambut secara real-time dan mendeteksi anomali tebas-bakar pra-penyalaan via radar Sentinel-1 SAR. Kestabilan numerik, kepatuhan fitur regulasi PP 57/2016 (muka air tanah gambut), dan kompatibilitas kontrak antarmuka Protobuf gRPC sangat penting dijaga.
2. **Kesesuaian Akselerator Perangkat Keras**: Lingkungan produksi dirancang khusus untuk infrastruktur komputasi tinggi AMD Cloud ROCm (AMD Instinct MI300X, MI250, MI210, serta GPU Radeon Pro/EPYC). Perubahan dependensi tanpa verifikasi kernel GPU berisiko menyebabkan memory leak atau degradasi performa komputasi paralel.
3. **Penyaringan PR, Issue, dan Merge**: Untuk menjaga kualitas kode, determinisme inferensi, dan stabilitas operasional, **pembuatan Pull Request, Issue, dan hak Merge di repositori upstream resmi dibatasi secara selektif hanya untuk maintainer utama dan kolaborator resmi yang telah disetujui**.

---

## 🤝 Cara Mengajukan Kontribusi atau Kolaborasi

Jika Anda telah mengembangkan arsitektur model baru yang lebih akurat, serialisasi gRPC yang lebih cepat, atau metode ekstraksi fitur penginderaan jauh yang lebih baik di fork Anda:

1. **Kembangkan di Fork Pribadi**: Buat fork repositori ini dan lakukan pengujian model serta benchmarking performa secara menyeluruh.
2. **Uji Coba Lokal**: Pastikan server gRPC dapat berjalan mulus (`python grpc_server.py`), skrip kompilasi proto berjalan (`python compile_proto.py`), dan pengujian klien (`python test_client.py`) berhasil tanpa galat.
3. **Hubungi Maintainer**: Sebelum membuka Pull Request di repositori utama, silakan hubungi maintainer untuk memaparkan hasil pengujian Anda:
   * **Maintainer**: Achmad Yusuf Ramadhan
   * **LinkedIn**: [https://linkedin.com/in/cupu2903](https://linkedin.com/in/cupu2903)
   * **GitHub**: [@ayusuf29](https://github.com/ayusuf29)
4. **Proses Kolaborasi / Review**: Setelah proposal diselaraskan dan disepakati, maintainer akan mengundang Anda sebagai kolaborator resmi atau memberikan izin pembukaan PR terarah untuk proses review dan merge.

> **Catatan**: Pull Request atau Issue publik yang dibuka tanpa konfirmasi atau keselarasan teknis sebelumnya dapat ditinjau secara selektif atau ditutup secara berkala.

---

## 💻 Standar Teknis untuk Kolaborator

Bagi kolaborator resmi yang mengerjakan branch atau PR:

* **Python & Dependensi**: Menggunakan Python 3.10+, kepatuhan ketat terhadap versi dependensi di `requirements.txt`.
* **Kontrak Protobuf & gRPC**: Tidak melakukan perubahan skema yang *breaking* pada `proto/fire_risk_service.proto` tanpa memperbarui stub klien dan server secara sinkron.
* **Model Serialization**: Simpan bobot model baru secara terstruktur di folder `models/` disertai dokumentasi metrik evaluasi.
* **Keamanan Data**: Dilarang keras melakukan commit data citra satelit mentah berskala besar yang melanggar hak cipta, kredensial cloud, atau token akses pribadi ke dalam repositori.

Terima kasih atas pengertian, apresiasi, dan dedikasi Anda dalam mendukung teknologi kecerdasan buatan untuk perlindungan hutan dan lahan gambut Indonesia!
