# Panduan Lengkap Pipeline Transkripsi Suara Lokal (OpenAI Whisper + Akselerasi GPU CUDA)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.4-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Whisper](https://img.shields.io/badge/Model-OpenAI%20Whisper-412991?logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![GPU](https://img.shields.io/badge/Tested%20Hardware-Lenovo%20IdeaPad%20%7C%20RTX%203050-76B900?logo=nvidia&logoColor=white)]()
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20Offline%20Air--Gapped-success)]()

Repositori ini memuat **panduan lengkap, struktur proyek terstandarisasi, analisis benchmark kecepatan nyata, serta skrip eksekusi *production-grade*** untuk mentranskripsi berkas audio dan video menjadi teks secara **100% offline** menggunakan model **OpenAI Whisper** dengan akselerasi perangkat keras **GPU NVIDIA CUDA 12.4**.

Pipeline ini dirancang khusus bagi mahasiswa, peneliti, dan profesional yang membutuhkan transkripsi rekaman perkuliahan, rapat daring, maupun wawancara berbahasa Indonesia berdurasi panjang (2 hingga 3+ jam) secara **cepat, akurat, hemat VRAM, dan menjaga privasi mutlak tanpa mengirim data ke internet**.

---

## ⚡ Hasil Uji Coba Nyata: Benchmark Kecepatan GPU NVIDIA RTX 3050

Seringkali pengguna pemula bingung: *"Berapa lama waktu yang dibutuhkan untuk mengubah rekaman video berdurasi 2–3 jam menjadi teks jika memakai laptop biasa?"*

Berikut adalah data hasil pengujian nyata (*real-world benchmark*) pada workstation laptop harian:

* **Spesifikasi Laptop Uji:** Lenovo IdeaPad Gaming 3 (AMD Ryzen 5, 16 GB RAM).
* **Kartu Grafis (GPU):** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6 VRAM, CUDA 12.4).
* **Ukuran Berkas Input:** Video rekaman materi kuliah berdurasi **2 Jam 45 Menit (165 Menit)** berbahasa Indonesia.

### Tabel Perbandingan Waktu Proses (Durasi 2 Jam 45 Menit):

| Varian Model Whisper | Alokasi VRAM GPU | Waktu Eksekusi (RTX 3050 CUDA) | Faktor Kecepatan (*Speedup*) | Akurasi Bahasa Indonesia |
|:---|:---:|:---:|:---:|:---|
| **`tiny`** | $\approx 1,0\text{ GB}$ | **$\approx 6 - 8\text{ Menit}$** | $\approx 25\times\text{ Realtime}$ | Cukup untuk audio jernih (kurang peka istilah teknis) |
| **`base`** *(Paling Direkomendasikan)* | **$\approx 1,2\text{ GB}$** | **$\approx 10 - 15\text{ Menit}$** | **$\approx 15\times\text{ Realtime}$** | **Sangat Baik, cepat, dan sangat ringan untuk RTX 3050** |
| **`small`** | $\approx 2,0\text{ GB}$ | $\approx 25 - 35\text{ Menit}$ | $\approx 6\times\text{ Realtime}$ | Bagus untuk rekaman dengan noise latar sedang |
| **`turbo`** *(Akurasi Tinggi)* | $\approx 3,2\text{ GB}$ | $\approx 20 - 30\text{ Menit}$ | $\approx 7\times\text{ Realtime}$ | **Akurasi tertinggi** untuk istilah teknis / campur bahasa |
| *Mode CPU (Tanpa GPU)* | RAM Sistem | $\approx 180 - 240\text{ Menit}$ (3–4 Jam) | $< 1\times\text{ Realtime}$ | Sangat lambat dan membuat laptop panas |

> **Kesimpulan Pengujian:** Menggunakan akselerasi CUDA pada GPU RTX 3050 memangkas waktu kerja dari berjam-jam menjadi hanya **10–15 menit saja**, dengan konsumsi VRAM yang sangat aman di bawah batas 4 GB.

---

## 🎯 Panduan Memilih Model yang Tepat (Agar Tidak Bingung)

Whisper menyediakan beberapa ukuran model. Gunakan panduan berikut sesuai kebutuhan Anda:

1. **Gunakan Model `base` jika:**
   * Anda ingin proses super kilat (video 2,5 jam selesai ~10 menit).
   * Kualitas rekaman audio cukup jelas (suara dosen/narasumber terdengar jelas).
   * Laptop sedang digunakan untuk pekerjaan lain sambil menunggu transkripsi berjalan.
2. **Gunakan Model `turbo` jika:**
   * Rekaman memuat banyak istilah asing/teknis (misal: istilah telekomunikasi, rumus, bahasa Inggris campur Indonesia).
   * Anda membutuhkan kapitalisasi huruf dan tanda baca (*punctuation*) yang sangat rapi untuk laporan formal.
3. **Hindari Model `large` pada GPU 4 GB:**
   * Varian `large-v3` membutuhkan VRAM $> 6\text{ GB}$ yang dapat memicu galat `CUDA Out of Memory (OOM)` pada RTX 3050.

---

## 📂 Struktur Berkas Proyek yang Rapi & Terstandarisasi

Agar file tidak berantakan dan mudah dikelola oleh siapa pun, repositori ini mengadopsi struktur folder modular:

```text
local-whisper-transcription-pipeline/
│
├── src/                               # Direktori kode sumber utama
│   └── transcribe.py                  # Skrip CLI transkripsi mandiri (dukungan TXT & SRT)
│
├── examples/                          # Sampel keluaran hasil transkripsi nyata
│   ├── kuliah_telekomunikasi_sample.txt   # Contoh format teks biasa (.txt)
│   └── kuliah_telekomunikasi_sample.srt   # Contoh format subtitle video lengkap (.srt)
│
├── data/                              # (Opsional) Tempat Anda meletakkan file audio (.mp3/.mp4)
├── output/                            # Folder otomatis tempat hasil transkrip disimpan
│
├── requirements.txt                   # Daftar dependensi library Python
└── README.md                          # Buku panduan lengkap (file ini)
```

---

## 📖 Tutorial Instalasi & Penggunaan dari Nol (Step-by-Step Pemula)

Ikuti 4 langkah mudah berikut di laptop Anda (Windows / Linux):

### Langkah 1: Pasang FFmpeg (Wajib untuk Membaca Audio/Video)
Whisper membutuhkan utilitas audio **FFmpeg** untuk membaca berbagai format berkas (`.mp3`, `.m4a`, `.wav`, `.mp4`, `.mkv`).

* **Di Ubuntu / Debian Linux:**
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
* **Di Windows:**
  Unduh installer FFmpeg dari situs resmi atau gunakan package manager:
  ```powershell
  winget install Gyan.FFmpeg
  ```

### Langkah 2: Buat Lingkungan Virtual Python & Pasang Dependensi
Buka Terminal / Command Prompt pada folder proyek ini:

```bash
# 1. Buat virtual environment agar library tidak bentrok
python -m venv .venv

# 2. Aktifkan virtual environment
# Di Linux:
source .venv/bin/activate
# Di Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Pasang PyTorch dengan dukungan CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Pasang OpenAI Whisper dan modul pendukung
pip install -r requirements.txt
```

### Langkah 3: Menjalankan Transkripsi Berkas Audio/Video

Cukup jalankan satu baris perintah berikut di terminal:

#### 1. Mode Cepat Rekomendasi (Model `base` - Selesai ~10 Menit):
```bash
python src/transcribe.py "data/rekaman_kuliah.mp3" --model base --lang id --outdir ./output
```

#### 2. Mode Akurasi Tinggi (Model `turbo`):
```bash
python src/transcribe.py "data/rekaman_kuliah.mp4" --model turbo --lang id --outdir ./output
```

### Langkah 4: Mengambil Hasil Transkripsi
Setelah selesai, skrip otomatis menghasilkan **2 berkas sekaligus** di folder `output/`:
1. **`rekaman_kuliah.txt`:** Naskah teks lengkap yang siap disalin ke Microsoft Word untuk laporan/catatan kuliah.
2. **`rekaman_kuliah.srt`:** Berkas subtitle lengkap dengan penanda waktu (*timestamp*) detik per detik yang bisa langsung dimasukkan ke pemutar video VLC atau YouTube!

---

## 🏛️ Diagram Alir Arsitektur Pipeline

```text
┌───────────────────────┐
│ Input Audio / Video   │ (.mp3, .wav, .m4a, .mp4, .mkv)
└──────────┬────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ Pre-Processing Engine (FFmpeg Core)                    │
│ • Resampling otomatis ke 16.000 Hz mono                │
│ • Konversi gelombang suara ke Log-Mel Spectrogram      │
└──────────┬─────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ NVIDIA CUDA 12.4 Hardware Acceleration (PyTorch)       │
│ • Model Whisper (Encoder-Decoder Transformer)          │
│ • Pemrosesan tensor FP16 berkecepatan tinggi           │
│ • Ekstraksi teks autoregresif dengan Beam Search       │
└──────────┬─────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ Output Generator                                       │
│ • File Naskah Dokumen Lengkap : output/*.txt           │
│ • File Subtitle Sinkronisasi  : output/*.srt           │
└────────────────────────────────────────────────────────┘
```

---

## 🛡️ Jaminan Privasi Data 100% (Air-Gapped)

Seluruh proses komputasi kecerdasan buatan dilakukan secara lokal di perangkat keras komputer Anda. Tidak ada data suara, data sensitif pembicara, maupun teks transkripsi yang dikirimkan ke cloud atau server pihak ketiga mana pun.
