# Local Offline Speech-to-Text Pipeline (OpenAI Whisper + CUDA Acceleration)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.4-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Whisper](https://img.shields.io/badge/Model-OpenAI%20Whisper%20Turbo-412991?logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20Offline%20Air--Gapped-success)]()

Pipeline transkripsi otomatis audio-ke-teks (*Speech-to-Text* / STT) berkinerja tinggi yang beroperasi secara **100% offline** pada mesin komputasi lokal. Sistem ini mengintegrasikan model arsitektur encoder-decoder transformator **OpenAI Whisper (turbo)** berbasis **PyTorch** dengan akselerasi perangkat keras GPU **NVIDIA CUDA 12.4**.

Proyek rekayasa perangkat lunak ini dirancang untuk memproses berkas audio perkuliahan, rapat, dan wawancara berbahasa Indonesia berdurasi panjang (2–3+ jam) secara presisi tanpa ketergantungan pada API cloud pihak ketiga, menjamin kerahasiaan dan privasi data secara menyeluruh.

---

## 🏛️ Arsitektur Pipeline Pemrosesan

```text
┌───────────────────────┐
│ Input Audio           │ (.mp3, .wav, .m4a, .aac)
└──────────┬────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ Pre-Processing Audio Engine                            │
│ • Resampling otomatis ke 16 kHz mono                   │
│ • Ekstraksi representasi Log-Mel Spectrogram (80-chan) │
└──────────┬─────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ NVIDIA CUDA 12.4 GPU Acceleration (PyTorch)            │
│ • Model Whisper (Varian 'turbo' / 809M Parameter)      │
│ • Encoder Audio Transformer (Self-Attention Blocks)    │
│ • Decoder Autoregresif dengan Dekode Beam-Search       │
└──────────┬─────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ Bahasa & Post-Processing Teks Indonesia                │
│ • Decoding deterministik (Temperature = 0.0)           │
│ • Rekonstruksi tanda baca dan kapitalisasi otomatis    │
│ • Ekspor berkas transkrip teks (.txt / .srt)           │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Fitur Unggulan

* **Zero-Cloud Dependency (Privasi Mutlak):** Seluruh inferensi pemodelan kecerdasan buatan dijalankan di workstation lokal. Tidak ada data audio, suara pembicara, maupun transkrip teks yang terkirim ke internet.
* **Akselerasi Komputasi GPU CUDA:** Mengoptimalkan komputasi tensor FP16 pada kartu grafis NVIDIA RTX (teruji pada GPU Lenovo IdeaPad Gaming 3 - RTX 3050, 4GB VRAM).
* **Toleransi Terhadap Audio Panjang:** Menggunakan algoritma jendela geser dinamis (*sliding-window chunking*) untuk memproses rekaman berdurasi 2 hingga 3 jam tanpa mengalami *memory leak* atau degradasi akurasi waktu (*timestamp*).
* **Optimasi Kosakata Bahasa Indonesia:** Mengonfigurasi parameter token bahasa (`language="id"`) dan *temperature* $0,0$ untuk menekan halusinasi kata pada tuturan formal akademis maupun percakapan santai.

---

## ⚙️ Spesifikasi Lingkungan Pengujian

| Komponen Sistem | Spesifikasi Rekayasa |
|:---|:---|
| **Perangkat Keras Uji** | Laptop Lenovo IdeaPad Gaming 3 (AMD Ryzen 5, 16 GB RAM) |
| **Akselerator Grafis** | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6) |
| **CUDA Toolkit Version** | **NVIDIA CUDA 12.4** (Driver version $\ge 550.x$) |
| **Framework Deep Learning**| PyTorch 2.2+ dengan dukungan pustaka `cu124` |
| **Bobot Model Whisper** | `whisper-turbo` (Arsitektur Whisper v3 yang dipadatkan, $\approx 809\text{M}$ parameter) |

---

## 📂 Struktur Repositori

```text
├── src/
│   └── transcribe.py                # Skrip inti eksekusi pipeline transkripsi audio
├── requirements.txt                 # Daftar paket pustaka Python yang dibutuhkan
└── README.md
```

---

## 🛠️ Panduan Instalasi dan Penggunaan

### 1. Instalasi Lingkungan Virtual & Dependensi

Pastikan driver kartu grafis NVIDIA dan Python 3.10+ telah terpasang:

```bash
# Clone repositori
git clone https://github.com/Xloress/local-whisper-transcription-pipeline.git
cd local-whisper-transcription-pipeline

# Buat virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Pasang PyTorch dengan dukungan CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Pasang dependensi Whisper
pip install -r requirements.txt
```

### 2. Menjalankan Transkripsi Berkas Audio

Untuk mentranskripsi rekaman audio, jalankan perintah berikut:

```bash
python src/transcribe.py "data/rekaman_kuliah.mp3"
```

Skrip akan otomatis mendeteksi ketersediaan GPU CUDA, memuat bobot model ke VRAM, dan menyimpan berkas teks hasil transkripsi ke `data/rekaman_kuliah_transcript.txt`.
