# Local Offline Speech-to-Text Pipeline (OpenAI Whisper + CUDA Acceleration)

Pipeline transkripsi otomatis audio-ke-teks (*Speech-to-Text* / STT) offline berbahasa Indonesia menggunakan model **OpenAI Whisper (turbo)** yang dioptimasi untuk komputasi perangkat keras berbasis **GPU NVIDIA CUDA 12.4**.

Sistem ini dirancang untuk memproses berkas audio rapat, perkuliahan, atau wawancara berdurasi panjang (2–3 jam) secara cepat, akurat, dan sepenuhnya mandiri tanpa mengirim data ke layanan *cloud* pihak ketiga.

---

## 🚀 Fitur Utama

* **100% Offline & Private:** Tidak ada berkas audio atau transkrip teks yang keluar dari mesin lokal.
* **Akselerasi Hardware GPU:** Menggunakan PyTorch dengan akselerasi NVIDIA CUDA 12.4 (teruji pada GPU laptop Lenovo IdeaPad Gaming 3 - RTX 3050).
* **Pemrosesan Audio Durasi Panjang:** Mampu mentranskripsi rekaman berdurasi berjam-jam dengan latensi rendah dan konsumsi VRAM terukur.
* **Optimasi Bahasa Indonesia:** Konfigurasi parameter suhu rendah (`temperature=0.0`) untuk meminimalkan halusinasi kata pada tuturan formal maupun santai.

---

## 🛠️ Instalasi & Persyaratan

Pastikan driver NVIDIA dan toolkit CUDA sudah terpasang di sistem operasi Anda:

```bash
pip install -r requirements.txt
```

---

## 💻 Penggunaan

Jalankan script transkripsi dengan menyertakan path berkas audio:

```bash
python src/transcribe.py contoh_rekaman.mp3
```

Keluaran teks transkripsi akan otomatis disimpan ke berkas `contoh_rekaman_transcript.txt`.
