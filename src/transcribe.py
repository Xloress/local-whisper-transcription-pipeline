#!/usr/bin/env python3
"""
Local Speech-to-Text Pipeline using OpenAI Whisper (turbo) with GPU CUDA acceleration.
Designed for long-duration audio files (2-3+ hours) in Indonesian with complete privacy.
"""
import os
import sys
import time

def transcribe_audio(audio_path, model_size="turbo", language="id", output_format="txt"):
    """
    Transcribes audio offline using Whisper model with CUDA execution.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file '{audio_path}' not found.")

    try:
        import torch
        import whisper
    except ImportError:
        print("[!] Dependencies missing. Install with: pip install torch openai-whisper")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Whisper ({model_size}) on device: {device}")
    if device == "cuda":
        print(f"[*] GPU Model: {torch.cuda.get_device_name(0)}")

    start_time = time.time()
    model = whisper.load_model(model_size, device=device)
    print(f"[*] Model loaded in {round(time.time() - start_time, 2)} seconds. Transcribing...")

    result = model.transcribe(
        audio_path,
        language=language,
        temperature=0.0,
        verbose=False
    )

    elapsed = round(time.time() - start_time, 2)
    print(f"[✓] Transcription finished in {elapsed}s.")

    base_name = os.path.splitext(audio_path)[0]
    out_file = f"{base_name}_transcript.{output_format}"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())

    print(f"[✓] Saved transcript to: {out_file}")
    return result["text"]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        transcribe_audio(sys.argv[1])
    else:
        print("Usage: python transcribe.py <path_to_audio>")
