#!/usr/bin/env python3
"""
Unit and Integration Test Suite for Local Whisper STT Pipeline.
Validates timestamp formatting, SRT serialization, and dependency integrity.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from transcribe import format_timestamp, write_srt

class TestWhisperPipeline(unittest.TestCase):

    def test_format_timestamp_zero(self):
        self.assertEqual(format_timestamp(0.0, always_include_hours=True), "00:00:00,000")

    def test_format_timestamp_standard(self):
        self.assertEqual(format_timestamp(65.42, always_include_hours=True), "00:01:05,420")

    def test_format_timestamp_hours(self):
        self.assertEqual(format_timestamp(3661.125, always_include_hours=True), "01:01:01,125")

    def test_srt_generation(self):
        dummy_segments = [
            {"start": 0.0, "end": 2.5, "text": "Halo dunia."},
            {"start": 2.5, "end": 5.8, "text": "Pengujian transkripsi otomatis."}
        ]
        test_file = "/tmp/test_sample.srt"
        write_srt(dummy_segments, test_file)
        
        self.assertTrue(os.path.exists(test_file))
        with open(test_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("00:00:00,000 --> 00:00:02,500", content)
        self.assertIn("Halo dunia.", content)
        self.assertIn("00:00:02,500 --> 00:00:05,800", content)
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    unittest.main()
