#!/usr/bin/env python3
"""CLI verification: generate a test clip and run feature extraction."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.analyzer import analyze_video
from app.scorer import score_features


def generate_test_video(path: Path, duration_s: float = 3.0, fps: int = 30) -> None:
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    n = int(duration_s * fps)
    for i in range(n):
        hue = int((i / max(1, n - 1)) * 180)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (hue, 200, 220)
        if i % 8 == 0 and i > 0:
            frame = 255 - frame
        writer.write(frame)
    writer.release()


def main() -> int:
    print("EditIQ verification")
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "test_edit.mp4"
        print("  Generating synthetic test video…")
        generate_test_video(video)

        print("  Running analyzer…")
        result = analyze_video(video)
        features = result.to_features()
        scores = score_features(features)

    print(f"  Duration: {result.duration:.2f}s")
    print(f"  Cuts: {result.cuts_count} ({result.cuts_per_second:.2f}/s)")
    print(f"  Motion: {result.motion_intensity:.4f}")
    print(f"  Viral score: {scores['viral_score']}")
    print("  OK — feature extraction completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
