"""Video and audio feature extraction for TikTok-style edits."""

from __future__ import annotations

import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

PROCESS_HEIGHT = 240
FRAME_SKIP = 2
HIST_CUT_THRESHOLD = 0.35
MOTION_SAMPLE_EVERY = 2
AUDIO_FRAME_MS = 100
SPIKE_WINDOW = 5
AV_SYNC_TOLERANCE_S = 0.15


@dataclass
class AnalysisResult:
    duration: float
    cuts_count: int = 0
    cuts_per_second: float = 0.0
    avg_scene_duration: float = 0.0
    motion_intensity: float = 0.0
    visual_change_rate: float = 0.0
    visual_stability: float = 0.0
    hook_speed: float = 0.0
    audio_energy: float = 0.0
    audio_spikes_count: int = 0
    audio_pacing: float = 0.0
    av_sync_score: float = 0.0
    hook_motion_3s: float = 0.0
    motion_first_2s: float = 0.0
    cuts_first_3s: int = 0
    longest_scene_gap: float = 0.0
    shake_intensity: float = 0.0
    color_variance: float = 0.0
    loop_potential_score: float = 0.0
    audio_dynamic_range: float = 0.0
    silence_ratio: float = 0.0
    early_hold_score: float = 0.0
    retention_proxy: float = 0.0
    timeline: dict = field(default_factory=dict)

    def to_features(self) -> dict:
        return {
            "cuts_count": self.cuts_count,
            "cuts_per_second": self.cuts_per_second,
            "avg_scene_duration": self.avg_scene_duration,
            "motion_intensity": self.motion_intensity,
            "visual_change_rate": self.visual_change_rate,
            "visual_stability": self.visual_stability,
            "hook_speed": self.hook_speed,
            "audio_energy": self.audio_energy,
            "audio_spikes_count": self.audio_spikes_count,
            "audio_pacing": self.audio_pacing,
            "av_sync_score": self.av_sync_score,
            "hook_motion_3s": self.hook_motion_3s,
            "motion_first_2s": self.motion_first_2s,
            "cuts_first_3s": self.cuts_first_3s,
            "longest_scene_gap": self.longest_scene_gap,
            "shake_intensity": self.shake_intensity,
            "color_variance": self.color_variance,
            "loop_potential_score": self.loop_potential_score,
            "audio_dynamic_range": self.audio_dynamic_range,
            "silence_ratio": self.silence_ratio,
            "early_hold_score": self.early_hold_score,
            "retention_proxy": self.retention_proxy,
            "timeline": self.timeline,
        }


def analyze_video(video_path: str | Path) -> AnalysisResult:
    video_path = Path(video_path)
    result = AnalysisResult(duration=0.0)

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        has_audio = _extract_audio(video_path, wav_path)
        audio_data = (
            _analyze_audio(wav_path, result)
            if has_audio
            else _empty_audio(result)
        )
        _analyze_video_frames(video_path, result, audio_data)

    _compute_derived_scores(result)
    return result


def _compute_derived_scores(result: AnalysisResult) -> None:
    hook_penalty = min(1.0, result.hook_speed / 2.0)
    early_motion = min(1.0, result.motion_first_2s * 5)
    result.early_hold_score = round(
        max(0.0, min(1.0, 0.6 * (1 - hook_penalty) + 0.4 * early_motion)), 3
    )
    loop_len = min(1.0, 15.0 / max(result.duration, 1.0))
    rhythm = min(1.0, result.cuts_per_second / 2.5)
    result.loop_potential_score = round(
        max(0.0, min(1.0, 0.5 * loop_len + 0.3 * rhythm + 0.2 * result.av_sync_score)),
        3,
    )
    gap_penalty = min(1.0, result.longest_scene_gap / 5.0)
    result.retention_proxy = round(
        max(
            0.0,
            min(
                1.0,
                0.35 * result.early_hold_score
                + 0.25 * result.motion_intensity * 4
                + 0.2 * result.av_sync_score
                + 0.2 * (1 - gap_penalty),
            ),
        ),
        3,
    )


def _extract_audio(video_path: Path, wav_path: Path) -> bool:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "22050",
        "-ac",
        "1",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    return proc.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 0


def _empty_audio(result: AnalysisResult) -> dict:
    result.timeline["audio_energy"] = []
    result.timeline["audio_spike_times"] = []
    return {"spike_times": [], "energy_curve": [], "sample_rate": 22050}


def _analyze_audio(wav_path: Path, result: AnalysisResult) -> dict:
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        return _empty_audio(result)

    sample_rate, audio = _read_wav_mono(wav_path)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    frame_samples = int(sample_rate * AUDIO_FRAME_MS / 1000)
    n_frames = max(1, len(audio) // frame_samples)
    energies = []
    for i in range(n_frames):
        chunk = audio[i * frame_samples : (i + 1) * frame_samples]
        rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) else 0.0
        energies.append(rms)

    energies_arr = np.array(energies)
    result.audio_energy = float(np.mean(energies_arr)) if len(energies_arr) else 0.0
    if len(energies_arr):
        result.audio_dynamic_range = float(np.max(energies_arr) - np.min(energies_arr))
        result.silence_ratio = float(np.mean(energies_arr < 0.03))

    spike_times: list[float] = []
    if len(energies_arr) >= SPIKE_WINDOW:
        kernel = np.ones(SPIKE_WINDOW) / SPIKE_WINDOW
        rolling = np.convolve(energies_arr, kernel, mode="same")
        threshold = rolling * 1.4 + 0.02
        for i, e in enumerate(energies_arr):
            if e > threshold[i] and e > 0.05:
                t = (i * frame_samples) / sample_rate
                if not spike_times or t - spike_times[-1] > 0.08:
                    spike_times.append(t)

    result.audio_spikes_count = len(spike_times)
    duration_audio = len(audio) / sample_rate
    result.audio_pacing = (
        len(spike_times) / duration_audio if duration_audio > 0 else 0.0
    )

    time_axis = [
        (i * frame_samples) / sample_rate for i in range(len(energies_arr))
    ]
    result.timeline["audio_energy"] = [
        {"t": round(t, 3), "v": round(float(v), 4)}
        for t, v in zip(time_axis, energies_arr)
    ]
    result.timeline["audio_spike_times"] = [round(t, 3) for t in spike_times]

    return {
        "spike_times": spike_times,
        "energy_curve": energies_arr.tolist(),
        "sample_rate": sample_rate,
    }


def _read_wav_mono(wav_path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(wav_path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    dtype = np.int16 if sample_width == 2 else np.uint8
    audio = np.frombuffer(frames, dtype=dtype)
    if sample_width == 1:
        audio = audio.astype(np.float64) - 128.0
    else:
        audio = audio.astype(np.float64)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return sample_rate, audio


def _resize_frame(frame: np.ndarray, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h <= target_h:
        return frame
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _hist_diff(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    hsv_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return 1.0 - float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def _analyze_video_frames(
    video_path: Path,
    result: AnalysisResult,
    audio_data: dict,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    result.duration = total_frames / fps if total_frames > 0 else 0.0

    cut_times: list[float] = []
    motion_series: list[dict] = []
    hist_diffs: list[float] = []
    color_vars: list[float] = []
    shake_samples: list[float] = []
    prev_gray: np.ndarray | None = None
    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        small = _resize_frame(frame, PROCESS_HEIGHT)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        t = frame_idx / fps

        color_vars.append(float(np.std(small)))

        if prev_gray is not None:
            diff = _hist_diff(
                cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR),
                small,
            )
            hist_diffs.append(diff)
            if diff >= HIST_CUT_THRESHOLD:
                if not cut_times or t - cut_times[-1] > 0.12:
                    cut_times.append(t)

            if processed % MOTION_SAMPLE_EVERY == 0:
                motion = float(
                    np.mean(cv2.absdiff(prev_gray, gray)) / 255.0
                )
                motion_series.append({"t": round(t, 3), "v": round(motion, 4)})
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                shake_samples.append(float(np.var(lap)) / 10000.0)

        prev_gray = gray
        frame_idx += 1
        processed += 1

    cap.release()

    result.cuts_count = len(cut_times)
    result.cuts_first_3s = sum(1 for t in cut_times if t <= 3.0)
    if result.duration > 0:
        result.cuts_per_second = result.cuts_count / result.duration
        result.avg_scene_duration = (
            result.duration / max(1, result.cuts_count + 1)
        )

    gaps = [cut_times[i + 1] - cut_times[i] for i in range(len(cut_times) - 1)]
    if gaps:
        result.longest_scene_gap = float(max(gaps))
    elif result.duration > 0:
        result.longest_scene_gap = result.duration

    if hist_diffs:
        result.visual_change_rate = float(np.mean(hist_diffs))

    motions = [m["v"] for m in motion_series]
    if motions:
        result.motion_intensity = float(np.mean(motions))
        result.visual_stability = float(1.0 - min(1.0, np.std(motions) * 4))
        early = [m["v"] for m in motion_series if m["t"] <= 3.0]
        early2 = [m["v"] for m in motion_series if m["t"] <= 2.0]
        if early:
            result.hook_motion_3s = float(np.mean(early))
        if early2:
            result.motion_first_2s = float(np.mean(early2))

    if shake_samples:
        result.shake_intensity = float(np.mean(shake_samples))
    if color_vars:
        result.color_variance = float(np.mean(color_vars)) / 128.0

    result.hook_speed = _compute_hook_speed(cut_times, motion_series)
    result.av_sync_score = _compute_av_sync(cut_times, audio_data.get("spike_times", []))

    result.timeline["cut_times"] = [round(t, 3) for t in cut_times]
    result.timeline["motion"] = motion_series


def _compute_hook_speed(
    cut_times: list[float],
    motion_series: list[dict],
) -> float:
    if cut_times:
        return cut_times[0]
    if motion_series:
        threshold = np.percentile([m["v"] for m in motion_series], 75)
        for m in motion_series:
            if m["v"] >= threshold:
                return m["t"]
    return 3.0


def _compute_av_sync(cut_times: list[float], spike_times: list[float]) -> float:
    if not cut_times or not spike_times:
        return 0.5
    aligned = 0
    for ct in cut_times:
        for st in spike_times:
            if abs(ct - st) <= AV_SYNC_TOLERANCE_S:
                aligned += 1
                break
    return min(1.0, aligned / len(cut_times))
