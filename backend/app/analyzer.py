import os
import cv2
import numpy as np
import subprocess
import tempfile
import scipy.io.wavfile as wavfile
import imageio_ffmpeg as gp

def extract_audio_features(video_path: str) -> dict:
    """
    Extracts audio from video to a temporary WAV file and computes energy and spikes (beats).
    """
    temp_wav = tempfile.mktemp(suffix=".wav")
    ffmpeg_exe = gp.get_ffmpeg_exe()
    
    # Command to extract audio: 16kHz mono, 16-bit PCM wav
    cmd = [
        ffmpeg_exe, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        temp_wav
    ]
    
    try:
        # Run subprocess silently
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        # If extraction fails (e.g. no audio stream), clean up and return defaults
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except:
                pass
        return {
            "audio_energy": 0.0,
            "audio_spikes_count": 0,
            "audio_pacing": 0.0,
            "spike_timestamps": []
        }

    if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
        return {
            "audio_energy": 0.0,
            "audio_spikes_count": 0,
            "audio_pacing": 0.0,
            "spike_timestamps": []
        }

    try:
        sample_rate, data = wavfile.read(temp_wav)
    except Exception:
        return {
            "audio_energy": 0.0,
            "audio_spikes_count": 0,
            "audio_pacing": 0.0,
            "spike_timestamps": []
        }
    finally:
        # Clean up temporary WAV file
        try:
            os.remove(temp_wav)
        except Exception:
            pass

    # Convert to float and normalize
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    duration = len(data) / sample_rate
    if duration == 0:
        return {
            "audio_energy": 0.0,
            "audio_spikes_count": 0,
            "audio_pacing": 0.0,
            "spike_timestamps": []
        }

    # Audio energy in 100ms frames (10 frames per second)
    frame_size = int(sample_rate * 0.1)
    if frame_size == 0 or len(data) < frame_size:
        return {
            "audio_energy": 0.0,
            "audio_spikes_count": 0,
            "audio_pacing": 0.0,
            "spike_timestamps": []
        }

    num_frames = len(data) // frame_size
    rms_energies = []
    for i in range(num_frames):
        frame = data[i * frame_size : (i + 1) * frame_size]
        rms = np.sqrt(np.mean(frame**2) + 1e-8)
        rms_energies.append(rms)

    rms_energies = np.array(rms_energies)
    mean_energy = float(np.mean(rms_energies))

    # Detect spikes (audio beats) using a simple rolling threshold
    # An audio frame is a beat if it is a local maximum and exceeds 1.6x the rolling average
    spikes = []
    spike_timestamps = []
    rolling_window = 15  # 1.5 seconds rolling window for thresholding
    
    for i in range(1, len(rms_energies) - 1):
        val = rms_energies[i]
        start_idx = max(0, i - rolling_window // 2)
        end_idx = min(len(rms_energies), i + rolling_window // 2 + 1)
        local_avg = np.mean(rms_energies[start_idx:end_idx])
        
        # Threshold: minimum amplitude of 0.03, and at least 1.6x the local mean
        if val > 0.03 and val > 1.6 * local_avg:
            if val > rms_energies[i - 1] and val > rms_energies[i + 1]:
                spikes.append(i)
                spike_timestamps.append(float(i * 0.1))

    audio_pacing = len(spikes) / duration if duration > 0 else 0.0

    return {
        "audio_energy": mean_energy,
        "audio_spikes_count": len(spikes),
        "audio_pacing": audio_pacing,
        "spike_timestamps": spike_timestamps
    }

def analyze_video_file(video_path: str) -> dict:
    """
    Reads the video file and extracts cuts, pacing, motion intensity, visual stability,
    visual change rate, hook speed, and AV sync (incorporating audio spikes).
    """
    # 1. Extract audio features first
    audio_data = extract_audio_features(video_path)
    audio_spikes = audio_data["spike_timestamps"]

    # 2. Open video stream
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open MP4 video file. The file might be corrupted or in an unsupported format.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        fps = 30.0
    duration = frame_count / fps

    prev_gray = None
    diffs = []
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Downscale frame for speed: 120x90 is sufficient for visual differences
        small = cv2.resize(frame, (120, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            # Frame difference
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = float(np.mean(diff))
            diffs.append(mean_diff)
        else:
            diffs.append(0.0)
            
        prev_gray = gray
        
    cap.release()

    if not diffs:
        raise ValueError("Video has no frames to analyze.")

    diffs = np.array(diffs)

    # 3. Detect cuts (hard cuts & rapid transitions) using dynamic thresholding
    # A cut is a peak in difference that is significantly larger than the surrounding local average.
    rolling_window = int(fps * 1.5)  # 1.5s rolling average
    if rolling_window % 2 == 0:
        rolling_window += 1

    cuts = []
    local_means = []
    
    for i in range(len(diffs)):
        start = max(0, i - rolling_window // 2)
        end = min(len(diffs), i + rolling_window // 2 + 1)
        local_means.append(np.mean(diffs[start:end]))

    for i in range(1, len(diffs) - 1):
        val = diffs[i]
        # Pixel difference must be at least 10 (out of 255) and 2.2x the local rolling mean
        if val > 10.0 and val > 2.2 * local_means[i]:
            # Peak condition (must be greater than immediate neighbors)
            if val > diffs[i - 1] and val > diffs[i + 1]:
                cuts.append(i)

    cut_timestamps = [float(c / fps) for c in cuts]

    # 4. Motion Intensity (average frame difference, ignoring cuts)
    non_cut_indices = list(set(range(len(diffs))) - set(cuts))
    if non_cut_indices:
        motion_intensity = float(np.mean(diffs[non_cut_indices]))
        motion_std = float(np.std(diffs[non_cut_indices]))
    else:
        motion_intensity = float(np.mean(diffs))
        motion_std = 0.0

    # 5. Visual Stability: higher means more consistent pacing/movement (low sudden erratic spikes)
    if motion_intensity > 0:
        stability_ratio = motion_std / motion_intensity
        # Higher variation means lower stability
        visual_stability = max(0.0, min(100.0, 100.0 * (1.0 / (1.0 + 0.5 * stability_ratio))))
    else:
        visual_stability = 100.0

    # 6. Hook Speed: time of the first cut or major motion peak (visual change)
    # We look for the first cut or frame where the difference is > 2.5x the overall average difference
    hook_speed = duration
    avg_diff = np.mean(diffs)
    for i in range(len(diffs)):
        if diffs[i] > max(15.0, 2.5 * avg_diff):
            hook_speed = i / fps
            break
            
    if cut_timestamps:
        hook_speed = min(hook_speed, cut_timestamps[0])

    # 7. Visual Change Rate: percentage of frames with visual change exceeding 1.3x average difference
    change_frames = np.sum(diffs > (1.3 * avg_diff))
    visual_change_rate = float(change_frames / len(diffs)) if len(diffs) > 0 else 0.0

    # 8. Audio-Visual Sync Score (AV-Sync)
    # The percentage of video cuts that land within 200ms of an audio beat spike
    aligned_cuts = 0
    if cut_timestamps and audio_spikes:
        for cut_t in cut_timestamps:
            for spike_t in audio_spikes:
                if abs(cut_t - spike_t) <= 0.20:
                    aligned_cuts += 1
                    break
        av_sync_score = (aligned_cuts / len(cut_timestamps)) * 100.0
    else:
        av_sync_score = 0.0

    cuts_count = len(cuts)
    cuts_per_second = cuts_count / duration if duration > 0 else 0.0
    avg_scene_duration = duration / (cuts_count + 1) if duration > 0 else 0.0

    return {
        "duration": duration,
        "cuts_count": cuts_count,
        "cuts_per_second": cuts_per_second,
        "avg_scene_duration": avg_scene_duration,
        "motion_intensity": motion_intensity,
        "visual_change_rate": visual_change_rate,
        "visual_stability": visual_stability,
        "hook_speed": hook_speed,
        "audio_energy": audio_data["audio_energy"],
        "audio_spikes_count": audio_data["audio_spikes_count"],
        "audio_pacing": audio_data["audio_pacing"],
        "av_sync_score": av_sync_score
    }
