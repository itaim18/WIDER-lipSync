"""
Lip-sync accuracy via SyncNet.
Primary: invoke LatentSync's bundled SyncNet evaluation via subprocess.
Fallback: audio-visual cross-correlation.
"""
import os
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

LATENTSYNC_DIR = "/home/itaim/wider-itai/LatentSync"
PYTHON = "/home/itaim/miniconda3/envs/latentsync/bin/python"


def compute_sync_metrics(video_path):
    """
    Returns: {"sync_conf": float, "av_offset": int}
    """
    try:
        return _run_latentsync_syncnet(video_path)
    except Exception as e:
        print(f"  SyncNet failed: {e}")
        try:
            print("  Trying fallback AV correlation...")
            return _fallback_av_correlation(video_path)
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            return {"sync_conf": None, "av_offset": None}


def _run_latentsync_syncnet(video_path):
    abs_video = os.path.abspath(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    detect_dir = f"/tmp/syncnet_detect_{video_name}"
    temp_dir = f"/tmp/syncnet_eval_{video_name}"

    for d in [detect_dir, temp_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

    eval_script = f'''
import sys, os, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "{LATENTSYNC_DIR}")
os.chdir("{LATENTSYNC_DIR}")

from eval.syncnet import SyncNetEval
from eval.syncnet_detect import SyncNetDetector

device = "cuda" if torch.cuda.is_available() else "cpu"

syncnet = SyncNetEval(device=device)
syncnet.loadParameters("checkpoints/auxiliary/syncnet_v2.model")

syncnet_detector = SyncNetDetector(device=device, detect_results_dir="{detect_dir}")
syncnet_detector(video_path="{abs_video}", min_track=25)

crop_dir = os.path.join("{detect_dir}", "crop")
if not os.path.exists(crop_dir) or not os.listdir(crop_dir):
    raise Exception("No face tracks detected")

crop_videos = sorted(os.listdir(crop_dir))
av_offsets = []
confs = []
for v in crop_videos:
    if not v.endswith(".avi"):
        continue
    av_offset, min_dist, conf = syncnet.evaluate(
        video_path=os.path.join(crop_dir, v),
        temp_dir="{temp_dir}"
    )
    av_offsets.append(av_offset)
    confs.append(conf)

if not confs:
    raise Exception("No valid sync evaluations")

from statistics import fmean
print(f"SYNCNET_RESULT: av_offset={{int(fmean(av_offsets))}}, conf={{fmean(confs):.6f}}")
'''

    result = subprocess.run(
        [PYTHON, "-c", eval_script],
        capture_output=True, text=True, timeout=300,
        cwd=LATENTSYNC_DIR,
    )

    output = result.stdout + result.stderr
    match = re.search(r"SYNCNET_RESULT: av_offset=(-?\d+), conf=([\d.e+-]+)", output)
    if match:
        return {
            "av_offset": int(match.group(1)),
            "sync_conf": float(match.group(2)),
        }
    else:
        raise RuntimeError(
            f"SyncNet parsing failed.\nstdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
        )


def _fallback_av_correlation(video_path):
    """
    Simplified AV sync: cross-correlate audio RMS energy with mouth-region
    pixel variance to estimate offset and sync quality.
    """
    from scipy.io import wavfile
    from scipy.signal import correlate

    # Extract audio to WAV
    tmp_wav = f"/tmp/av_sync_{os.getpid()}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "warning", "-i", video_path,
         "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp_wav],
        check=True, capture_output=True,
    )

    sr, audio = wavfile.read(tmp_wav)
    os.remove(tmp_wav)
    audio = audio.astype(np.float32)

    # Get video frames and compute mouth-region variance
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    variances = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        # Lower-center region as mouth proxy
        mouth = frame[int(h * 0.55):int(h * 0.75), int(w * 0.3):int(w * 0.7)]
        gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
        variances.append(np.var(gray.astype(np.float32)))
    cap.release()

    n_frames = len(variances)
    if n_frames < 10:
        return {"sync_conf": None, "av_offset": None}

    # Compute audio RMS energy per video frame window
    samples_per_frame = int(sr / fps)
    audio_energy = []
    for i in range(n_frames):
        start = i * samples_per_frame
        end = start + samples_per_frame
        if end > len(audio):
            break
        chunk = audio[start:end]
        audio_energy.append(np.sqrt(np.mean(chunk ** 2)))

    min_len = min(len(variances), len(audio_energy))
    v = np.array(variances[:min_len])
    a = np.array(audio_energy[:min_len])

    # Normalize
    v = (v - v.mean()) / (v.std() + 1e-8)
    a = (a - a.mean()) / (a.std() + 1e-8)

    # Cross-correlate
    corr = correlate(v, a, mode="full")
    lags = np.arange(-min_len + 1, min_len)
    max_lag_range = int(fps * 0.5)  # search within +/- 0.5 sec
    center = min_len - 1
    search = corr[center - max_lag_range:center + max_lag_range + 1]
    search_lags = lags[center - max_lag_range:center + max_lag_range + 1]

    best_idx = np.argmax(search)
    av_offset = int(search_lags[best_idx])
    conf = float(search[best_idx] / min_len)

    return {"av_offset": av_offset, "sync_conf": conf}
