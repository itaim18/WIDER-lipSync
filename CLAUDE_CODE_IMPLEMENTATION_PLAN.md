# Claude Code Implementation Plan: Talking-Head Lip-Sync Evaluation Pipeline

## Overview

This plan instructs Claude Code to:
1. Organize all output videos into a structured directory with descriptive names
2. Run a full evaluation suite (lip-sync, temporal stability, visual fidelity, identity preservation)
3. Generate a LaTeX semi-paper with per-video results, comparison tables, figures, and conclusions

---

## Phase 1: Video Organization

### 1.1 Create Directory Structure

```
eval_workspace/
├── videos/                          # All videos go here
│   ├── originals/                   # Original input videos (ground truth / pre-fix)
│   └── outputs/                     # Generated / fixed videos
├── results/                         # Raw metric outputs (JSON + CSV)
├── crops/                           # Extracted face/mouth crops & sample frames
├── figures/                         # Generated plots for the paper
├── paper/                           # LaTeX source
│   ├── main.tex
│   ├── figures/                     # Symlink or copy of figures/
│   └── references.bib
└── scripts/                         # All Python evaluation scripts
```

### 1.2 Video Naming Convention

Rename every video using this template:

```
{model}_{training_detail}_{steps}k_{lr}_{resolution}_{extra}.mp4
```

Examples:
- `wav2lip_gan_50k_1e4_256px_nofinetune.mp4`
- `videoretalking_base_100k_2e4_512px_enhancer.mp4`
- `dinet_custom_30k_5e5_256px_syncaug.mp4`
- `original_ground_truth_NA_NA_512px_reference.mp4`

**Rules for Claude Code:**
- Scan all video files the user provides (check `/mnt/user-data/uploads/` and any paths the user specifies).
- Ask the user (or read from a manifest/metadata file if one exists) for: model name, training variant, steps, learning rate, resolution, and any extra tags.
- If metadata is not available, use the filename or parent folder name to infer as much as possible, and tag unknowns as `unk`.
- Copy (not move) all videos into `eval_workspace/videos/outputs/` with the standardized name.
- Place original/reference videos into `eval_workspace/videos/originals/`.
- Write a `manifest.json` mapping each standardized filename to its original path and all known metadata fields.

---

## Phase 2: Evaluation Scripts

### 2.1 Dependencies to Install

```bash
pip install --break-system-packages \
    opencv-python-headless \
    scikit-image \
    lpips \
    torch torchvision \
    insightface \
    onnxruntime \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    tqdm
```

> **Note on SyncNet / LSE-C / LSE-D:** These require cloning the SyncNet repo or a compatible implementation. Claude Code should clone `https://github.com/joonson/syncnet_python` into `eval_workspace/tools/syncnet/` and follow its setup. If network restrictions prevent cloning, implement a simplified audio-video offset estimation using cross-correlation of mouth-area motion energy with audio energy as a fallback, and note the limitation.

### 2.2 Core Evaluation Script: `scripts/evaluate.py`

Claude Code should create this script with the following structure:

```python
"""
Main evaluation script.
Usage: python evaluate.py --originals videos/originals/ --outputs videos/outputs/ --results results/
"""

import argparse, json, os, csv
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Import metric modules (each defined below)
from metrics.sync import compute_sync_metrics        # LSE-C, LSE-D, AV offset
from metrics.identity import compute_identity_sim     # ArcFace cosine similarity
from metrics.temporal import compute_temporal_flicker # Mouth-region temporal stability
from metrics.fidelity import compute_fidelity         # SSIM, LPIPS on face/mouth crops
from utils.face_crop import extract_face_mouth_crops  # Face detection + cropping
from utils.frames import extract_frames               # Video → frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--originals", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--results", default="results/")
    parser.add_argument("--sample_rate", type=float, default=1.0,
                        help="Sample 1 frame every N seconds for identity/fidelity")
    parser.add_argument("--all_frames", action="store_true",
                        help="Use all frames for temporal metrics (default: True for temporal)")
    args = parser.parse_args()

    os.makedirs(args.results, exist_ok=True)

    # Build pairs: for each output video, find its corresponding original
    # Pairing logic: match by the content after last underscore or by manifest.json
    pairs = build_pairs(args.originals, args.outputs)

    all_results = []

    for output_video, original_video in tqdm(pairs, desc="Evaluating"):
        video_name = Path(output_video).stem
        print(f"\n{'='*60}")
        print(f"Evaluating: {video_name}")
        print(f"{'='*60}")

        result = {"video": video_name}

        # --- Step 1: Extract frames ---
        out_frames = extract_frames(output_video)
        orig_frames = extract_frames(original_video) if original_video else None

        # --- Step 2: Extract face/mouth crops ---
        out_crops = extract_face_mouth_crops(out_frames)
        orig_crops = extract_face_mouth_crops(orig_frames) if orig_frames is not None else None

        # Save sample crops for the paper
        save_sample_crops(out_crops, orig_crops, video_name, args.results)

        # --- Step 3: Lip-sync metrics ---
        sync = compute_sync_metrics(output_video)
        result.update(sync)

        # --- Step 4: Identity preservation ---
        identity = compute_identity_sim(
            orig_frames, out_frames, sample_every_n_sec=args.sample_rate
        )
        result.update(identity)

        # --- Step 5: Temporal flicker ---
        temporal = compute_temporal_flicker(out_crops["mouth_crops"])
        result.update(temporal)

        # --- Step 6: Visual fidelity (if original exists) ---
        if orig_crops is not None:
            fidelity = compute_fidelity(orig_crops, out_crops)
            result.update(fidelity)

        all_results.append(result)

    # --- Save results ---
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.results, "evaluation_results.csv"), index=False)
    with open(os.path.join(args.results, "evaluation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n\nFinal Results:")
    print(df.to_string(index=False))

    return df
```

### 2.3 Metric Modules

Claude Code should create each of these as separate files under `scripts/metrics/`:

#### `metrics/sync.py` — Lip-Sync Accuracy

```python
"""
Compute LSE-C, LSE-D, and AV offset.
Primary approach: use SyncNet (joonson/syncnet_python).
Fallback: audio-visual cross-correlation of mouth motion energy vs audio energy.
"""

def compute_sync_metrics(video_path: str) -> dict:
    """
    Returns:
        {
            "lse_c": float,        # higher is better
            "lse_d": float,        # lower is better
            "av_offset": float,    # frames, closer to 0 is better
        }
    """
    # Try SyncNet first
    try:
        return _syncnet_metrics(video_path)
    except Exception as e:
        print(f"SyncNet failed ({e}), using fallback AV correlation")
        return _fallback_av_correlation(video_path)


def _syncnet_metrics(video_path):
    """
    Clone and use https://github.com/joonson/syncnet_python
    Run its pipeline:
      1. Extract audio + video tracks
      2. Run SyncNet inference
      3. Parse LSE-C, LSE-D, offset from output
    """
    # Implementation: subprocess call to syncnet pipeline
    # Parse stdout/files for confidence and distance
    raise NotImplementedError("Claude Code: implement SyncNet wrapper here")


def _fallback_av_correlation(video_path):
    """
    Simplified AV sync estimation:
    1. Extract audio RMS energy per frame-window
    2. Extract mouth-region pixel variance per frame
    3. Cross-correlate to find optimal offset
    4. Report offset and correlation strength as proxy for LSE-C
    """
    # Implementation with cv2 + librosa/scipy
    raise NotImplementedError("Claude Code: implement fallback here")
```

#### `metrics/identity.py` — Identity Preservation

```python
"""
Face embedding cosine similarity using InsightFace/ArcFace.
"""
import numpy as np

def compute_identity_sim(orig_frames, out_frames, sample_every_n_sec=1.0, fps=25) -> dict:
    """
    Sample frames at the given rate.
    Compute ArcFace embeddings for original and output.
    Return mean and worst-10th-percentile cosine similarity.

    Returns:
        {
            "identity_cos_mean": float,      # higher is better, max 1.0
            "identity_cos_p10": float,        # worst 10th percentile
            "identity_cos_std": float,
            "identity_n_frames": int,
        }
    """
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    step = max(1, int(fps * sample_every_n_sec))
    similarities = []

    for i in range(0, min(len(orig_frames), len(out_frames)), step):
        orig_faces = app.get(orig_frames[i])
        out_faces = app.get(out_frames[i])

        if orig_faces and out_faces:
            emb_orig = orig_faces[0].embedding
            emb_out = out_faces[0].embedding
            cos_sim = np.dot(emb_orig, emb_out) / (
                np.linalg.norm(emb_orig) * np.linalg.norm(emb_out)
            )
            similarities.append(cos_sim)

    sims = np.array(similarities)
    return {
        "identity_cos_mean": float(np.mean(sims)) if len(sims) > 0 else None,
        "identity_cos_p10": float(np.percentile(sims, 10)) if len(sims) > 0 else None,
        "identity_cos_std": float(np.std(sims)) if len(sims) > 0 else None,
        "identity_n_frames": len(sims),
    }
```

#### `metrics/temporal.py` — Temporal Smoothness / Flicker

```python
"""
Temporal stability of the mouth region.
Measures frame-to-frame consistency via:
  - Mean absolute pixel difference
  - Optical flow magnitude stats
  - Consecutive-frame LPIPS (optional, expensive)
"""
import cv2
import numpy as np

def compute_temporal_flicker(mouth_crops: list) -> dict:
    """
    Input: list of mouth-region crops (numpy arrays, BGR) for consecutive frames.

    Returns:
        {
            "temporal_mad_mean": float,     # mean absolute difference, lower=smoother
            "temporal_mad_std": float,
            "temporal_mad_p95": float,      # 95th percentile spikes
            "temporal_flow_mean": float,    # optical flow magnitude mean
            "temporal_flow_std": float,
            "temporal_flow_p95": float,
        }
    """
    if len(mouth_crops) < 2:
        return {k: None for k in [
            "temporal_mad_mean", "temporal_mad_std", "temporal_mad_p95",
            "temporal_flow_mean", "temporal_flow_std", "temporal_flow_p95",
        ]}

    mads = []
    flow_mags = []

    for i in range(len(mouth_crops) - 1):
        curr = mouth_crops[i].astype(np.float32)
        nxt = mouth_crops[i + 1].astype(np.float32)

        # Mean absolute difference
        mad = np.mean(np.abs(curr - nxt))
        mads.append(mad)

        # Optical flow
        gray_curr = cv2.cvtColor(mouth_crops[i], cv2.COLOR_BGR2GRAY)
        gray_nxt = cv2.cvtColor(mouth_crops[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_curr, gray_nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_mags.append(np.mean(mag))

    mads = np.array(mads)
    flow_mags = np.array(flow_mags)

    return {
        "temporal_mad_mean": float(np.mean(mads)),
        "temporal_mad_std": float(np.std(mads)),
        "temporal_mad_p95": float(np.percentile(mads, 95)),
        "temporal_flow_mean": float(np.mean(flow_mags)),
        "temporal_flow_std": float(np.std(flow_mags)),
        "temporal_flow_p95": float(np.percentile(flow_mags, 95)),
    }
```

#### `metrics/fidelity.py` — Visual Fidelity (SSIM, LPIPS)

```python
"""
Visual fidelity metrics: SSIM and LPIPS on face and mouth crops.
These measure preservation relative to the original, NOT sync quality.
"""
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
import lpips

_lpips_model = None

def _get_lpips():
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex")
    return _lpips_model

def compute_fidelity(orig_crops: dict, out_crops: dict) -> dict:
    """
    orig_crops / out_crops each have keys: "face_crops", "mouth_crops"
    Each is a list of numpy arrays (BGR).

    Returns:
        {
            "face_ssim_mean": float,
            "face_lpips_mean": float,
            "mouth_ssim_mean": float,
            "mouth_lpips_mean": float,
        }
    """
    results = {}
    for region in ["face", "mouth"]:
        orig_list = orig_crops[f"{region}_crops"]
        out_list = out_crops[f"{region}_crops"]
        n = min(len(orig_list), len(out_list))

        ssim_vals = []
        lpips_vals = []

        for i in range(0, n, max(1, n // 50)):  # sample up to ~50 frames
            o = orig_list[i]
            g = out_list[i]

            # Resize to match if needed
            if o.shape != g.shape:
                g = cv2.resize(g, (o.shape[1], o.shape[0]))

            # SSIM (on grayscale)
            o_gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
            g_gray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
            s = ssim(o_gray, g_gray)
            ssim_vals.append(s)

            # LPIPS
            o_t = _to_tensor(o)
            g_t = _to_tensor(g)
            with torch.no_grad():
                lp = _get_lpips()(o_t, g_t).item()
            lpips_vals.append(lp)

        results[f"{region}_ssim_mean"] = float(np.mean(ssim_vals)) if ssim_vals else None
        results[f"{region}_lpips_mean"] = float(np.mean(lpips_vals)) if lpips_vals else None

    return results


def _to_tensor(img_bgr):
    """Convert BGR uint8 image to LPIPS-compatible tensor [-1, 1]."""
    import cv2
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    t = t * 2 - 1  # scale to [-1, 1]
    return t.unsqueeze(0)
```

### 2.4 Utility Modules

#### `utils/face_crop.py`

```python
"""
Face and mouth region detection and cropping.
Uses OpenCV's DNN face detector + dlib landmarks, or InsightFace.
"""
import cv2
import numpy as np

def extract_face_mouth_crops(frames, detector="insightface"):
    """
    For each frame, detect face, crop face region and mouth region.
    Returns dict with lists: {"face_crops": [...], "mouth_crops": [...]}
    """
    if frames is None:
        return None

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    face_crops = []
    mouth_crops = []

    for frame in frames:
        faces = app.get(frame)
        if not faces:
            continue

        face = faces[0]
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        # Pad face bbox by 10%
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        fx1 = max(0, x1 - pad_x)
        fy1 = max(0, y1 - pad_y)
        fx2 = min(w, x2 + pad_x)
        fy2 = min(h, y2 + pad_y)
        face_crop = frame[fy1:fy2, fx1:fx2]
        face_crops.append(face_crop)

        # Mouth region: use landmarks if available, else lower 1/3 of face
        if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
            lm = face.landmark_2d_106
            # Mouth landmarks are indices ~52-71 in 106-point model
            mouth_pts = lm[52:72]
            mx1 = int(mouth_pts[:, 0].min()) - 10
            my1 = int(mouth_pts[:, 1].min()) - 10
            mx2 = int(mouth_pts[:, 0].max()) + 10
            my2 = int(mouth_pts[:, 1].max()) + 10
        else:
            # Fallback: lower third of face
            face_h = y2 - y1
            my1 = y1 + int(face_h * 0.55)
            my2 = y2 + 5
            mx1 = x1 - 5
            mx2 = x2 + 5

        mx1 = max(0, mx1)
        my1 = max(0, my1)
        mx2 = min(w, mx2)
        my2 = min(h, my2)
        mouth_crop = frame[my1:my2, mx1:mx2]
        mouth_crops.append(mouth_crop)

    return {"face_crops": face_crops, "mouth_crops": mouth_crops}
```

#### `utils/frames.py`

```python
"""
Video frame extraction utilities.
"""
import cv2
import numpy as np

def extract_frames(video_path: str, max_frames: int = None) -> list:
    """Extract all (or up to max_frames) frames from a video as numpy arrays (BGR)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def get_video_info(video_path: str) -> dict:
    """Get FPS, frame count, resolution."""
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_sec"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info
```

### 2.5 Figure Generation Script: `scripts/generate_figures.py`

Claude Code should create this to produce plots for the paper:

```python
"""
Generate comparison plots from evaluation_results.csv.
Outputs to figures/ directory.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_all_figures(results_csv: str, output_dir: str = "figures/"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_csv)

    # Extract model name from video name (first token before underscore)
    df["model"] = df["video"].apply(lambda x: x.split("_")[0])

    # 1. Bar chart: LSE-C and LSE-D per video
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if "lse_c" in df.columns:
        df.plot.bar(x="video", y="lse_c", ax=axes[0], legend=False, color="steelblue")
        axes[0].set_title("LSE-C (higher = better sync)")
        axes[0].set_ylabel("LSE-C")
        axes[0].tick_params(axis='x', rotation=45)

        df.plot.bar(x="video", y="lse_d", ax=axes[1], legend=False, color="coral")
        axes[1].set_title("LSE-D (lower = better sync)")
        axes[1].set_ylabel("LSE-D")
        axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sync_metrics.pdf"), bbox_inches="tight")
    plt.close()

    # 2. Identity preservation scatter
    if "identity_cos_mean" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(df["video"], df["identity_cos_mean"], color="seagreen", alpha=0.8)
        ax.errorbar(df["video"], df["identity_cos_mean"],
                     yerr=df.get("identity_cos_std", 0), fmt="none", color="black", capsize=3)
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Identity Preservation (ArcFace Cosine Similarity)")
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "identity_preservation.pdf"), bbox_inches="tight")
        plt.close()

    # 3. Temporal flicker comparison
    if "temporal_mad_mean" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(df))
        ax.bar(x, df["temporal_mad_mean"], color="mediumpurple", alpha=0.8, label="Mean")
        ax.scatter(x, df["temporal_mad_p95"], color="red", zorder=5, label="95th pct spike")
        ax.set_xticks(x)
        ax.set_xticklabels(df["video"], rotation=45, ha="right")
        ax.set_ylabel("Mean Absolute Difference")
        ax.set_title("Temporal Stability — Mouth Region Flicker")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "temporal_flicker.pdf"), bbox_inches="tight")
        plt.close()

    # 4. Fidelity: SSIM + LPIPS grouped bars
    if "face_ssim_mean" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df.plot.bar(x="video", y=["face_ssim_mean", "mouth_ssim_mean"],
                    ax=axes[0], color=["steelblue", "skyblue"])
        axes[0].set_title("SSIM (higher = more preserved)")
        axes[0].set_ylabel("SSIM")
        axes[0].tick_params(axis='x', rotation=45)

        df.plot.bar(x="video", y=["face_lpips_mean", "mouth_lpips_mean"],
                    ax=axes[1], color=["coral", "lightsalmon"])
        axes[1].set_title("LPIPS (lower = more preserved)")
        axes[1].set_ylabel("LPIPS")
        axes[1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fidelity_metrics.pdf"), bbox_inches="tight")
        plt.close()

    # 5. Summary radar/heatmap
    metric_cols = [c for c in df.columns if c not in ["video", "model", "identity_n_frames"]]
    numeric_df = df[["video"] + metric_cols].set_index("video")
    if len(numeric_df.columns) > 0:
        # Normalize 0-1 for heatmap
        normed = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min() + 1e-8)
        fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.6)))
        sns.heatmap(normed, annot=numeric_df.round(3).values, fmt="",
                    cmap="YlOrRd_r", ax=ax, linewidths=0.5)
        ax.set_title("Evaluation Metrics Heatmap (raw values annotated)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metrics_heatmap.pdf"), bbox_inches="tight")
        plt.close()

    print(f"Figures saved to {output_dir}")
```

---

## Phase 3: LaTeX Paper Generation

### 3.1 LaTeX Template: `paper/main.tex`

Claude Code should generate this dynamically based on the actual results. Here is the template structure:

```latex
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{float}
\geometry{margin=2.5cm}

\title{Comparative Evaluation of Talking-Head Lip-Sync Models:\\
       A Multi-Metric Assessment}
\author{[Auto-generated Evaluation Report]}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This report presents a systematic quantitative evaluation of [N] talking-head
lip-sync model outputs. We assess each output across four evaluation dimensions:
lip-sync accuracy (LSE-C, LSE-D), temporal stability (mouth-region flicker),
visual fidelity (SSIM, LPIPS), and identity preservation (ArcFace cosine
similarity). Results are presented per-video with comparative analysis and
recommendations.
\end{abstract}

\section{Introduction}
% Claude Code: Write 2-3 paragraphs explaining:
% - The task (lip-sync correction/generation for talking head videos)
% - Why multi-metric evaluation matters (no single metric captures authenticity)
% - What models/methods were compared

\section{Evaluation Methodology}

\subsection{Metrics}

\subsubsection{Lip-Sync Accuracy}
% Explain LSE-C, LSE-D, AV offset and their origins (SyncNet)

\subsubsection{Temporal Stability}
% Explain mouth-region MAD, optical flow magnitude, what spikes mean

\subsubsection{Visual Fidelity}
% Explain SSIM and LPIPS, why they measure preservation not sync
% Emphasize: higher SSIM ≠ better sync; it means less visual change

\subsubsection{Identity Preservation}
% Explain ArcFace cosine similarity, sampling strategy

\subsection{Evaluation Pipeline}
% Describe the automated pipeline: frame extraction → face/mouth cropping →
% metric computation → aggregation

\section{Evaluated Models}
% Claude Code: auto-generate a table of all videos with their metadata
% from manifest.json

\begin{table}[H]
\centering
\caption{Summary of evaluated model outputs}
\begin{tabular}{lllll}
\toprule
Video ID & Model & Training & Steps & Resolution \\
\midrule
% AUTO-FILLED FROM manifest.json
\bottomrule
\end{tabular}
\end{table}

\section{Results}

\subsection{Lip-Sync Accuracy}
% Insert sync_metrics.pdf figure
% Discuss which model had best LSE-C/LSE-D
% Note any models with poor sync despite good visual quality

\subsection{Temporal Stability}
% Insert temporal_flicker.pdf figure
% Discuss flicker patterns, which models are smoothest
% Note any models with 95th-percentile spikes

\subsection{Visual Fidelity}
% Insert fidelity_metrics.pdf figure
% Discuss face vs mouth SSIM/LPIPS tradeoffs
% Important: note that lower SSIM in mouth region may be expected
% if the model correctly changed mouth shape for better sync

\subsection{Identity Preservation}
% Insert identity_preservation.pdf figure
% Discuss whether any model caused identity drift
% Report worst-case (p10) values

\subsection{Combined Results}
% Insert metrics_heatmap.pdf figure
% Insert the full results table

\begin{table}[H]
\centering
\caption{Full evaluation results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l*{9}{c}}
\toprule
Video & LSE-C↑ & LSE-D↓ & ID Cos↑ & Temp MAD↓ & Temp P95↓ &
Face SSIM↑ & Mouth LPIPS↓ & Flow Mean↓ \\
\midrule
% AUTO-FILLED FROM evaluation_results.csv
\bottomrule
\end{tabular}%
}
\end{table}

\section{Per-Video Analysis}
% Claude Code: for each video, write a short paragraph analyzing its
% strengths and weaknesses across all metrics. Include 1 sample frame
% figure per video if available.

\section{Discussion}
% Claude Code: write 3-4 paragraphs covering:
% - Which model performed best overall and why
% - Tradeoffs observed (e.g., better sync but more flicker)
% - Limitations of the evaluation (no human study, SyncNet limitations)
% - The importance of not using SSIM as a sync metric

\section{Conclusions}
% Claude Code: 1-2 paragraphs summarizing:
% - Best model recommendation
% - Key findings
% - Suggested next steps (human evaluation, larger test set, etc.)

\bibliographystyle{plain}
\begin{thebibliography}{9}
\bibitem{syncnet}
Chung, J.S. and Zisserman, A., ``Out of time: automated lip sync in the wild,''
\textit{ACCV Workshop}, 2016.

\bibitem{theval}
THEval evaluation framework for talking head video generation,
arXiv:2511.04520, 2025.

\bibitem{lpips}
Zhang, R., Isola, P., Efros, A.A., Shechtman, E., and Wang, O.,
``The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,''
\textit{CVPR}, 2018.

\bibitem{arcface}
Deng, J., Guo, J., Xue, N., and Zafeiriou, S.,
``ArcFace: Additive Angular Margin Loss for Deep Face Recognition,''
\textit{CVPR}, 2019.
\end{thebibliography}

\end{document}
```

### 3.2 Paper Generation Script: `scripts/generate_paper.py`

Claude Code should create a script that:

1. Reads `results/evaluation_results.csv` and `results/manifest.json`
2. Fills in the LaTeX template with actual data (tables, figure references)
3. Writes per-video analysis paragraphs by interpreting the metrics
4. Compiles the paper with `pdflatex`

```python
"""
Auto-generate the LaTeX paper from evaluation results.
Usage: python generate_paper.py --results results/ --output paper/
"""
# Claude Code: implement this to:
# 1. Load evaluation_results.csv into pandas
# 2. Load manifest.json for video metadata
# 3. Generate LaTeX table rows from the dataframe
# 4. Write per-video analysis (use metric thresholds to characterize each video)
# 5. Write discussion and conclusions based on comparative analysis
# 6. Output main.tex
# 7. Run: pdflatex -output-directory=paper paper/main.tex (twice for references)
```

---

## Phase 4: Execution Order (What Claude Code Should Run)

Claude Code should execute these steps **in this exact order**:

```
Step 1:  Scan for all video files and create eval_workspace/ structure
Step 2:  Rename and organize videos per naming convention → manifest.json
Step 3:  Install all Python dependencies
Step 4:  Attempt to clone SyncNet repo (handle failure gracefully)
Step 5:  Create all script files (evaluate.py, metrics/*, utils/*, etc.)
Step 6:  Run evaluate.py → produces results/evaluation_results.csv + .json
Step 7:  Run generate_figures.py → produces figures/*.pdf
Step 8:  Run generate_paper.py → produces paper/main.tex
Step 9:  Compile LaTeX → paper/main.pdf
Step 10: Print summary table to console
Step 11: Copy final outputs (paper PDF, results CSV, figures) to /mnt/user-data/outputs/
```

### Error Handling Rules

- If SyncNet clone fails → use fallback AV correlation, note in paper
- If InsightFace model download fails → skip identity metric, note as N/A
- If LPIPS download fails → skip LPIPS, keep SSIM only
- If a video has no detectable face → skip that video, log warning
- If LaTeX compilation fails → output the .tex file anyway for manual compilation
- Never let one video's failure stop the entire pipeline

### Final Deliverables in `/mnt/user-data/outputs/`

```
outputs/
├── evaluation_results.csv          # Full metrics table
├── evaluation_results.json         # Same, JSON format
├── evaluation_paper.pdf            # The compiled LaTeX paper
├── figures/
│   ├── sync_metrics.pdf
│   ├── identity_preservation.pdf
│   ├── temporal_flicker.pdf
│   ├── fidelity_metrics.pdf
│   └── metrics_heatmap.pdf
└── videos/                         # Renamed, organized videos
    └── *.mp4
```

---

## Quick-Start Command for Claude Code

Paste this into Claude Code to begin:

```
Read the file CLAUDE_CODE_IMPLEMENTATION_PLAN.md and execute it step by step.
My videos are in [PATH]. The original/reference video is [PATH].
Model details: [provide model names, training params, or say "infer from filenames"].
```
