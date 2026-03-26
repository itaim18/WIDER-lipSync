#!/usr/bin/env python3
"""
Main evaluation orchestrator.
Runs all metrics on all output videos and saves results.
"""
import os
import sys
import json
import traceback

import pandas as pd
from pathlib import Path
from tqdm import tqdm

WORKSPACE = "/home/itaim/wider-itai/eval_workspace"
VIDEOS_DIR = os.path.join(WORKSPACE, "videos/outputs")
ORIGINALS_DIR = os.path.join(WORKSPACE, "videos/originals")
RESULTS_DIR = os.path.join(WORKSPACE, "results")
ORIGINAL_VIDEO = os.path.join(ORIGINALS_DIR, "original_reference.mp4")

sys.path.insert(0, os.path.join(WORKSPACE, "scripts"))

from utils.frames import extract_frames, get_video_info
from utils.face_crop import FaceCropper
from metrics.sync import compute_sync_metrics
from metrics.identity import compute_identity_sim
from metrics.temporal import compute_temporal_flicker
from metrics.fidelity import compute_fidelity


def main():
    manifest_path = os.path.join(RESULTS_DIR, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load original video once
    print("Loading original video frames...")
    orig_info = get_video_info(ORIGINAL_VIDEO)
    orig_frames = extract_frames(ORIGINAL_VIDEO)
    print(f"  Original: {orig_info['frame_count']} frames @ {orig_info['fps']}fps, "
          f"{orig_info['width']}x{orig_info['height']}")

    # Initialize face cropper (loads InsightFace once)
    print("Initializing InsightFace face cropper...")
    cropper = FaceCropper()

    print("Extracting original face/mouth crops...")
    orig_crops = cropper.extract_crops(orig_frames)
    print(f"  Original crops: {len(orig_crops['face_crops'])} faces, "
          f"{len(orig_crops['mouth_crops'])} mouths")

    # Process each output video
    all_results = []
    output_videos = sorted([
        f for f in os.listdir(VIDEOS_DIR) if f.endswith(".mp4")
    ])

    for video_file in output_videos:
        video_name = Path(video_file).stem
        video_path = os.path.join(VIDEOS_DIR, video_file)
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {video_name}")
        print(f"{'=' * 60}")

        result = {"video": video_name}
        meta = manifest.get(video_name, {})
        result["model"] = meta.get("model", "unknown")
        result["variant"] = meta.get("variant", "unknown")

        out_info = get_video_info(video_path)
        result["fps"] = out_info["fps"]
        result["frame_count"] = out_info["frame_count"]

        # --- Sync metrics ---
        print("  Running SyncNet evaluation...")
        try:
            sync = compute_sync_metrics(video_path)
            result.update(sync)
            print(f"    sync_conf={sync.get('sync_conf')}, av_offset={sync.get('av_offset')}")
        except Exception as e:
            print(f"    SYNC FAILED: {e}")
            result.update({"sync_conf": None, "av_offset": None})

        # --- Load output frames and crops ---
        print("  Extracting frames and crops...")
        try:
            out_frames = extract_frames(video_path)
            out_crops = cropper.extract_crops(out_frames)
            print(f"    {len(out_frames)} frames, {len(out_crops['face_crops'])} face crops")
        except Exception as e:
            print(f"    FRAME EXTRACTION FAILED: {e}")
            all_results.append(result)
            continue

        # --- Identity ---
        print("  Computing identity preservation...")
        try:
            identity = compute_identity_sim(
                orig_frames, out_frames,
                orig_fps=orig_info["fps"],
                out_fps=out_info["fps"],
                cropper=cropper,
                sample_interval=1.0,
            )
            result.update(identity)
            print(f"    cos_mean={identity.get('identity_cos_mean')}")
        except Exception as e:
            print(f"    IDENTITY FAILED: {e}")
            traceback.print_exc()

        # --- Temporal flicker ---
        print("  Computing temporal stability...")
        try:
            temporal = compute_temporal_flicker(out_crops["mouth_crops"])
            result.update(temporal)
            print(f"    mad_mean={temporal.get('temporal_mad_mean'):.2f}, "
                  f"flow_mean={temporal.get('temporal_flow_mean'):.2f}")
        except Exception as e:
            print(f"    TEMPORAL FAILED: {e}")
            traceback.print_exc()

        # --- Fidelity ---
        print("  Computing visual fidelity...")
        try:
            fidelity = compute_fidelity(
                orig_crops, out_crops,
                orig_fps=orig_info["fps"],
                out_fps=out_info["fps"],
            )
            result.update(fidelity)
            print(f"    face_ssim={fidelity.get('face_ssim_mean')}, "
                  f"mouth_lpips={fidelity.get('mouth_lpips_mean')}")
        except Exception as e:
            print(f"    FIDELITY FAILED: {e}")
            traceback.print_exc()

        all_results.append(result)

        # Free output frames
        del out_frames, out_crops

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    json_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))
    print(f"\nResults saved to:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    main()
