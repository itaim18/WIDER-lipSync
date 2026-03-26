"""
Temporal stability of the mouth region.
Measures frame-to-frame consistency via MAD and optical flow.
"""
import cv2
import numpy as np

CROP_SIZE = (96, 96)


def compute_temporal_flicker(mouth_crops):
    """
    Input: list of mouth-region crops (numpy arrays, BGR).

    Returns:
        {
            "temporal_mad_mean": float,
            "temporal_mad_std": float,
            "temporal_mad_p95": float,
            "temporal_flow_mean": float,
            "temporal_flow_std": float,
            "temporal_flow_p95": float,
        }
    """
    none_result = {k: None for k in [
        "temporal_mad_mean", "temporal_mad_std", "temporal_mad_p95",
        "temporal_flow_mean", "temporal_flow_std", "temporal_flow_p95",
    ]}

    if len(mouth_crops) < 2:
        return none_result

    # Resize all crops to consistent size
    resized = []
    for crop in mouth_crops:
        if crop.size == 0:
            continue
        resized.append(cv2.resize(crop, CROP_SIZE))

    if len(resized) < 2:
        return none_result

    mads = []
    flow_mags = []

    for i in range(len(resized) - 1):
        curr = resized[i].astype(np.float32)
        nxt = resized[i + 1].astype(np.float32)

        mad = np.mean(np.abs(curr - nxt))
        mads.append(mad)

        gray_curr = cv2.cvtColor(resized[i], cv2.COLOR_BGR2GRAY)
        gray_nxt = cv2.cvtColor(resized[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_curr, gray_nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
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
