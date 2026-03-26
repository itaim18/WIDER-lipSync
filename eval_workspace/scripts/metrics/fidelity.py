"""
Visual fidelity metrics: SSIM and LPIPS on face and mouth crops.
"""
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
import lpips

_lpips_model = None


def _get_lpips():
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex")
        if torch.cuda.is_available():
            _lpips_model = _lpips_model.cuda()
    return _lpips_model


def _to_tensor(img_bgr):
    """Convert BGR uint8 image to LPIPS-compatible tensor [-1, 1]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    t = t * 2 - 1
    t = t.unsqueeze(0)
    if torch.cuda.is_available():
        t = t.cuda()
    return t


def compute_fidelity(orig_crops, out_crops, orig_fps=24, out_fps=25):
    """
    Compute SSIM and LPIPS between time-aligned face and mouth crops.

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

        if not orig_list or not out_list:
            results[f"{region}_ssim_mean"] = None
            results[f"{region}_lpips_mean"] = None
            continue

        # Time-aligned sampling: pick ~50 evenly spaced indices
        n_orig = len(orig_list)
        n_out = len(out_list)
        n_samples = min(50, n_orig, n_out)

        ssim_vals = []
        lpips_vals = []

        for s in range(n_samples):
            frac = s / max(n_samples - 1, 1)
            oi = min(int(frac * (n_orig - 1)), n_orig - 1)
            gi = min(int(frac * (n_out - 1)), n_out - 1)

            o = orig_list[oi]
            g = out_list[gi]

            # Resize to match
            target_h = min(o.shape[0], g.shape[0])
            target_w = min(o.shape[1], g.shape[1])
            if target_h < 8 or target_w < 8:
                continue
            o = cv2.resize(o, (target_w, target_h))
            g = cv2.resize(g, (target_w, target_h))

            # SSIM
            o_gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
            g_gray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
            win_size = min(7, target_h, target_w)
            if win_size % 2 == 0:
                win_size -= 1
            if win_size >= 3:
                s_val = ssim(o_gray, g_gray, win_size=win_size)
                ssim_vals.append(s_val)

            # LPIPS
            o_t = _to_tensor(o)
            g_t = _to_tensor(g)
            with torch.no_grad():
                lp = _get_lpips()(o_t, g_t).item()
            lpips_vals.append(lp)

        results[f"{region}_ssim_mean"] = float(np.mean(ssim_vals)) if ssim_vals else None
        results[f"{region}_lpips_mean"] = float(np.mean(lpips_vals)) if lpips_vals else None

    return results
