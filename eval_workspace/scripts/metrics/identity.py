"""
Identity preservation: ArcFace cosine similarity between original and output frames.
"""
import numpy as np


def compute_identity_sim(orig_frames, out_frames, orig_fps=24, out_fps=25,
                         cropper=None, sample_interval=1.0):
    """
    Time-aligned ArcFace cosine similarity.

    Returns:
        {
            "identity_cos_mean": float,
            "identity_cos_p10": float,
            "identity_cos_std": float,
            "identity_n_frames": int,
        }
    """
    orig_dur = len(orig_frames) / orig_fps
    out_dur = len(out_frames) / out_fps
    min_dur = min(orig_dur, out_dur)

    similarities = []
    t = 0.0
    while t < min_dur:
        orig_idx = min(int(t * orig_fps), len(orig_frames) - 1)
        out_idx = min(int(t * out_fps), len(out_frames) - 1)

        orig_face = cropper.get_face(orig_frames[orig_idx])
        out_face = cropper.get_face(out_frames[out_idx])

        if orig_face is not None and out_face is not None:
            emb_orig = orig_face.embedding
            emb_out = out_face.embedding
            cos_sim = np.dot(emb_orig, emb_out) / (
                np.linalg.norm(emb_orig) * np.linalg.norm(emb_out) + 1e-8
            )
            similarities.append(float(cos_sim))

        t += sample_interval

    sims = np.array(similarities) if similarities else np.array([])
    return {
        "identity_cos_mean": float(np.mean(sims)) if len(sims) > 0 else None,
        "identity_cos_p10": float(np.percentile(sims, 10)) if len(sims) > 0 else None,
        "identity_cos_std": float(np.std(sims)) if len(sims) > 0 else None,
        "identity_n_frames": len(sims),
    }
