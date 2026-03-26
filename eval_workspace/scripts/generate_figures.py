#!/usr/bin/env python3
"""
Generate comparison plots from evaluation_results.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

WORKSPACE = "/home/itaim/wider-itai/eval_workspace"
RESULTS_CSV = os.path.join(WORKSPACE, "results/evaluation_results.csv")
FIGURES_DIR = os.path.join(WORKSPACE, "figures")


def short_name(video):
    """Shorten video names for plot labels."""
    return video.replace("latentsync_", "LS_").replace("musetalk_", "MT_")


def generate_all_figures():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(RESULTS_CSV)
    df["short"] = df["video"].apply(short_name)

    colors_model = {"musetalk": "#4C72B0", "latentsync": "#DD8452"}
    bar_colors = [colors_model.get(m, "#888888") for m in df["model"]]

    # 1. Sync metrics
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if "sync_conf" in df.columns and df["sync_conf"].notna().any():
        axes[0].bar(df["short"], df["sync_conf"], color=bar_colors, alpha=0.85)
        axes[0].set_title("Sync Confidence (higher = better)")
        axes[0].set_ylabel("Confidence")
        axes[0].tick_params(axis="x", rotation=35)

        axes[1].bar(df["short"], df["av_offset"].abs(), color=bar_colors, alpha=0.85)
        axes[1].set_title("AV Offset (closer to 0 = better)")
        axes[1].set_ylabel("|Offset| (frames)")
        axes[1].tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sync_metrics.pdf"), bbox_inches="tight")
    plt.close()

    # 2. Identity preservation
    if "identity_cos_mean" in df.columns and df["identity_cos_mean"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(df["short"], df["identity_cos_mean"], color=bar_colors, alpha=0.85)
        if "identity_cos_std" in df.columns:
            ax.errorbar(df["short"], df["identity_cos_mean"],
                        yerr=df["identity_cos_std"], fmt="none", color="black", capsize=3)
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Identity Preservation (ArcFace)")
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "identity_preservation.pdf"), bbox_inches="tight")
        plt.close()

    # 3. Temporal flicker
    if "temporal_mad_mean" in df.columns and df["temporal_mad_mean"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(df))
        ax.bar(x, df["temporal_mad_mean"], color=bar_colors, alpha=0.85, label="Mean MAD")
        ax.scatter(x, df["temporal_mad_p95"], color="red", zorder=5, s=50, label="95th pct")
        ax.set_xticks(x)
        ax.set_xticklabels(df["short"], rotation=35, ha="right")
        ax.set_ylabel("Mean Absolute Difference")
        ax.set_title("Temporal Stability - Mouth Region Flicker")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "temporal_flicker.pdf"), bbox_inches="tight")
        plt.close()

    # 4. Fidelity: SSIM + LPIPS
    fidelity_cols = ["face_ssim_mean", "mouth_ssim_mean", "face_lpips_mean", "mouth_lpips_mean"]
    if any(c in df.columns for c in fidelity_cols):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ssim_cols = [c for c in ["face_ssim_mean", "mouth_ssim_mean"] if c in df.columns]
        if ssim_cols:
            df.plot.bar(x="short", y=ssim_cols, ax=axes[0],
                        color=["steelblue", "skyblue"])
            axes[0].set_title("SSIM (higher = more preserved)")
            axes[0].set_ylabel("SSIM")
            axes[0].tick_params(axis="x", rotation=35)

        lpips_cols = [c for c in ["face_lpips_mean", "mouth_lpips_mean"] if c in df.columns]
        if lpips_cols:
            df.plot.bar(x="short", y=lpips_cols, ax=axes[1],
                        color=["coral", "lightsalmon"])
            axes[1].set_title("LPIPS (lower = more preserved)")
            axes[1].set_ylabel("LPIPS")
            axes[1].tick_params(axis="x", rotation=35)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "fidelity_metrics.pdf"), bbox_inches="tight")
        plt.close()

    # 5. Summary heatmap
    metric_cols = [c for c in df.columns
                   if c not in ["video", "short", "model", "variant", "fps",
                                "frame_count", "identity_n_frames"]
                   and df[c].dtype in ["float64", "int64"]
                   and df[c].notna().any()]
    if metric_cols:
        numeric_df = df[["short"] + metric_cols].set_index("short")
        normed = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min() + 1e-8)

        fig, ax = plt.subplots(figsize=(max(10, len(metric_cols) * 1.2),
                                        max(4, len(df) * 0.8)))
        sns.heatmap(normed, annot=numeric_df.round(3).values, fmt="",
                    cmap="YlOrRd_r", ax=ax, linewidths=0.5,
                    xticklabels=metric_cols)
        ax.set_title("Evaluation Metrics Heatmap (raw values annotated)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "metrics_heatmap.pdf"), bbox_inches="tight")
        plt.close()

    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
