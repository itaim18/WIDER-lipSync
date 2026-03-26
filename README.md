# WIDER-lipSync

Comparative evaluation of open-source talking-head lip-sync models on a real-world video (grandma holding a plate of cookies while talking).

## Models Tested

- **LatentSync v1.6** -- Latent diffusion-based lip-sync (20 and 40 inference steps)
- **MuseTalk v1.5** -- VAE-based real-time lip-sync
- **Occlusion fix** -- BiSeNet face-parsing post-composite to restore objects (plate/cookies) overwritten by face generation

## Results

**Best result: `latentsync_smooth`** (LatentSync, 40 steps, no DeepCache) -- highest identity preservation (0.96), best mouth quality (LPIPS 0.106), and near-zero occlusion issues without any post-processing.

The occlusion fix is mathematically correct per-frame but introduces visible temporal flickering due to inconsistent face parsing masks between frames.

Full evaluation paper: [`eval_workspace/paper/main.pdf`](eval_workspace/paper/main.pdf)

## Videos

| Video | Description |
|-------|-------------|
| [`latentsync_smooth.mp4`](https://github.com/itaim18/WIDER-lipSync/releases/tag/v1.0) | Best overall -- LatentSync 40 steps |
| [`latentsync_fast.mp4`](https://github.com/itaim18/WIDER-lipSync/releases/tag/v1.0) | LatentSync 20 steps + DeepCache |
| [`latentsync_occfix.mp4`](https://github.com/itaim18/WIDER-lipSync/releases/tag/v1.0) | LatentSync + occlusion fix |
| [`musetalk_raw.mp4`](https://github.com/itaim18/WIDER-lipSync/releases/tag/v1.0) | MuseTalk v1.5 baseline |
| [`musetalk_occfix.mp4`](https://github.com/itaim18/WIDER-lipSync/releases/tag/v1.0) | MuseTalk + occlusion fix |
| [`original_reference.mp4`](https://github.com/itaim18/WIDER-lipSync/releases/tag/v1.0) | Original input video |

## Evaluation Metrics

- **Lip-sync accuracy** -- SyncNet confidence + AV offset
- **Identity preservation** -- ArcFace cosine similarity
- **Temporal stability** -- Mouth-region MAD + optical flow
- **Visual fidelity** -- SSIM + LPIPS on face/mouth crops

## Repo Structure

```
eval_workspace/
  scripts/          # Evaluation pipeline (Python)
  results/          # Metrics CSV + JSON
  figures/          # Generated plots
  paper/            # LaTeX source + compiled PDF
  videos/           # All video outputs
occlusion_fix.py    # Post-composite occlusion restoration script
MuseTalk/           # MuseTalk model code (weights not included)
LatentSync/         # LatentSync model code (weights not included)
```

## Running the Evaluation

```bash
conda activate latentsync
cd eval_workspace
python scripts/evaluate_all.py      # Run all metrics
python scripts/generate_figures.py   # Generate plots
python scripts/generate_paper.py     # Generate + compile PDF
```

Requires: PyTorch 2.7+, InsightFace, LPIPS, scikit-image, OpenCV, pandas, matplotlib, seaborn.
