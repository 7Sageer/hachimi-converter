# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

hachimi-converter is an audio style transfer system that converts music into "Hachimi style" using a mel spectrogram U-Net + HiFi-GAN vocoder. Training data comes from paired original/Hachimi song versions, aligned via chroma cross-correlation.

The project is written in Chinese (README, comments) — follow this convention.

## Commands

```bash
# Download paired audio — two paths:
python scripts/download.py        # from Bilibili (uses pairs.jsonl)
python scripts/download_v2.py     # from Hachimi World API + YouTube (uses pairs_full.jsonl)

# Validate pair quality (chroma alignment scoring)
python scripts/validate_pairs.py

# Download HiFi-GAN vocoder weights
python scripts/download_vocoder.py

# Align & slice pairs into 3-second training segments
python scripts/slice_v2.py

# Train U-Net (GAN: L1 + PatchGAN adversarial)
python scripts/train.py                        # default 120 epochs
python scripts/train.py --epochs 200
python scripts/train.py --exclude song_name    # hold out songs for eval
python scripts/train.py --amp                  # AMP mixed precision (CUDA only)

# Train Flow Matching (recommended, replaces GAN)
python scripts/train_fm.py                     # default 120 epochs
python scripts/train_fm.py --epochs 200
python scripts/train_fm.py --device mps --batch-size 32

# Fine-tune HiFi-GAN vocoder on hachimi data
python scripts/train_vocoder.py

# Inference (convert audio)
python scripts/inference.py input.wav output.wav [duration] [offset]        # GAN model
python scripts/inference_fm.py input.wav output.wav [duration] [offset]     # Flow Matching
python scripts/inference_fm.py input.wav output.wav --steps 20              # more ODE steps = better quality
python scripts/batch_inference.py              # auto-detect FM or GAN model
python scripts/batch_inference.py --fm --steps 10   # force Flow Matching

# Visualize mel spectrogram comparison
python scripts/visualize.py
```

No test suite exists. The project uses uv for dependency management. Dependencies: PyTorch, torchaudio, librosa, soundfile, matplotlib, yt-dlp, ffmpeg, gdown.

Download scripts require a proxy at `http://localhost:10808` (hardcoded in `download.py` and `download_v2.py`).

## Architecture

**GAN pipeline:** input wav → mel spectrogram → log → normalize [0,1] → U-Net → denormalize → clamp → HiFi-GAN vocoder → output wav

**Flow Matching pipeline:** input wav → mel → log → normalize → Euler ODE (N steps through FlowNet) → denormalize → clamp → HiFi-GAN → output wav

Flow Matching learns a velocity field v(x_t, t) that transports source_mel → target_mel along straight (optimal transport) paths. At inference, start from source_mel and integrate the ODE with Euler steps.

**Key mel parameters (must stay consistent across all scripts):**
- SR=22050, N_FFT=1024, HOP=256, WIN=1024, N_MELS=80, FMIN=0, FMAX=8000
- Normalization: `(log_mel + 5.0) / 6.5` — maps to ~[0,1] for U-Net
- These are tightly coupled to HiFi-GAN UNIVERSAL_V1 config

### Core modules

- `scripts/mel_utils.py` — Single source of truth for mel parameters and transforms. All other scripts import from here. Functions: mel_spectrogram, log_mel, normalize, denormalize, exp_mel.
- `scripts/model.py` — HachimiUNet: 3-level encoder-decoder with skip connections, base_ch=32, MaxPool2d(2) downsampling, ConvTranspose2d upsampling. Input/output: (B,1,80,T). Time dim must be padded to multiple of 8.
- `scripts/model_fm.py` — HachimiFlowNet: same U-Net architecture as HachimiUNet but with FiLM time conditioning (sinusoidal embedding → MLP → scale+shift in each ConvBlock). Input: (B,1,80,T) + scalar t. Output: velocity (B,1,80,T).
- `scripts/hifigan_model.py` — HiFi-GAN Generator vocoder (pretrained, frozen). Expects log-mel input (B,80,T). Config at `models/hifigan/config.json`.
- `scripts/discriminator.py` — PatchDiscriminator: 2D PatchGAN on mel spectrograms, ~16-frame receptive field (~186ms).
- `scripts/losses.py` — LSGAN loss functions for generator and discriminator.
- `scripts/train.py` — GAN training: L1 + PatchGAN adversarial (LSGAN). AdamW, cosine annealing. Device: CUDA → MPS → CPU. Supports AMP mixed precision on CUDA. Saves best + final to `models/`.
- `scripts/train_fm.py` — Flow Matching training: single MSE velocity-prediction loss. No discriminator. Dramatically simpler than GAN. Saves to `models/hachimi_fm_{best,final}.pt`.
- `scripts/train_vocoder.py` — Fine-tunes HiFi-GAN vocoder on hachimi paired data to adapt to U-Net output distribution.
- `scripts/inference.py` — Loads U-Net + HiFi-GAN, pads time to multiple of 8, clamps denormalized output to [-11.5, 0.5].
- `scripts/inference_fm.py` — Flow Matching inference with Euler ODE solver. Default 10 steps. Configurable via --steps.
- `scripts/batch_inference.py` — Batch converts preset sample list using `inference.convert()`.

### Data pipeline

Two download paths exist:
1. `download.py` — yt-dlp from Bilibili using `pairs.jsonl` (both hachimi + original from Bilibili)
2. `download_v2.py` — Hachimi audio from Hachimi World API + original from YouTube search using `pairs_full.jsonl`

Then:
3. `validate_pairs.py` — Chroma alignment scoring, filters pairs with score < 0.5 → writes `pairs_validated.jsonl`
4. `align_v2.py` — Chroma cross-correlation to find offset of hachimi within original (threshold: score > 0.3)
5. `slice_v2.py` — Cuts aligned pairs into 3-second segments → `data/paired/`

### Data directory layout

```
data/
  original/    # full original songs (.wav)
  hachimi/     # full hachimi versions (.wav or .mp3)
  paired/      # training segments: {name}_orig_{idx}.wav, {name}_hach_{idx}.wav
```

## Key Constraints

- Mel parameters are tightly coupled to HiFi-GAN UNIVERSAL_V1 — changing them requires retraining everything.
- `align_and_slice.py` is a legacy DTW-based approach; `align_v2.py` + `slice_v2.py` is the current pipeline.
- Model weights are in `models/` (gitignored). Vocoder weights must be downloaded separately via `download_vocoder.py`.
- Training device priority: CUDA → MPS (Apple Silicon) → CPU.
- All scripts in `scripts/` use relative imports — run them from the `scripts/` directory or as `python scripts/foo.py`.
- HiFi-GAN checkpoint uses `weights_only=False` (old format) — this is intentional, not a bug.

## Coding Conventions

- 4-space indentation, PEP 8 style. `snake_case` for functions/files, `PascalCase` for classes, `UPPER_CASE` for constants.
- Prefer `pathlib.Path` for filesystem paths.
- Commit style: short, imperative, lowercase summaries, often with scope (e.g. `add validate_pairs.py: chroma alignment quality check`).
- Do not commit generated artifacts from ignored paths (`data/`, `models/`, `__pycache__/`, `/tmp/`).
