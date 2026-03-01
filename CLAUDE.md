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

# Train U-Net
python scripts/train.py                        # default 80 epochs
python scripts/train.py --epochs 120
python scripts/train.py --exclude song_name    # hold out songs for eval

# Inference (convert audio)
python scripts/inference.py input.wav output.wav [duration] [offset]

# Visualize mel spectrogram comparison
python scripts/visualize.py
```

No test suite exists. No requirements.txt — dependencies are PyTorch, torchaudio, librosa, soundfile, matplotlib, yt-dlp, ffmpeg, gdown.

Download scripts require a proxy at `http://localhost:10808` (hardcoded in `download.py` and `download_v2.py`).

## Architecture

**Inference pipeline:** input wav → mel spectrogram → log → normalize [0,1] → U-Net → denormalize → clamp → HiFi-GAN vocoder → output wav

**Key mel parameters (must stay consistent across all scripts):**
- SR=22050, N_FFT=1024, HOP=256, WIN=1024, N_MELS=80, FMIN=0, FMAX=8000
- Normalization: `(log_mel + 5.0) / 6.5` — maps to ~[0,1] for U-Net
- These are tightly coupled to HiFi-GAN UNIVERSAL_V1 config

### Core modules

- `scripts/mel_utils.py` — Single source of truth for mel parameters and transforms. All other scripts import from here. Functions: mel_spectrogram, log_mel, normalize, denormalize, exp_mel.
- `scripts/model.py` — HachimiUNet: 3-level encoder-decoder with skip connections, base_ch=32, MaxPool2d(2) downsampling, ConvTranspose2d upsampling. Input/output: (B,1,80,T). Time dim must be padded to multiple of 8.
- `scripts/hifigan_model.py` — HiFi-GAN Generator vocoder (pretrained, frozen). Expects log-mel input (B,80,T). Config at `models/hifigan/config.json`.
- `scripts/train.py` — L1 loss, Adam lr=1e-3, cosine annealing. Device: MPS → CPU fallback. Saves best + final to `models/`.
- `scripts/inference.py` — Loads U-Net + HiFi-GAN, pads time to multiple of 8, clamps denormalized output to [-11.5, 0.5].

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
- Training device defaults to MPS (Apple Silicon); falls back to CPU.
- All scripts in `scripts/` use relative imports — run them from the `scripts/` directory or as `python scripts/foo.py`.
- HiFi-GAN checkpoint uses `weights_only=False` (old format) — this is intentional, not a bug.
