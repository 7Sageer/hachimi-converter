#!/usr/bin/env python3
"""Download pretrained HiFi-GAN UNIVERSAL_V1 vocoder weights from Google Drive."""

import json
from pathlib import Path

HIFIGAN_DIR = Path(__file__).parent.parent / "models" / "hifigan"

# jik876/hifi-gan UNIVERSAL_V1 — trained on multiple speakers, best generalization
CHECKPOINT_ID = "1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW"
CONFIG_URL = "https://raw.githubusercontent.com/jik876/hifi-gan/master/config_v1.json"

# Default UNIVERSAL_V1 config (fallback if download fails)
DEFAULT_CONFIG = {
    "resblock": "1",
    "num_gpus": 0,
    "batch_size": 16,
    "learning_rate": 0.0002,
    "adam_b1": 0.8,
    "adam_b2": 0.99,
    "lr_decay": 0.999,
    "seed": 1234,
    "upsample_rates": [8, 8, 2, 2],
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "segment_size": 8192,
    "num_mels": 80,
    "num_freq": 1025,
    "n_fft": 1024,
    "hop_size": 256,
    "win_size": 1024,
    "sampling_rate": 22050,
    "fmin": 0,
    "fmax": 8000,
    "fmax_for_loss": None,
}


def download_checkpoint():
    """Download UNIVERSAL_V1 generator checkpoint using gdown."""
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        import subprocess
        subprocess.check_call(["pip", "install", "gdown"])
        import gdown

    HIFIGAN_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = HIFIGAN_DIR / "generator_v1"
    if ckpt_path.exists():
        print(f"Checkpoint already exists: {ckpt_path}")
    else:
        print("Downloading HiFi-GAN UNIVERSAL_V1 checkpoint...")
        gdown.download(id=CHECKPOINT_ID, output=str(ckpt_path), quiet=False)
        print(f"Saved: {ckpt_path}")

    config_path = HIFIGAN_DIR / "config.json"
    if config_path.exists():
        print(f"Config already exists: {config_path}")
    else:
        print("Writing HiFi-GAN config...")
        with open(config_path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"Saved: {config_path}")

    return ckpt_path, config_path


if __name__ == "__main__":
    ckpt, cfg = download_checkpoint()
    print(f"\nReady. Files in {HIFIGAN_DIR}/")
    print(f"  checkpoint: {ckpt}")
    print(f"  config:     {cfg}")
