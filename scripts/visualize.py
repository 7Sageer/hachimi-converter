#!/usr/bin/env python3
"""Visualize mel spectrogram comparison between original and hachimi versions."""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR / "viz"
OUT_DIR.mkdir(exist_ok=True)

SR = 22050
N_MELS = 128
HOP = 512
DURATION = 15  # seconds to compare


def load_mel(path: Path, duration=DURATION):
    y, sr = librosa.load(path, sr=SR, duration=duration)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db, y


def plot_pair(name: str):
    orig_path = DATA_DIR / "original" / f"{name}.wav"
    hach_path = DATA_DIR / "hachimi" / f"{name}.wav"

    if not orig_path.exists() or not hach_path.exists():
        print(f"[skip] {name}: missing files")
        return

    print(f"[viz] {name}")
    orig_mel, orig_y = load_mel(orig_path)
    hach_mel, hach_y = load_mel(hach_path)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Original
    librosa.display.specshow(orig_mel, sr=SR, hop_length=HOP,
                             x_axis='time', y_axis='mel', ax=axes[0])
    axes[0].set_title(f'{name} - Original')

    # Hachimi
    librosa.display.specshow(hach_mel, sr=SR, hop_length=HOP,
                             x_axis='time', y_axis='mel', ax=axes[1])
    axes[1].set_title(f'{name} - Hachimi')

    # Difference (truncate to same length)
    min_len = min(orig_mel.shape[1], hach_mel.shape[1])
    diff = orig_mel[:, :min_len] - hach_mel[:, :min_len]
    librosa.display.specshow(diff, sr=SR, hop_length=HOP,
                             x_axis='time', y_axis='mel', ax=axes[2],
                             cmap='RdBu_r')
    axes[2].set_title(f'{name} - Difference (Original - Hachimi)')

    plt.tight_layout()
    out_path = OUT_DIR / f"{name}_comparison.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  saved: {out_path}")


def main():
    names = [p.stem for p in (DATA_DIR / "hachimi").glob("*.wav")]
    for name in sorted(names):
        plot_pair(name)
    print(f"\nDone. Check {OUT_DIR}/")


if __name__ == "__main__":
    main()
