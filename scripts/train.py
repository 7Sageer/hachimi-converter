#!/usr/bin/env python3
"""Train hachimi style transfer U-Net on paired mel spectrograms."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from pathlib import Path
from model import HachimiUNet

DATA_DIR = Path(__file__).parent.parent / "data" / "paired"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

SR = 22050
N_MELS = 128
N_FFT = 2048
HOP = 512
# 780M iGPU hangs with conv2d under ROCm, use CPU for now
DEVICE = "cpu"


class PairedMelDataset(Dataset):
    """Load paired original/hachimi segments as mel spectrograms."""

    def __init__(self, data_dir: Path, n_mels=N_MELS, sr=SR):
        self.sr = sr
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=N_FFT, hop_length=HOP, n_mels=n_mels
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

        # Find all pairs
        self.pairs = []
        hach_files = sorted(data_dir.glob("*_hach_*.wav"))
        for hf in hach_files:
            of = data_dir / hf.name.replace("_hach_", "_orig_")
            if of.exists():
                self.pairs.append((of, hf))
        print(f"Found {len(self.pairs)} paired segments")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        orig_path, hach_path = self.pairs[idx]

        orig_wav, _ = torchaudio.load(orig_path)
        hach_wav, _ = torchaudio.load(hach_path)

        # To mel spectrogram (dB scale)
        orig_mel = self.amp_to_db(self.mel_transform(orig_wav))
        hach_mel = self.amp_to_db(self.mel_transform(hach_wav))

        # Truncate to same time length
        min_t = min(orig_mel.shape[-1], hach_mel.shape[-1])
        # Pad to multiple of 8 for U-Net pooling
        pad_t = ((min_t + 7) // 8) * 8
        orig_mel = self._pad_time(orig_mel[:, :, :min_t], pad_t)
        hach_mel = self._pad_time(hach_mel[:, :, :min_t], pad_t)

        # Normalize to [-1, 1]
        orig_mel = self._normalize(orig_mel)
        hach_mel = self._normalize(hach_mel)

        return orig_mel, hach_mel

    def _pad_time(self, mel, target_t):
        pad = target_t - mel.shape[-1]
        if pad > 0:
            mel = nn.functional.pad(mel, [0, pad])
        return mel

    def _normalize(self, mel):
        # mel dB range is roughly [-80, 0], normalize to [-1, 1]
        return (mel + 40) / 40  # maps -80->-1, 0->1


def train(epochs=50, batch_size=4, lr=1e-3):
    dataset = PairedMelDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Smaller model for CPU training
    model = HachimiUNet(n_mels=128, base_ch=16).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()

    print(f"Training on {DEVICE}, {len(dataset)} samples, {epochs} epochs")
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}")

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for orig_mel, hach_mel in loader:
            orig_mel = orig_mel.to(DEVICE)
            hach_mel = hach_mel.to(DEVICE)

            pred = model(orig_mel)
            loss = criterion(pred, hach_mel)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_DIR / "hachimi_unet_best.pt")

    torch.save(model.state_dict(), MODEL_DIR / "hachimi_unet_final.pt")
    print(f"Done. Best loss: {best_loss:.4f}")
    print(f"Models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    train()
