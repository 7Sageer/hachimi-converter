#!/usr/bin/env python3
"""Train hachimi style transfer U-Net on paired mel spectrograms."""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from mel_utils import N_MELS, log_mel, mel_spectrogram, normalize
from model import HachimiUNet
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).parent.parent / "data" / "paired"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def get_default_num_workers():
    """Choose a practical default for Apple Silicon and similar CPUs."""
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count - 2))


class PairedMelDataset(Dataset):
    """Load paired original/hachimi segments as mel spectrograms."""

    def __init__(self, data_dir: Path, exclude_names=None):
        # Find all pairs, optionally excluding songs by name prefix
        self.pairs = []
        hach_files = sorted(data_dir.glob("*_hach_*.wav"))
        for hf in hach_files:
            if exclude_names and any(
                hf.name.startswith(n + "_") for n in exclude_names
            ):
                continue
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

        # Mono
        if orig_wav.shape[0] > 1:
            orig_wav = orig_wav.mean(dim=0, keepdim=True)
        if hach_wav.shape[0] > 1:
            hach_wav = hach_wav.mean(dim=0, keepdim=True)

        # To normalized log-mel
        orig_mel = normalize(log_mel(mel_spectrogram(orig_wav.squeeze(0))))
        hach_mel = normalize(log_mel(mel_spectrogram(hach_wav.squeeze(0))))

        # Truncate to same time length
        min_t = min(orig_mel.shape[-1], hach_mel.shape[-1])
        # Pad to multiple of 8 for U-Net pooling
        pad_t = ((min_t + 7) // 8) * 8
        orig_mel = self._pad_time(orig_mel[:, :, :min_t], pad_t)
        hach_mel = self._pad_time(hach_mel[:, :, :min_t], pad_t)

        # mel shape is (1, 80, T) — dim 0 serves as channel for U-Net
        return orig_mel, hach_mel

    def _pad_time(self, mel, target_t):
        pad = target_t - mel.shape[-1]
        if pad > 0:
            mel = nn.functional.pad(mel, [0, pad])
        return mel


def train(
    epochs=120,
    batch_size=32,
    lr=1e-3,
    exclude_names=None,
    device=DEFAULT_DEVICE,
    num_workers=None,
    prefetch_factor=4,
    persistent_workers=True,
):
    dataset = PairedMelDataset(DATA_DIR, exclude_names=exclude_names)
    if num_workers is None:
        num_workers = get_default_num_workers()
    if num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1")

    use_pin_memory = device == "cuda"
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    model = HachimiUNet(n_mels=N_MELS, base_ch=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    criterion = nn.L1Loss()

    print(
        f"Training on {device}, {len(dataset)} samples, {epochs} epochs, batch={batch_size}"
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}")
    if num_workers > 0:
        print(
            f"DataLoader: workers={num_workers}, persistent={persistent_workers}, prefetch={prefetch_factor}, pin_memory={use_pin_memory}"
        )
    else:
        print(f"DataLoader: workers=0, pin_memory={use_pin_memory}")

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for orig_mel, hach_mel in loader:
            orig_mel = orig_mel.to(device, non_blocking=use_pin_memory)
            hach_mel = hach_mel.to(device, non_blocking=use_pin_memory)

            pred = model(orig_mel)
            loss = criterion(pred, hach_mel)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_DIR / "hachimi_unet_best.pt")

    torch.save(model.state_dict(), MODEL_DIR / "hachimi_unet_final.pt")
    print(f"Done. Best loss: {best_loss:.4f}")
    print(f"Models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", choices=["auto", "mps", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument(
        "--exclude", nargs="*", help="Song names to exclude from training"
    )
    args = parser.parse_args()
    selected_device = DEFAULT_DEVICE if args.device == "auto" else args.device
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        exclude_names=args.exclude,
        device=selected_device,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.no_persistent_workers,
    )
