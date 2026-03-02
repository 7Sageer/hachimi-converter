#!/usr/bin/env python3
"""Train hachimi style transfer U-Net with PatchGAN adversarial loss."""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from discriminator import PatchDiscriminator
from losses import gan_loss_d, gan_loss_g
from mel_utils import N_MELS, log_mel, mel_spectrogram, normalize
from model import HachimiUNet
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).parent.parent / "data" / "paired"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def get_default_num_workers():
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count - 2))


class PairedMelDataset(Dataset):
    """Load paired original/hachimi segments as mel spectrograms."""

    def __init__(self, data_dir: Path, exclude_names=None):
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

        if orig_wav.shape[0] > 1:
            orig_wav = orig_wav.mean(dim=0, keepdim=True)
        if hach_wav.shape[0] > 1:
            hach_wav = hach_wav.mean(dim=0, keepdim=True)

        orig_mel = normalize(log_mel(mel_spectrogram(orig_wav.squeeze(0))))
        hach_mel = normalize(log_mel(mel_spectrogram(hach_wav.squeeze(0))))

        min_t = min(orig_mel.shape[-1], hach_mel.shape[-1])
        pad_t = ((min_t + 7) // 8) * 8
        orig_mel = self._pad_time(orig_mel[:, :, :min_t], pad_t)
        hach_mel = self._pad_time(hach_mel[:, :, :min_t], pad_t)

        return orig_mel, hach_mel

    def _pad_time(self, mel, target_t):
        pad = target_t - mel.shape[-1]
        if pad > 0:
            mel = nn.functional.pad(mel, [0, pad])
        return mel


def train(
    epochs=120,
    batch_size=32,
    lr_g=1e-3,
    lr_d=1e-4,
    lambda_l1=10.0,
    lambda_adv=1.0,
    exclude_names=None,
    device=DEFAULT_DEVICE,
    num_workers=None,
    prefetch_factor=4,
    persistent_workers=True,
):
    dataset = PairedMelDataset(DATA_DIR, exclude_names=exclude_names)
    if num_workers is None:
        num_workers = get_default_num_workers()

    use_pin_memory = device == "cuda"
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": True,  # GAN 训练需要固定 batch size
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    # Generator (U-Net) + Discriminator (PatchGAN)
    gen = HachimiUNet(n_mels=N_MELS, base_ch=32).to(device)
    disc = PatchDiscriminator(in_ch=1, base_ch=32).to(device)

    opt_g = torch.optim.AdamW(gen.parameters(), lr=lr_g, weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=lr_d, weight_decay=1e-4)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=epochs, eta_min=1e-5
    )
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_d, T_max=epochs, eta_min=1e-6
    )
    criterion_l1 = nn.L1Loss()

    g_params = sum(p.numel() for p in gen.parameters())
    d_params = sum(p.numel() for p in disc.parameters())
    print(
        f"Training on {device}, {len(dataset)} samples, {epochs} epochs, batch={batch_size}"
    )
    print(f"Loss: {lambda_l1}*L1 + {lambda_adv}*GAN(LSGAN)")
    print(f"Generator params: {g_params:,}  Discriminator params: {d_params:,}")

    best_loss = float("inf")
    for epoch in range(epochs):
        gen.train()
        disc.train()
        total_g = 0
        total_d = 0
        total_l1 = 0
        n_batches = 0

        for orig_mel, hach_mel in loader:
            orig_mel = orig_mel.to(device, non_blocking=use_pin_memory)
            hach_mel = hach_mel.to(device, non_blocking=use_pin_memory)

            fake_mel = gen(orig_mel)

            # ── Train Discriminator ──
            opt_d.zero_grad(set_to_none=True)
            loss_d = gan_loss_d(disc, hach_mel, fake_mel)
            loss_d.backward()
            opt_d.step()

            # ── Train Generator ──
            opt_g.zero_grad(set_to_none=True)
            loss_adv = gan_loss_g(disc, fake_mel)
            loss_l1 = criterion_l1(fake_mel, hach_mel)
            loss_g = lambda_adv * loss_adv + lambda_l1 * loss_l1
            loss_g.backward()
            opt_g.step()

            total_g += loss_g.item()
            total_d += loss_d.item()
            total_l1 += loss_l1.item()
            n_batches += 1

        sched_g.step()
        sched_d.step()

        avg_g = total_g / n_batches
        avg_d = total_d / n_batches
        avg_l1 = total_l1 / n_batches

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  epoch {epoch + 1}/{epochs}  "
                f"G={avg_g:.4f} D={avg_d:.4f} L1={avg_l1:.4f}  "
                f"lr_g={sched_g.get_last_lr()[0]:.6f}"
            )

        # 用 L1 部分来选最佳模型（L1 反映还原质量）
        if avg_l1 < best_loss:
            best_loss = avg_l1
            torch.save(gen.state_dict(), MODEL_DIR / "hachimi_unet_best.pt")

    torch.save(gen.state_dict(), MODEL_DIR / "hachimi_unet_final.pt")
    torch.save(disc.state_dict(), MODEL_DIR / "hachimi_disc.pt")
    print(f"Done. Best L1: {best_loss:.4f}")
    print(f"Models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-g", type=float, default=1e-3)
    parser.add_argument("--lr-d", type=float, default=1e-4)
    parser.add_argument("--lambda-l1", type=float, default=10.0)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
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
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lambda_l1=args.lambda_l1,
        lambda_adv=args.lambda_adv,
        exclude_names=args.exclude,
        device=selected_device,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.no_persistent_workers,
    )
