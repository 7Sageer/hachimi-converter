#!/usr/bin/env python3
"""Train hachimi style transfer Conformer with PatchGAN adversarial loss."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
from discriminator import PatchDiscriminator
from losses import gan_loss_d, gan_loss_g, feature_matching_loss
from mel_utils import N_MELS, log_mel, mel_spectrogram, normalize
from model import HachimiConformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data" / "paired"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class PairedMelDataset(Dataset):
    """Load paired original/hachimi segments as mel spectrograms.

    所有 mel 在 __init__ 时预计算并缓存在内存中，避免训练时反复读盘和 CPU 计算。
    """

    def __init__(self, data_dir: Path, exclude_names=None):
        pairs = []
        hach_files = sorted(data_dir.glob("*_hach_*.wav"))
        for hf in hach_files:
            if exclude_names and any(
                hf.name.startswith(n + "_") for n in exclude_names
            ):
                continue
            of = data_dir / hf.name.replace("_hach_", "_orig_")
            if of.exists():
                pairs.append((of, hf))
        print(f"Found {len(pairs)} paired segments, pre-computing mels...")

        self.cached_mels = []
        for orig_path, hach_path in tqdm(pairs, desc="Caching mels"):
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

            self.cached_mels.append((orig_mel, hach_mel))

        print(f"Cached {len(self.cached_mels)} mel pairs in memory")

    def __len__(self):
        return len(self.cached_mels)

    def __getitem__(self, idx):
        return self.cached_mels[idx]

    def _pad_time(self, mel, target_t):
        pad = target_t - mel.shape[-1]
        if pad > 0:
            mel = nn.functional.pad(mel, [0, pad])
        return mel


def plot_curves(history, save_path):
    """绘制训练曲线并保存。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")
    epochs = range(1, len(history["g"]) + 1)

    # 左上：G/D loss
    ax = axes[0, 0]
    ax.plot(epochs, history["g"], label="G (total)", linewidth=1.5)
    ax.plot(epochs, history["d"], label="D", linewidth=1.5)
    ax.set_title("Generator / Discriminator Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右上：分项 loss（未乘 lambda 的原始量级）
    ax = axes[0, 1]
    ax.plot(epochs, history["l1"], label="L1", linewidth=1.5)
    ax.plot(epochs, history["adv"], label="adv", linewidth=1.5)
    ax.plot(epochs, history["fm"], label="FM", linewidth=1.5)
    ax.set_title("Loss Components (raw, unweighted)")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 左下：delta 幅度
    ax = axes[1, 0]
    ax.plot(epochs, history["delta"], color="tab:orange", linewidth=1.5)
    ax.set_title("Delta Magnitude (|output - input|)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mean |δ|")
    ax.grid(True, alpha=0.3)

    # 右下：学习率
    ax = axes[1, 1]
    ax.plot(epochs, history["lr_g"], label="lr_g", linewidth=1.5)
    ax.plot(epochs, history["lr_d"], label="lr_d", linewidth=1.5)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved: {save_path}")


def train(
    epochs=120,
    batch_size=32,
    lr_g=1e-3,
    lr_d=5e-4,
    lambda_l1=2.0,
    lambda_adv=2.0,
    lambda_fm=2.0,
    exclude_names=None,
    device=DEFAULT_DEVICE,
    amp=False,
):
    dataset = PairedMelDataset(DATA_DIR, exclude_names=exclude_names)

    # CUDA 优化
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # AMP 混合精度（仅 CUDA）
    use_amp = amp and device == "cuda"
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 数据已在内存中，num_workers=0 避免进程间序列化开销
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # Generator (Conformer) + Discriminator (PatchGAN)
    gen = HachimiConformer(n_mels=N_MELS).to(device)
    disc = PatchDiscriminator(in_ch=1, base_ch=32).to(device)

    opt_g = torch.optim.AdamW(gen.parameters(), lr=lr_g, weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=lr_d, weight_decay=1e-4)
    warmup_epochs = min(10, epochs // 10)
    sched_g = torch.optim.lr_scheduler.SequentialLR(opt_g, [
        torch.optim.lr_scheduler.LinearLR(opt_g, start_factor=0.01, total_iters=warmup_epochs),
        torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs - warmup_epochs, eta_min=1e-5),
    ], milestones=[warmup_epochs])
    sched_d = torch.optim.lr_scheduler.SequentialLR(opt_d, [
        torch.optim.lr_scheduler.LinearLR(opt_d, start_factor=0.01, total_iters=warmup_epochs),
        torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs - warmup_epochs, eta_min=1e-6),
    ], milestones=[warmup_epochs])
    criterion_l1 = nn.L1Loss()

    g_params = sum(p.numel() for p in gen.parameters())
    d_params = sum(p.numel() for p in disc.parameters())
    print(
        f"Training on {device}, {len(dataset)} samples, {epochs} epochs, batch={batch_size}"
    )
    print(f"Loss: {lambda_l1}*L1 + {lambda_adv}*adv + {lambda_fm}*FM  |  lr_g={lr_g}, lr_d={lr_d}")
    print(f"Generator params: {g_params:,}  Discriminator params: {d_params:,}")
    if use_amp:
        print("AMP 混合精度: 已启用")

    # 训练历史（用于绘图）
    history = {"g": [], "d": [], "l1": [], "adv": [], "fm": [], "delta": [], "lr_g": [], "lr_d": []}
    best_loss = float("inf")

    for epoch in tqdm(range(epochs), desc="Training"):
        gen.train()
        disc.train()
        total_g = total_d = total_l1 = total_adv = total_fm = total_delta = 0.0
        n_batches = 0

        for orig_mel, hach_mel in loader:
            orig_mel = orig_mel.to(device, non_blocking=True)
            hach_mel = hach_mel.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                fake_mel = gen(orig_mel)

            # 监控 delta 幅度（global residual: fake = orig + delta）
            with torch.no_grad():
                delta_mag = (fake_mel - orig_mel).abs().mean().item()

            # ── Train Discriminator ──
            opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss_d = gan_loss_d(disc, hach_mel, fake_mel)
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # ── Train Generator ──
            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss_adv, fake_feats = gan_loss_g(disc, fake_mel)
                loss_fm = feature_matching_loss(disc, hach_mel, fake_feats)
                loss_l1 = criterion_l1(fake_mel, hach_mel)
                loss_g = lambda_adv * loss_adv + lambda_l1 * loss_l1 + lambda_fm * loss_fm
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            total_g += loss_g.item()
            total_d += loss_d.item()
            total_l1 += loss_l1.item()
            total_adv += loss_adv.item()
            total_fm += loss_fm.item()
            total_delta += delta_mag
            n_batches += 1

        sched_g.step()
        sched_d.step()

        avg = {k: v / n_batches for k, v in zip(
            ["g", "d", "l1", "adv", "fm", "delta"],
            [total_g, total_d, total_l1, total_adv, total_fm, total_delta],
        )}
        for k, v in avg.items():
            history[k].append(v)
        history["lr_g"].append(sched_g.get_last_lr()[0])
        history["lr_d"].append(sched_d.get_last_lr()[0])

        # 精简日志：每 10 epoch 或首尾 epoch 打印一行
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            tqdm.write(
                f"  [{epoch+1:>3}/{epochs}]  "
                f"G={avg['g']:.3f}  D={avg['d']:.3f}  "
                f"L1={avg['l1']:.4f}  adv={avg['adv']:.3f}  fm={avg['fm']:.4f}  "
                f"δ={avg['delta']:.4f}"
            )

        # 用综合 generator loss 选最佳模型
        if avg["g"] < best_loss:
            best_loss = avg["g"]
            torch.save(gen.state_dict(), MODEL_DIR / "hachimi_unet_best.pt")

        # 每 50 epoch 保存中间曲线
        if (epoch + 1) % 50 == 0:
            plot_curves(history, MODEL_DIR / "training_curves.png")

    torch.save(gen.state_dict(), MODEL_DIR / "hachimi_unet_final.pt")
    torch.save(disc.state_dict(), MODEL_DIR / "hachimi_disc.pt")
    plot_curves(history, MODEL_DIR / "training_curves.png")
    print(f"Done. Best G_loss: {best_loss:.4f}")
    print(f"Models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-g", type=float, default=1e-3)
    parser.add_argument("--lr-d", type=float, default=5e-4)
    parser.add_argument("--lambda-l1", type=float, default=2.0)
    parser.add_argument("--lambda-adv", type=float, default=2.0)
    parser.add_argument("--lambda-fm", type=float, default=2.0)
    parser.add_argument(
        "--device", choices=["auto", "mps", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--amp", action="store_true", help="启用混合精度训练(仅CUDA)")
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
        lambda_fm=args.lambda_fm,
        exclude_names=args.exclude,
        device=selected_device,
        amp=args.amp,
    )
