#!/usr/bin/env python3
"""Train hachimi style transfer with Conditional Flow Matching.

Flow Matching 训练极其简洁：
1. 在 source_mel 和 target_mel 之间随机插值得到 x_t
2. 网络预测速度场 v(x_t, t)
3. 目标速度就是 target - source（直线路径）
4. Loss = MSE(v_pred, v_target)

没有判别器，没有对抗训练，没有多个 loss 权重需要调节。
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from mel_utils import N_MELS, log_mel, mel_spectrogram, normalize
from model_fm import HachimiFlowNet
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data" / "paired"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class PairedMelDataset(Dataset):
    """Load paired original/hachimi segments as mel spectrograms.

    与 train.py 中完全相同的数据加载逻辑。
    所有 mel 在 __init__ 时预计算并缓存在内存中。
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


def build_freq_weight(n_mels=80, low_weight=3.0, mid_weight=1.5, device="cpu"):
    """构建频率权重向量：低频加权，迫使模型关注人声/鼓点。

    mel bin 分布（SR=22050, fmax=8000）:
    - bin 0-30:  ~0-1500 Hz  (bass, kick drum, male vocals) → weight=3.0
    - bin 30-55: ~1500-4000 Hz (vocals, snare, harmonics) → weight=1.5
    - bin 55-80: ~4000-8000 Hz (hi-hat, sibilance, air) → weight=1.0
    """
    w = torch.ones(n_mels, device=device)
    w[:30] = low_weight
    w[30:55] = mid_weight
    # 归一化使平均权重 = 1（不改变 loss 量级）
    w = w / w.mean()
    return w.view(1, 1, n_mels, 1)  # 广播到 (B, 1, n_mels, T)


def plot_curves(history, save_path):
    """绘制 Flow Matching 训练曲线。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Flow Matching Training Curves", fontsize=14, fontweight="bold")
    epochs = range(1, len(history["loss"]) + 1)

    # 左上：总 loss
    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], linewidth=1.5, color="tab:blue")
    ax.set_title("Weighted MSE Loss (total)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # 右上：低频 vs 高频 loss
    ax = axes[0, 1]
    if "loss_low" in history and history["loss_low"]:
        ax.plot(epochs, history["loss_low"], linewidth=1.5, label="Low (0-30)", color="tab:red")
        ax.plot(epochs, history["loss_high"], linewidth=1.5, label="High (55-80)", color="tab:cyan")
        ax.legend()
    ax.set_title("Freq-band MSE (unweighted)")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # 左下：delta magnitude
    ax = axes[1, 0]
    ax.plot(epochs, history["delta"], linewidth=1.5, color="tab:orange")
    ax.set_title("Delta Magnitude (|v_pred| mean)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mean |v|")
    ax.grid(True, alpha=0.3)

    # 右下：学习率
    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], linewidth=1.5, color="tab:green")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved: {save_path}")


def train(
    epochs=120,
    batch_size=32,
    lr=1e-3,
    exclude_names=None,
    device=DEFAULT_DEVICE,
    amp=False,
    resume=None,
):
    dataset = PairedMelDataset(DATA_DIR, exclude_names=exclude_names)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    model = HachimiFlowNet(n_mels=N_MELS).to(device)

    # 可选：从上一轮训练恢复
    if resume:
        ckpt_path = Path(resume)
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            print(f"Resumed from {ckpt_path}")
        else:
            print(f"Warning: {ckpt_path} not found, training from scratch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    warmup_epochs = min(10, epochs // 10)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-5),
    ], milestones=[warmup_epochs])

    # 频率权重：低频 3x，中频 1.5x
    freq_weight = build_freq_weight(N_MELS, device=device)
    print(f"Frequency weighting: low(0-30)=3.0x, mid(30-55)=1.5x, high(55-80)=1.0x")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training Flow Matching on {device}, {len(dataset)} samples, {epochs} epochs, batch={batch_size}")
    print(f"Model params: {n_params:,}  |  lr={lr}")
    if use_amp:
        print("AMP 混合精度: 已启用")

    history = {"loss": [], "loss_low": [], "loss_high": [], "delta": [], "lr": []}
    best_loss = float("inf")

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        total_loss = 0.0
        total_loss_low = 0.0
        total_loss_high = 0.0
        total_delta = 0.0
        n_batches = 0

        for orig_mel, hach_mel in loader:
            orig_mel = orig_mel.to(device, non_blocking=True)
            hach_mel = hach_mel.to(device, non_blocking=True)

            # ── Flow Matching 核心 ──
            B = orig_mel.shape[0]

            # 1. 随机采样时间步 t ∈ [0, 1]
            t = torch.rand(B, device=device)

            # 2. 线性插值：x_t = (1-t)*source + t*target
            t_expand = t[:, None, None, None]  # (B, 1, 1, 1)
            x_t = (1 - t_expand) * orig_mel + t_expand * hach_mel

            # 3. 目标速度：v = target - source（直线 OT 路径）
            v_target = hach_mel - orig_mel

            # 4. 前向 + 频率加权 loss
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                v_pred = model(x_t, t)
                # 频率加权 MSE：低频 bin 权重更高
                diff_sq = (v_pred - v_target) ** 2
                loss = (diff_sq * freq_weight).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 监控：分频段 loss（无权重）
            with torch.no_grad():
                delta_mag = v_pred.abs().mean().item()
                loss_low = diff_sq[:, :, :30, :].mean().item()
                loss_high = diff_sq[:, :, 55:, :].mean().item()

            total_loss += loss.item()
            total_loss_low += loss_low
            total_loss_high += loss_high
            total_delta += delta_mag
            n_batches += 1

        scheduler.step()

        avg_loss = total_loss / n_batches
        avg_loss_low = total_loss_low / n_batches
        avg_loss_high = total_loss_high / n_batches
        avg_delta = total_delta / n_batches
        current_lr = scheduler.get_last_lr()[0]

        history["loss"].append(avg_loss)
        history["loss_low"].append(avg_loss_low)
        history["loss_high"].append(avg_loss_high)
        history["delta"].append(avg_delta)
        history["lr"].append(current_lr)

        # 日志：每 10 epoch 或首尾打印
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            tqdm.write(
                f"  [{epoch+1:>3}/{epochs}]  "
                f"loss={avg_loss:.6f}  low={avg_loss_low:.6f}  high={avg_loss_high:.6f}  "
                f"|v|={avg_delta:.4f}  lr={current_lr:.2e}"
            )

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_DIR / "hachimi_fm_best.pt")

        # 每 50 epoch 保存中间曲线
        if (epoch + 1) % 50 == 0:
            plot_curves(history, MODEL_DIR / "training_curves_fm.png")

    torch.save(model.state_dict(), MODEL_DIR / "hachimi_fm_final.pt")
    plot_curves(history, MODEL_DIR / "training_curves_fm.png")
    print(f"Done. Best loss: {best_loss:.6f}")
    print(f"Models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Flow Matching for hachimi style transfer")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", choices=["auto", "mps", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--amp", action="store_true", help="启用混合精度训练(仅CUDA)")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练")
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
        amp=args.amp,
        resume=args.resume,
    )

