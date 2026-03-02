#!/usr/bin/env python3
"""微调 HiFi-GAN vocoder，用哈基米音频数据适配 U-Net 输出的 mel 分布。"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset

from hifigan_model import Generator, AttrDict
from mel_utils import mel_spectrogram, log_mel, SR, HOP_SIZE

MODEL_DIR = Path(__file__).parent.parent / "models"
HIFIGAN_DIR = MODEL_DIR / "hifigan"
DATA_DIR = Path(__file__).parent.parent / "data" / "paired"

SEGMENT_SIZE = 8192  # ~0.37s 随机裁剪


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HachimiWavDataset(Dataset):
    """加载哈基米音频片段，返回 (log_mel, waveform) 对。预加载到内存避免 IO 瓶颈。"""

    def __init__(self, data_dir: Path, segment_size=SEGMENT_SIZE):
        self.segment_size = segment_size
        files = sorted(data_dir.glob("*_hach_*.wav"))
        print(f"Found {len(files)} hachimi segments, preloading to memory...")

        def _load(f):
            wav, sr = torchaudio.load(f)
            if sr != SR:
                wav = torchaudio.functional.resample(wav, sr, SR)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav.squeeze(0)

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as pool:
            self.wavs = list(pool.map(_load, files))
        print(f"Preloaded {len(self.wavs)} segments")

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        wav = self.wavs[idx]

        # 随机裁剪 segment_size 样本
        if wav.shape[0] >= self.segment_size:
            start = torch.randint(0, wav.shape[0] - self.segment_size, (1,)).item()
            wav = wav[start:start + self.segment_size]
        else:
            wav = nn.functional.pad(wav, (0, self.segment_size - wav.shape[0]))

        # 计算 log-mel（HiFi-GAN 输入格式）
        mel = log_mel(mel_spectrogram(wav))  # (1, 80, T_mel)
        return mel.squeeze(0), wav  # (80, T_mel), (T,)


def train(epochs=50, batch_size=16, lr=2e-4, device=None, amp=False):
    if device is None:
        device = get_device()

    use_amp = amp and device == "cuda"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── 加载 Generator（预训练权重）──
    with open(HIFIGAN_DIR / "config.json") as f:
        h = AttrDict(json.load(f))

    generator = Generator(h).to(device)
    ckpt = torch.load(HIFIGAN_DIR / "generator_v1", map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["generator"])
    generator.train()
    print(f"Loaded pretrained HiFi-GAN generator")

    # ── 判别器（bigvgan MPD + MRD）──
    from bigvgan.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
    from bigvgan.loss import discriminator_loss, generator_loss, feature_loss

    disc_cfg = AttrDict({
        "mpd_reshapes": [2, 3, 5, 7, 11],
        "discriminator_channel_mult": 1,
        "use_spectral_norm": False,
        "resolutions": [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]],
    })
    mpd = MultiPeriodDiscriminator(disc_cfg).to(device)
    mrd = MultiResolutionDiscriminator(disc_cfg).to(device)

    # ── 数据 ──
    dataset = HachimiWavDataset(DATA_DIR)
    # 数据已预加载到内存，无需多进程；macOS 上 num_workers>0 容易死锁
    num_workers = 0 if device == "mps" else max(2, min(8, (os.cpu_count() or 4) - 2))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        persistent_workers=num_workers > 0,
    )

    # ── 优化器 ──
    opt_g = torch.optim.AdamW(generator.parameters(), lr=lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(mrd.parameters()), lr=lr, betas=(0.8, 0.99)
    )
    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.999)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=0.999)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in mpd.parameters()) + sum(p.numel() for p in mrd.parameters())
    print(f"Training on {device}, {len(dataset)} samples, {epochs} epochs, batch={batch_size}")
    print(f"Generator: {g_params:,} params  Discriminators: {d_params:,} params")
    if use_amp:
        print("AMP 混合精度: 已启用")

    best_mel_loss = float("inf")
    l1_loss_fn = nn.L1Loss()
    total_steps = len(loader)

    for epoch in range(epochs):
        generator.train()
        mpd.train()
        mrd.train()
        sum_g = sum_d = sum_mel = 0
        n = 0

        for step, (mel_in, wav_real) in enumerate(loader):
            mel_in = mel_in.to(device, non_blocking=True)
            wav_real = wav_real.to(device, non_blocking=True).unsqueeze(1)  # (B, 1, T)

            # ── Generator forward ──
            with torch.amp.autocast("cuda", enabled=use_amp):
                wav_fake = generator(mel_in)  # (B, 1, T)

            # 对齐长度
            min_len = min(wav_fake.shape[-1], wav_real.shape[-1])
            wav_fake = wav_fake[..., :min_len]
            wav_real = wav_real[..., :min_len]

            # ── Train Discriminator ──
            opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                # MPD
                mpd_real, mpd_fake, _, _ = mpd(wav_real, wav_fake.detach())
                loss_mpd, _, _ = discriminator_loss(mpd_real, mpd_fake)
                # MRD
                mrd_real, mrd_fake, _, _ = mrd(wav_real, wav_fake.detach())
                loss_mrd, _, _ = discriminator_loss(mrd_real, mrd_fake)
                loss_d = loss_mpd + loss_mrd

            scaler.scale(loss_d).backward()
            scaler.step(opt_d)
            scaler.update()

            # ── Train Generator ──
            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Mel reconstruction loss
                mel_fake = log_mel(mel_spectrogram(wav_fake.squeeze(1)))
                mel_real = log_mel(mel_spectrogram(wav_real.squeeze(1)))
                loss_mel = l1_loss_fn(mel_fake, mel_real) * 45

                # Adversarial + feature matching
                mpd_real, mpd_fake, mpd_fmap_r, mpd_fmap_g = mpd(wav_real, wav_fake)
                mrd_real, mrd_fake, mrd_fmap_r, mrd_fmap_g = mrd(wav_real, wav_fake)

                loss_gen_mpd, _ = generator_loss(mpd_fake)
                loss_gen_mrd, _ = generator_loss(mrd_fake)
                loss_fm_mpd = feature_loss(mpd_fmap_r, mpd_fmap_g)
                loss_fm_mrd = feature_loss(mrd_fmap_r, mrd_fmap_g)

                loss_g = loss_gen_mpd + loss_gen_mrd + (loss_fm_mpd + loss_fm_mrd) * 2 + loss_mel

            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            scaler.update()

            sum_g += loss_g.item()
            sum_d += loss_d.item()
            sum_mel += loss_mel.item() / 45  # 记录未加权的 mel loss
            n += 1

            if step % 20 == 0:
                print(f"  [{epoch+1}/{epochs}] step {step}/{total_steps}  G={loss_g.item():.3f} D={loss_d.item():.3f} mel={loss_mel.item()/45:.4f}")

        sched_g.step()
        sched_d.step()

        avg_mel = sum_mel / n
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs}  G={sum_g/n:.4f} D={sum_d/n:.4f} mel={avg_mel:.4f}")

        if avg_mel < best_mel_loss:
            best_mel_loss = avg_mel
            torch.save(
                {"generator": generator.state_dict()},
                HIFIGAN_DIR / "generator_v1_finetuned",
            )

    # 保存 final
    torch.save(
        {"generator": generator.state_dict()},
        HIFIGAN_DIR / "generator_v1_finetuned_final",
    )
    print(f"Done. Best mel loss: {best_mel_loss:.4f}")
    print(f"Saved to {HIFIGAN_DIR}/generator_v1_finetuned")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="微调 HiFi-GAN vocoder")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--amp", action="store_true", help="启用混合精度(仅CUDA)")
    args = parser.parse_args()

    dev = None if args.device == "auto" else args.device
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=dev, amp=args.amp)
