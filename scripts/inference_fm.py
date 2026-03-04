#!/usr/bin/env python3
"""Flow Matching 推理：使用 Euler ODE 求解器将原始音频转换为哈基米风格。

从 x_0 = source_mel 出发，沿预测的速度场积分到 x_1 ≈ target_mel。
默认 10 步 Euler 积分，可通过 --steps 调节。
"""

import json
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from model_fm import HachimiFlowNet
from hifigan_model import Generator, AttrDict
from mel_utils import mel_spectrogram, log_mel, normalize, denormalize, SR, N_MELS

MODEL_DIR = Path(__file__).parent.parent / "models"
HIFIGAN_DIR = MODEL_DIR / "hifigan"


def load_model(path=None, device="cpu"):
    if path is None:
        path = MODEL_DIR / "hachimi_fm_best.pt"
    model = HachimiFlowNet(n_mels=N_MELS)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model.to(device)


def load_hifigan(device="cpu", vocoder_path=None):
    """Load HiFi-GAN generator. 与 inference.py 中完全相同。"""
    config_path = HIFIGAN_DIR / "config.json"

    if vocoder_path is not None:
        ckpt_path = Path(vocoder_path)
    elif (HIFIGAN_DIR / "generator_v1_finetuned").exists():
        ckpt_path = HIFIGAN_DIR / "generator_v1_finetuned"
    else:
        ckpt_path = HIFIGAN_DIR / "generator_v1"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"HiFi-GAN checkpoint not found at {ckpt_path}\n"
            "Run: python scripts/download_vocoder.py"
        )

    with open(config_path) as f:
        h = AttrDict(json.load(f))

    generator = Generator(h).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    generator.remove_weight_norm()
    print(f"Loaded vocoder: {ckpt_path.name}")
    return generator


@torch.no_grad()
def euler_ode_solve(model, x_0, num_steps=10, device="cpu"):
    """Euler ODE 求解器：从 x_0 沿速度场积分到 x_1。

    x_{i+1} = x_i + v(x_i, t_i) * dt,  dt = 1/num_steps
    """
    dt = 1.0 / num_steps
    x = x_0.to(device)

    for i in range(num_steps):
        t = torch.full((x.shape[0],), i * dt, device=device)
        v = model(x, t)
        x = x + v * dt

    return x


def convert(input_path: str, output_path: str, duration: float = 15.0,
            offset: float = 0.0, vocoder_path=None, num_steps=10, device="cpu"):
    """Convert a segment of audio to hachimi style using Flow Matching."""
    model = load_model(device=device)
    vocoder = load_hifigan(device=device, vocoder_path=vocoder_path)

    # Load input audio
    wav, orig_sr = torchaudio.load(input_path)
    if orig_sr != SR:
        wav = torchaudio.functional.resample(wav, orig_sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Slice [offset, offset+duration]
    start = int(offset * SR)
    end = start + int(duration * SR)
    wav = wav[:, start:end]

    # To normalized log-mel
    mel = normalize(log_mel(mel_spectrogram(wav.squeeze(0))))  # (1, n_mels, T)
    mel = mel.unsqueeze(0)  # (1, 1, n_mels, T)

    # Pad time to multiple of 8
    t = mel.shape[-1]
    pad_t = ((t + 7) // 8) * 8
    if pad_t > t:
        mel = torch.nn.functional.pad(mel, [0, pad_t - t])

    # Flow Matching ODE inference
    print(f"Running {num_steps}-step Euler ODE...")
    pred = euler_ode_solve(model, mel, num_steps=num_steps, device=device)

    # Denormalize to log-mel for HiFi-GAN, clamp to realistic range
    pred_log = denormalize(pred[:, :, :, :t]).clamp(-11.5, 0.5)

    # HiFi-GAN vocoding
    with torch.no_grad():
        audio_out = vocoder(pred_log.squeeze(0))

    audio_out = audio_out.squeeze().cpu().clamp(-1, 1)

    # Save
    sf.write(output_path, audio_out.numpy(), SR)
    print(f"Saved: {output_path} ({audio_out.shape[-1]/SR:.1f}s)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert audio to hachimi style (Flow Matching)")
    parser.add_argument("input", help="Input wav file")
    parser.add_argument("output", help="Output wav file")
    parser.add_argument("duration", type=float, nargs="?", default=15.0)
    parser.add_argument("offset", type=float, nargs="?", default=0.0)
    parser.add_argument("--steps", type=int, default=10, help="ODE 求解步数 (default: 10)")
    parser.add_argument("--vocoder", type=str, default=None, help="Path to vocoder checkpoint")
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        dev = args.device

    convert(args.input, args.output, args.duration, args.offset,
            vocoder_path=args.vocoder, num_steps=args.steps, device=dev)
