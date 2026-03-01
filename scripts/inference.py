#!/usr/bin/env python3
"""Convert audio to hachimi style using trained U-Net."""

import json
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from model import HachimiUNet
from hifigan_model import Generator, AttrDict
from mel_utils import mel_spectrogram, log_mel, normalize, denormalize, SR, N_MELS

MODEL_DIR = Path(__file__).parent.parent / "models"
HIFIGAN_DIR = MODEL_DIR / "hifigan"


def load_model(path=None):
    if path is None:
        path = MODEL_DIR / "hachimi_unet_best.pt"
    model = HachimiUNet(n_mels=N_MELS, base_ch=32)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_hifigan(device="cpu"):
    """Load pretrained HiFi-GAN UNIVERSAL_V1 generator."""
    config_path = HIFIGAN_DIR / "config.json"
    ckpt_path = HIFIGAN_DIR / "generator_v1"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"HiFi-GAN checkpoint not found at {ckpt_path}\n"
            "Run: python scripts/download_vocoder.py"
        )

    with open(config_path) as f:
        h = AttrDict(json.load(f))

    generator = Generator(h).to(device)
    # Old checkpoint format — needs weights_only=False
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    generator.remove_weight_norm()
    return generator


def convert(input_path: str, output_path: str, duration: float = 15.0, offset: float = 0.0):
    """Convert a segment of audio to hachimi style."""
    model = load_model()
    vocoder = load_hifigan()

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

    # U-Net inference
    with torch.no_grad():
        pred = model(mel)

    # Denormalize to log-mel for HiFi-GAN, clamp to realistic range
    pred_log = denormalize(pred[:, :, :, :t]).clamp(-11.5, 0.5)  # (1, 1, n_mels, T)

    # HiFi-GAN vocoding (expects log-mel input, shape (B, 80, T))
    with torch.no_grad():
        audio_out = vocoder(pred_log.squeeze(0))  # (1, 1, samples)

    audio_out = audio_out.squeeze().clamp(-1, 1)

    # Save
    sf.write(output_path, audio_out.numpy(), SR)
    print(f"Saved: {output_path} ({audio_out.shape[-1]/SR:.1f}s)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python inference.py <input.wav> <output.wav> [duration_sec] [offset_sec]")
        sys.exit(1)
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    off = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    convert(sys.argv[1], sys.argv[2], dur, off)
