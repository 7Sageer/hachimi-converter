#!/usr/bin/env python3
"""Convert audio to hachimi style using trained U-Net."""

import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path
from model import HachimiUNet

MODEL_DIR = Path(__file__).parent.parent / "models"
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP = 512


def load_model(path=None):
    if path is None:
        path = MODEL_DIR / "hachimi_unet_best.pt"
    model = HachimiUNet(n_mels=N_MELS, base_ch=16)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def convert(input_path: str, output_path: str, duration: float = 15.0):
    """Convert a segment of audio to hachimi style."""
    model = load_model()

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    # Load input audio
    wav, orig_sr = torchaudio.load(input_path)
    if orig_sr != SR:
        wav = torchaudio.functional.resample(wav, orig_sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Trim to duration
    max_samples = int(duration * SR)
    wav = wav[:, :max_samples]

    # To mel
    mel = amp_to_db(mel_transform(wav))  # (1, n_mels, T)
    mel = mel.unsqueeze(0)  # (1, 1, n_mels, T)

    # Normalize
    mel_norm = (mel + 40) / 40

    # Pad time to multiple of 8
    t = mel_norm.shape[-1]
    pad_t = ((t + 7) // 8) * 8
    if pad_t > t:
        mel_norm = torch.nn.functional.pad(mel_norm, [0, pad_t - t])

    # Inference
    with torch.no_grad():
        pred = model(mel_norm)

    # Denormalize
    pred_db = pred[:, :, :, :t] * 40 - 40  # back to dB scale
    pred_db = pred_db.squeeze(0)  # (1, n_mels, T)

    # Griffin-Lim to reconstruct audio from mel
    pred_linear = torchaudio.transforms.InverseMelScale(
        n_stft=N_FFT // 2 + 1, n_mels=N_MELS, sample_rate=SR
    )(10 ** (pred_db / 10))  # dB -> power -> linear

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=N_FFT, hop_length=HOP, power=1.0
    )
    audio_out = griffin_lim(pred_linear.sqrt())  # power -> magnitude

    # Save
    sf.write(output_path, audio_out.squeeze().numpy(), SR)
    print(f"Saved: {output_path} ({audio_out.shape[-1]/SR:.1f}s)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python inference.py <input.wav> <output.wav> [duration_sec]")
        sys.exit(1)
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    convert(sys.argv[1], sys.argv[2], dur)
