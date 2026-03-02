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


def load_model(path=None, use_transformer=False, tf_layers=1):
    if path is None:
        path = MODEL_DIR / "hachimi_unet_best.pt"
    model = HachimiUNet(n_mels=N_MELS, base_ch=32, use_transformer=use_transformer, tf_layers=tf_layers)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_hifigan(device="cpu", vocoder_path=None):
    """Load HiFi-GAN generator. Uses finetuned model if available."""
    config_path = HIFIGAN_DIR / "config.json"

    # 优先级：指定路径 > 微调模型 > 预训练模型
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
    # Old checkpoint format — needs weights_only=False
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    generator.remove_weight_norm()
    print(f"Loaded vocoder: {ckpt_path.name}")
    return generator


def convert(input_path: str, output_path: str, duration: float = 15.0, offset: float = 0.0, vocoder_path=None, use_transformer=False, tf_layers=1):
    """Convert a segment of audio to hachimi style."""
    model = load_model(use_transformer=use_transformer, tf_layers=tf_layers)
    vocoder = load_hifigan(vocoder_path=vocoder_path)

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
    import argparse
    parser = argparse.ArgumentParser(description="Convert audio to hachimi style")
    parser.add_argument("input", help="Input wav file")
    parser.add_argument("output", help="Output wav file")
    parser.add_argument("duration", type=float, nargs="?", default=15.0)
    parser.add_argument("offset", type=float, nargs="?", default=0.0)
    parser.add_argument("--vocoder", type=str, default=None, help="Path to vocoder checkpoint")
    parser.add_argument("--transformer", action="store_true", help="Use transformer model")
    parser.add_argument("--tf-layers", type=int, default=1, help="Transformer layers")
    args = parser.parse_args()
    convert(args.input, args.output, args.duration, args.offset, vocoder_path=args.vocoder, use_transformer=args.transformer, tf_layers=args.tf_layers)
