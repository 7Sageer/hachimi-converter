#!/usr/bin/env python3
"""Shared mel spectrogram utilities aligned with HiFi-GAN UNIVERSAL_V1 parameters."""

import warnings
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

# ── Audio constants (match HiFi-GAN UNIVERSAL_V1) ──
SR = 22050
N_FFT = 1024
HOP_SIZE = 256
WIN_SIZE = 1024
N_MELS = 80
FMIN = 0
FMAX = 8000

# ── Mel basis cache ──
_mel_basis_cache = {}
_hann_cache = {}


def _get_mel_basis(sr, n_fft, n_mels, fmin, fmax, device):
    key = f"{sr}_{n_fft}_{n_mels}_{fmin}_{fmax}_{device}"
    if key not in _mel_basis_cache:
        mel_np = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        _mel_basis_cache[key] = torch.from_numpy(mel_np).float().to(device)
    return _mel_basis_cache[key]


def _get_hann_window(win_size, device):
    key = f"{win_size}_{device}"
    if key not in _hann_cache:
        _hann_cache[key] = torch.hann_window(win_size).to(device)
    return _hann_cache[key]


def mel_spectrogram(y, sr=SR, n_fft=N_FFT, hop_size=HOP_SIZE, win_size=WIN_SIZE,
                    n_mels=N_MELS, fmin=FMIN, fmax=FMAX, center=False):
    """Compute mel spectrogram matching HiFi-GAN's pipeline.

    Args:
        y: (B, T) or (T,) waveform tensor
    Returns:
        (B, n_mels, frames) mel spectrogram (linear scale, NOT log)
    """
    if y.dim() == 1:
        y = y.unsqueeze(0)

    # Pad signal so STFT frames align with HiFi-GAN expectations
    pad_amount = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y, (pad_amount, pad_amount), mode="reflect")

    hann = _get_hann_window(win_size, y.device)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
                      window=hann, center=center, pad_mode="reflect",
                      normalized=False, onesided=True, return_complex=True)
    spec = torch.abs(spec)  # (B, n_fft//2+1, frames)

    mel_basis = _get_mel_basis(sr, n_fft, n_mels, fmin, fmax, y.device)
    mel = torch.matmul(mel_basis, spec)
    return mel


def log_mel(mel, clip_val=1e-5):
    """Convert linear mel to log scale: log(clamp(mel))."""
    return torch.log(torch.clamp(mel, min=clip_val))


def normalize(log_mel_spec):
    """Normalize log-mel to roughly [0, 1] range for U-Net input."""
    return (log_mel_spec + 5.0) / 6.5


def denormalize(norm_mel):
    """Inverse of normalize — returns log-mel scale."""
    return norm_mel * 6.5 - 5.0


def exp_mel(log_mel_spec):
    """Convert log-mel back to linear scale for HiFi-GAN input."""
    return torch.exp(log_mel_spec)
