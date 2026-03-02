#!/usr/bin/env python3
"""Compare HiFi-GAN vs BigVGAN vocoder on U-Net output."""
import bigvgan, torch, torchaudio, numpy as np, sys, soundfile as sf
from scipy.signal import welch
from mel_utils import mel_spectrogram, log_mel, normalize, denormalize
from model import HachimiUNet
# Patch BigVGAN loading
_orig = bigvgan.BigVGAN._from_pretrained
@classmethod
def _patched(cls, **kwargs):
    kwargs.setdefault('proxies', None)
    kwargs.setdefault('resume_download', False)
    return _orig.__func__(cls, **kwargs)
bigvgan.BigVGAN._from_pretrained = _patched

# Load models
bvg = bigvgan.BigVGAN.from_pretrained(
    'nvidia/bigvgan_v2_22khz_80band_256x', use_cuda_kernel=False
)
bvg.remove_weight_norm()
bvg.eval()

gen = HachimiUNet(n_mels=80, base_ch=32)
gen.load_state_dict(torch.load('models/hachimi_unet_best.pt', map_location='cpu', weights_only=True))
gen.eval()

from pathlib import Path
import glob

# 用之前 HiFi-GAN 生成过的同一批歌，方便对比
target_names = ['Rubia', '命运', '快乐的基米', '半岛铁哈', '我的猫香',
                '哈在基米前', '甜甜的', '基战摇', 'Daisy Bell',
                '【哈基米】Seven-Tobu']
songs = [f'data/original/{n}.wav' for n in target_names
         if Path(f'data/original/{n}.wav').exists()]
out_dir = Path('output')

for song_path in songs:
    name = Path(song_path).stem
    print(f'\n[{name}]')

    wav, sr = torchaudio.load(song_path)
    if sr != 22050:
        wav = torchaudio.functional.resample(wav, sr, 22050)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav[:, :int(15 * 22050)]

    mel = normalize(log_mel(mel_spectrogram(wav.squeeze(0))))
    t = mel.shape[-1]
    pad_t = ((t + 7) // 8) * 8
    mel_pad = torch.nn.functional.pad(mel, [0, pad_t - t]).unsqueeze(0)

    with torch.no_grad():
        pred = gen(mel_pad)
    pred_log = denormalize(pred[:, :, :, :t]).clamp(-11.5, 0.5)
    vocoder_input = pred_log.squeeze(0)

    with torch.no_grad():
        wav_bvg = bvg(vocoder_input).squeeze().numpy()

    sf.write(str(out_dir / f'{name}_bigvgan.wav'), wav_bvg, 22050)

    f, psd = welch(wav_bvg, fs=22050, nperseg=4096)
    idx86 = np.argmin(np.abs(f - 86))
    bg = (f > 60) & (f < 120) & (np.abs(f - 86) > 10)
    print(f'  BigVGAN 86Hz={psd[idx86] / np.mean(psd[bg]):.2f}x')
    print(f'  Saved: {name}_bigvgan.wav')
