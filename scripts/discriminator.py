#!/usr/bin/env python3
"""PatchGAN discriminator for mel spectrogram adversarial training.

Classifies local patches of mel spectrograms as real or generated.
Small receptive field forces the generator to produce realistic local
temporal dynamics rather than smooth averages.
"""

import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """2D PatchGAN discriminator operating on mel spectrograms.

    Input: (B, 1, 80, T) — single-channel mel spectrogram.
    Output: (B, 1, H', T') — per-patch real/fake logits.

    Uses 4 conv layers with stride-2 to get ~16-frame receptive field
    (~186ms at hop=256, sr=22050), matching the temporal scale where
    the over-smoothing is most audible.
    """

    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            # (B, 1, 80, T) -> (B, 32, 40, T/2)
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (B, 64, 20, T/4)
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (B, 128, 10, T/8)
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (B, 1, 10, T/8) — per-patch logit
            nn.Conv2d(base_ch * 4, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)
