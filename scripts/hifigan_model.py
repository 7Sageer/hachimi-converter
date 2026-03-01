#!/usr/bin/env python3
"""Self-contained HiFi-GAN Generator for vocoding mel spectrograms to audio.

Extracted from jik876/hifi-gan (models.py + env.py), adapted for modern PyTorch.
Only includes Generator (inference) — no discriminators needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm
# Use legacy weight_norm for compatibility with jik876/hifi-gan pretrained checkpoints
# (parametrizations.weight_norm uses a different state dict format)
from torch.nn.utils import weight_norm


class AttrDict(dict):
    """Dict subclass that allows attribute-style access."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(nn.Module):
    """HiFi-GAN ResBlock type 1 — 3 dilated conv layers with skip connections."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=d,
                                  padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=1,
                                  padding=get_padding(kernel_size, 1)))
            for _ in dilation
        ])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class ResBlock2(nn.Module):
    """HiFi-GAN ResBlock type 2 — 2 dilated conv layers with skip connections."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=d,
                                  padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in self.convs:
            remove_weight_norm(c)


class Generator(nn.Module):
    """HiFi-GAN Generator — upsamples mel spectrogram to waveform."""

    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.conv_pre = weight_norm(
            nn.Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(ch * 2, ch, k, u, padding=(k - u) // 2)))

        ResBlock = ResBlock1 if h.resblock == "1" else ResBlock2

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
