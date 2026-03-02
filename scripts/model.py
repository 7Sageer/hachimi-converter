#!/usr/bin/env python3
"""Mel spectrogram U-Net for hachimi style transfer."""

import torch
import torch.nn as nn
import math


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class TemporalTransformer(nn.Module):
    """Lightweight temporal self-attention on bottleneck features.

    Projects C*H down to attn_dim before attention to keep params manageable.
    """

    def __init__(self, full_dim, attn_dim=256, nhead=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Linear(full_dim, attn_dim)
        self.proj_out = nn.Linear(attn_dim, full_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim, nhead=nhead,
            dim_feedforward=attn_dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, C, H, T) -> attend along T
        B, C, H, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * H)  # (B, T, C*H)
        x = self.proj_in(x)   # (B, T, attn_dim)
        x = self.encoder(x)
        x = self.proj_out(x)  # (B, T, C*H)
        x = x.reshape(B, T, C, H).permute(0, 2, 3, 1)  # (B, C, H, T)
        return x


class HachimiUNet(nn.Module):
    """Lightweight U-Net: mel spectrogram -> mel spectrogram."""

    def __init__(self, n_mels=80, base_ch=32, use_transformer=False, tf_layers=1):
        super().__init__()
        self.use_transformer = use_transformer

        # Encoder
        self.enc1 = ConvBlock(1, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

        # Temporal Transformer (on bottleneck output)
        if use_transformer:
            bt_h = n_mels // 8  # after 3 poolings
            self.temporal_tf = TemporalTransformer(
                full_dim=base_ch * 8 * bt_h,
                attn_dim=256, nhead=4,
                num_layers=tf_layers, dropout=0.15,
            )

        # Decoder — upsample + conv to avoid checkerboard artifacts
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch * 8, base_ch * 4, 3, padding=1),
        )
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
        )
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
        )
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.final = nn.Conv2d(base_ch, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        if self.use_transformer:
            b = b + self.temporal_tf(b)  # residual connection

        d3 = self.up3(b)
        d3 = self._pad_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._pad_cat(d1, e1)
        d1 = self.dec1(d1)

        return self.final(d1)

    def _pad_cat(self, x, skip):
        """Pad x to match skip's spatial dims, then concat."""
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        x = nn.functional.pad(x, [0, dw, 0, dh])
        return torch.cat([x, skip], dim=1)
