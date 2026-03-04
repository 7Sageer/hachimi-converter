#!/usr/bin/env python3
"""Flow Matching U-Net for hachimi style transfer.

基于 HachimiUNet 架构，增加时间步条件注入（FiLM），
用于 Conditional Flow Matching 训练。

输入：(B, 1, 80, T) 插值 mel + 标量时间步 t ∈ [0, 1]
输出：(B, 1, 80, T) 速度场 v(x_t, t)
"""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """正弦位置编码，将标量 t ∈ [0,1] 映射到高维向量。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) or (B, 1)
        t = t.view(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb  # (B, dim)


class TimeConditionedConvBlock(nn.Module):
    """两层 3×3 卷积 + BN + ReLU，通过 FiLM 注入时间条件。

    FiLM: scale, shift = MLP(t_emb)
    output = scale * BN(conv(x)) + shift
    """

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        # FiLM: 时间条件 → (scale, shift) for each conv layer
        self.film1 = nn.Linear(time_dim, out_ch * 2)
        self.film2 = nn.Linear(time_dim, out_ch * 2)

    def forward(self, x, t_emb):
        # Conv 1 + FiLM
        h = self.conv1(x)
        h = self.bn1(h)
        scale1, shift1 = self.film1(t_emb).chunk(2, dim=-1)
        h = h * (1 + scale1[:, :, None, None]) + shift1[:, :, None, None]
        h = self.act(h)

        # Conv 2 + FiLM
        h = self.conv2(h)
        h = self.bn2(h)
        scale2, shift2 = self.film2(t_emb).chunk(2, dim=-1)
        h = h * (1 + scale2[:, :, None, None]) + shift2[:, :, None, None]
        h = self.act(h)

        return h


class BottleneckAttention(nn.Module):
    """在 U-Net bottleneck 位置沿时间轴做 self-attention。

    与 model.py 中的版本相同，捕获全局时序依赖。
    """

    def __init__(self, channels, height, attn_dim=256, n_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        full_dim = channels * height
        self.channels = channels
        self.height = height

        self.norm = nn.LayerNorm(full_dim)
        self.proj_in = nn.Linear(full_dim, attn_dim)
        self.proj_out = nn.Linear(attn_dim, full_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim, nhead=n_heads,
            dim_feedforward=attn_dim * 4,
            dropout=dropout, batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        B, C, H, T = x.shape
        x_flat = x.permute(0, 3, 1, 2).reshape(B, T, C * H)
        h = self.norm(x_flat)
        h = self.proj_in(h)
        h = self.encoder(h)
        h = self.proj_out(h)
        x_flat = x_flat + h
        return x_flat.reshape(B, T, C, H).permute(0, 2, 3, 1)


class GatedSkip(nn.Module):
    """学习 skip connection 的门控。与 model.py 中的版本相同。"""

    def __init__(self, channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate_conv[0].weight)
        nn.init.constant_(self.gate_conv[0].bias, 2.0)

    def forward(self, skip):
        return self.gate_conv(skip) * skip


class HachimiFlowNet(nn.Module):
    """Flow Matching U-Net：时间条件 + FiLM 注入。

    结构与 HachimiUNet 一致（3 级编码器-解码器 + BottleneckAttention + GatedSkip），
    但通过 FiLM 在每个 ConvBlock 中注入时间步信息。

    Input: x_t (B, 1, 80, T), t (B,) 或 (B, 1)
    Output: v(x_t, t) (B, 1, 80, T) — 速度场
    """

    def __init__(self, n_mels=80, base_ch=32, time_dim=128):
        super().__init__()
        self.time_dim = time_dim

        # 时间步编码
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Encoder（带时间条件）
        self.enc1 = TimeConditionedConvBlock(1, base_ch, time_dim)
        self.enc2 = TimeConditionedConvBlock(base_ch, base_ch * 2, time_dim)
        self.enc3 = TimeConditionedConvBlock(base_ch * 2, base_ch * 4, time_dim)

        # Bottleneck（带时间条件）
        self.bottleneck = TimeConditionedConvBlock(base_ch * 4, base_ch * 8, time_dim)

        # Bottleneck Self-Attention
        bt_h = n_mels // 8
        self.attention = BottleneckAttention(
            channels=base_ch * 8,
            height=bt_h,
            attn_dim=256,
            n_heads=4,
            num_layers=2,
            dropout=0.1,
        )

        # Gated Skip Connections
        self.gate1 = GatedSkip(base_ch)
        self.gate2 = GatedSkip(base_ch * 2)
        self.gate3 = GatedSkip(base_ch * 4)

        # Decoder（带时间条件）
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch * 8, base_ch * 4, 3, padding=1),
        )
        self.dec3 = TimeConditionedConvBlock(base_ch * 8, base_ch * 4, time_dim)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
        )
        self.dec2 = TimeConditionedConvBlock(base_ch * 4, base_ch * 2, time_dim)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
        )
        self.dec1 = TimeConditionedConvBlock(base_ch * 2, base_ch, time_dim)

        # 输出层：无激活函数，速度场可以是任意值
        self.final = nn.Conv2d(base_ch, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        """
        Args:
            x: (B, 1, n_mels, T) — 插值后的 mel x_t
            t: (B,) 或 (B, 1) — 时间步 ∈ [0, 1]
        Returns:
            v: (B, 1, n_mels, T) — 预测的速度场
        """
        # 时间步编码
        t_emb = self.time_embed(t.view(-1))  # (B, time_dim)

        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)

        # Bottleneck
        b = self.bottleneck(self.pool(e3), t_emb)
        b = self.attention(b)

        # Decoder with Gated Skip
        d3 = self.up3(b)
        d3 = self._pad_cat(d3, self.gate3(e3))
        d3 = self.dec3(d3, t_emb)

        d2 = self.up2(d3)
        d2 = self._pad_cat(d2, self.gate2(e2))
        d2 = self.dec2(d2, t_emb)

        d1 = self.up1(d2)
        d1 = self._pad_cat(d1, self.gate1(e1))
        d1 = self.dec1(d1, t_emb)

        return self.final(d1)

    def _pad_cat(self, x, skip):
        """Pad x to match skip's spatial dims, then concat."""
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        x = nn.functional.pad(x, [0, dw, 0, dh])
        return torch.cat([x, skip], dim=1)
