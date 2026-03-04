#!/usr/bin/env python3
"""U-Net + Bottleneck Attention + Gated Skip for hachimi style transfer.

基于原始 U-Net（效果已验证），增加两个针对性改进：
1. Bottleneck Self-Attention — 捕获长距离时序依赖（"哈基→米"）
2. Gated Skip Connection — 让模型可以选择性修改低频特征
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """两层 3×3 卷积 + BN + ReLU。"""

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


class BottleneckAttention(nn.Module):
    """在 U-Net bottleneck 位置沿时间轴做 self-attention。

    输入 (B, C, H, T_down)，将 C*H 投影到 attn_dim，
    沿 T_down 维做多头自注意力，捕获全局时序依赖。
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

        # 初始化 proj_out 为近零，让残差连接初期 ≈ identity
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        # x: (B, C, H, T)
        B, C, H, T = x.shape
        # 展平为 (B, T, C*H)
        x_flat = x.permute(0, 3, 1, 2).reshape(B, T, C * H)
        # 残差 attention
        h = self.norm(x_flat)
        h = self.proj_in(h)       # (B, T, attn_dim)
        h = self.encoder(h)
        h = self.proj_out(h)      # (B, T, C*H)
        x_flat = x_flat + h       # 残差连接
        # 还原为 (B, C, H, T)
        return x_flat.reshape(B, T, C, H).permute(0, 2, 3, 1)


class GatedSkip(nn.Module):
    """学习 skip connection 的门控：gate = sigmoid(conv(skip))。

    gate 可以学习在某些频率 bin 或时间位置"关闭"skip，
    让 decoder 自由修改这些区域（如低频基音、时序转折点）。
    """

    def __init__(self, channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        # 初始化 gate bias 为 +2，sigmoid(2)≈0.88，默认大部分特征通过
        # 这保持了与原始 U-Net 相近的初始行为
        nn.init.zeros_(self.gate_conv[0].weight)
        nn.init.constant_(self.gate_conv[0].bias, 2.0)

    def forward(self, skip):
        return self.gate_conv(skip) * skip


class HachimiUNet(nn.Module):
    """U-Net + Bottleneck Attention + Gated Skip for mel spectrogram style transfer.

    Input/Output: (B, 1, 80, T) — 与原始 U-Net 接口完全兼容。
    """

    def __init__(self, n_mels=80, base_ch=32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(1, base_ch)          # (B, 32, 80, T)
        self.enc2 = ConvBlock(base_ch, base_ch * 2) # (B, 64, 40, T/2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)  # (B, 128, 20, T/4)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)  # (B, 256, 10, T/8)

        # Bottleneck Self-Attention（沿时间轴）
        bt_h = n_mels // 8  # 3次 MaxPool2d 后的 height = 80//8 = 10
        self.attention = BottleneckAttention(
            channels=base_ch * 8,  # 256
            height=bt_h,           # 10
            attn_dim=256,
            n_heads=4,
            num_layers=2,
            dropout=0.1,
        )

        # Gated Skip Connections
        self.gate1 = GatedSkip(base_ch)
        self.gate2 = GatedSkip(base_ch * 2)
        self.gate3 = GatedSkip(base_ch * 4)

        # Decoder — upsample + conv 避免棋盘伪影
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

        # Bottleneck Attention（全局时序建模）
        b = self.attention(b)

        # Decoder with Gated Skip Connections
        d3 = self.up3(b)
        d3 = self._pad_cat(d3, self.gate3(e3))
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_cat(d2, self.gate2(e2))
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._pad_cat(d1, self.gate1(e1))
        d1 = self.dec1(d1)

        return self.final(d1)

    def _pad_cat(self, x, skip):
        """Pad x to match skip's spatial dims, then concat."""
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        x = nn.functional.pad(x, [0, dw, 0, dh])
        return torch.cat([x, skip], dim=1)
