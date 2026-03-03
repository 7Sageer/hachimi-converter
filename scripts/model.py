#!/usr/bin/env python3
"""Conformer Encoder-Decoder for hachimi style transfer.

替代 U-Net 架构：无 skip connection，强制模型学习完整的风格转换映射。
Conformer encoder 捕获长距离时序依赖，Transformer decoder 生成目标 mel。
"""

import math
import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    """正弦位置编码，支持任意长度序列。"""

    def __init__(self, d_model, max_len=8000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ConvModule(nn.Module):
    """Conformer 卷积模块：Pointwise → GLU → DepthwiseConv → BN → Swish → Pointwise → Dropout"""

    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Linear(d_model, d_model * 2)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        x = self.layer_norm(x)
        x = self.pointwise1(x)  # (B, T, 2*d_model)
        a, b = x.chunk(2, dim=-1)
        x = a * torch.sigmoid(b)  # 标准 GLU gate
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)  # (B, T, d_model)
        x = torch.nn.functional.silu(x)
        x = self.pointwise2(x)
        return self.dropout(x)


class ConformerBlock(nn.Module):
    """Macaron-style Conformer block: 0.5*FFN → MHSA → ConvModule → 0.5*FFN → LayerNorm"""

    def __init__(self, d_model=256, n_heads=4, ff_dim=1024, conv_kernel=31, dropout=0.1):
        super().__init__()
        # 前半 FFN
        self.ffn1_norm = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        # MHSA
        self.mhsa_norm = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mhsa_dropout = nn.Dropout(dropout)
        # ConvModule
        self.conv_module = ConvModule(d_model, conv_kernel, dropout)
        # 后半 FFN
        self.ffn2_norm = nn.LayerNorm(d_model)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        # 最终 LayerNorm
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        # 0.5 * FFN
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))
        # MHSA
        residual = x
        x_norm = self.mhsa_norm(x)
        attn_out, _ = self.mhsa(x_norm, x_norm, x_norm)
        x = residual + self.mhsa_dropout(attn_out)
        # ConvModule
        x = x + self.conv_module(x)
        # 0.5 * FFN
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        # Final LayerNorm
        return self.final_norm(x)


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block: Self-Attn + Cross-Attn + FFN（无因果掩码）。"""

    def __init__(self, d_model=256, n_heads=4, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_attn_dropout = nn.Dropout(dropout)

        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, memory):
        # Self-Attention（双向，无因果掩码）
        residual = x
        x_norm = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = residual + self.self_attn_dropout(attn_out)
        # Cross-Attention
        residual = x
        x_norm = self.cross_attn_norm(x)
        attn_out, _ = self.cross_attn(x_norm, memory, memory)
        x = residual + self.cross_attn_dropout(attn_out)
        # FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


class HachimiConformer(nn.Module):
    """Conformer Encoder-Decoder for mel-to-mel style transfer.

    Input/Output: (B, 1, 80, T) — 与 U-Net 接口完全兼容。
    """

    def __init__(
        self,
        n_mels=80,
        d_model=256,
        n_heads=4,
        ff_dim=1024,
        conv_kernel=31,
        n_enc=4,
        n_dec=4,
        dropout=0.1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model

        # 输入投影: mel bins → d_model
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)

        # Conformer Encoder
        self.encoder = nn.ModuleList([
            ConformerBlock(d_model, n_heads, ff_dim, conv_kernel, dropout)
            for _ in range(n_enc)
        ])

        # Transformer Decoder
        self.decoder_input_proj = nn.Linear(n_mels, d_model)
        self.decoder_pos_enc = SinusoidalPE(d_model, dropout=dropout)
        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_dec)
        ])

        # 输出投影: d_model → mel bins
        self.output_proj = nn.Linear(d_model, n_mels)

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化线性层。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        identity = x  # 保存输入用于全局残差
        B, _, M, T = x.shape

        # (B, 1, M, T) → (B, T, M)
        x_seq = x.squeeze(1).transpose(1, 2)

        # Encoder
        enc = self.input_proj(x_seq)  # (B, T, d_model)
        enc = self.pos_enc(enc)
        for block in self.encoder:
            enc = block(enc)

        # Decoder（输入也是原始 mel，让 decoder 通过 cross-attn 从 encoder 获取风格信息）
        dec = self.decoder_input_proj(x_seq)  # (B, T, d_model)
        dec = self.decoder_pos_enc(dec)
        for block in self.decoder:
            dec = block(dec, enc)

        # 输出投影 → delta
        delta = self.output_proj(dec)  # (B, T, M)
        delta = delta.transpose(1, 2).unsqueeze(1)  # (B, 1, M, T)

        # 全局残差：output = input + delta
        # Conformer 只学习风格差异，高频细节通过 identity 保留
        return identity + delta
