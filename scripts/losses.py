#!/usr/bin/env python3
"""Loss functions for mel spectrogram GAN training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gan_loss_d(disc, real, fake):
    """Discriminator loss: real → 1, fake → 0 (LSGAN)."""
    pred_real, _ = disc(real)
    pred_fake, _ = disc(fake.detach())
    loss_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
    loss_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
    return (loss_real + loss_fake) * 0.5


def gan_loss_g(disc, fake):
    """Generator adversarial loss: fool discriminator (LSGAN).

    Returns (adv_loss, fake_feats) — fake_feats 用于 feature matching。
    """
    pred_fake, fake_feats = disc(fake)
    adv_loss = F.mse_loss(pred_fake, torch.ones_like(pred_fake))
    return adv_loss, fake_feats


def feature_matching_loss(disc, real, fake_feats):
    """Feature matching loss: 判别器中间层特征的 L1 距离。

    用真实样本的中间特征作为目标，让生成器的中间特征靠近。
    比纯 LSGAN 提供更稳定、更丰富的梯度信号。
    """
    with torch.no_grad():
        _, real_feats = disc(real)

    loss = 0.0
    for rf, ff in zip(real_feats, fake_feats):
        loss += F.l1_loss(ff, rf.detach())
    return loss / len(real_feats)
