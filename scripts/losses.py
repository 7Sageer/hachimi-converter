#!/usr/bin/env python3
"""Loss functions for mel spectrogram GAN training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gan_loss_d(disc, real, fake):
    """Discriminator loss: real → 1, fake → 0 (LSGAN)."""
    pred_real = disc(real)
    pred_fake = disc(fake.detach())
    loss_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
    loss_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
    return (loss_real + loss_fake) * 0.5


def gan_loss_g(disc, fake):
    """Generator adversarial loss: fool discriminator (LSGAN)."""
    pred_fake = disc(fake)
    return F.mse_loss(pred_fake, torch.ones_like(pred_fake))
