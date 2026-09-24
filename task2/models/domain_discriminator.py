"""binary domain prediction with gradient reversal."""

import math

import torch
from torch import nn


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, strength):
        ctx.strength = strength
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


def reverse(inputs, strength):
    return GradientReversal.apply(inputs, strength)


def schedule(progress):
    return 2 / (1 + math.exp(-10 * progress)) - 1


def make_discriminator(input_dim=512):
    return nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(256, 2))
