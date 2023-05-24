import math

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

from constants import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SineEmbedding(nn.Module):
    """
    Encodes each value with sine with exponentially increasing frequencies.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        embed = torch.zeros((*x.shape, EMBED_DIM), device=x.device)
        freqs = torch.exp(torch.linspace(math.log(FREQ_MIN), math.log(FREQ_MAX), EMBED_DIM))
        for i in range(EMBED_DIM):
            embed[..., i] = torch.sin(x * freqs[i])
        return embed


class NeRF(nn.Module):
    """
    Input: EMBED_DIM * d_input
    Output: 4 (x, y, z, density)
    """

    def __init__(self, d_input):
        """
        :param d_input: Number of values going in. E.g. x, y, z: d_input = 3
        """
        super().__init__()

        self.embedding = SineEmbedding()

        layers = []
        for i in range(MLP_DEPTH):
            in_dim = EMBED_DIM * d_input if i == 0 else MLP_DIM
            out_dim = 4 if i == MLP_DEPTH - 1 else MLP_DIM
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.mlp(x)
        return x


def renderer(nerf: NeRF, loc, ray, clipping, steps):
    """
    Use volume rendering, from loc for distance clipping.
    """
    # Get samples at intervals.
    samples = torch.empty(steps, 4, device=DEVICE)
    step_ray = ray / np.linalg.norm(ray) * clipping / steps
    curr_loc = loc
    for i in range(steps):
        curr_loc = curr_loc + step_ray
        samples[i] = nerf(curr_loc)
    color = samples[:, :3]
    density = samples[:, 3]
    density_cum = torch.exp(-torch.cumsum(density, dim=0))

    result = torch.zeros(3, device=DEVICE)
    for i in range(steps):
        result += density_cum[i] * density[i] * color[i]

    return result


if __name__ == "__main__":
    embedding = SineEmbedding()
    data = torch.linspace(0, 1, 1000)
    embeds = embedding(data).detach().numpy()
    embeds = np.interp(embeds, [np.min(embeds), np.max(embeds)], [0, 1])
    plt.imshow(embeds.T, aspect="auto", cmap="gray")
    plt.savefig("embed.png")
