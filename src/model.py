import math

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

from constants import *


class SineEmbedding(nn.Module):
    """
    Encodes each value with sine with exponentially increasing frequencies.
    """

    def __init__(self, freq_min=1, freq_max=256, embed_dim=256):
        super().__init__()
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.embed_dim = embed_dim

    def forward(self, x):
        embed = torch.zeros((*x.shape, self.embed_dim), device=x.device)
        freqs = torch.exp(torch.linspace(math.log(self.freq_min), math.log(self.freq_max), self.embed_dim))
        for i in range(self.embed_dim):
            embed[..., i] = torch.sin(x * freqs[i])
        return embed


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


if __name__ == "__main__":
    embedding = SineEmbedding()
    data = torch.linspace(0, 1, 1000)
    embeds = embedding(data).detach().numpy()
    embeds = np.interp(embeds, [np.min(embeds), np.max(embeds)], [0, 1])
    plt.imshow(embeds.T, aspect="auto", cmap="gray")
    plt.savefig("embed.png")
