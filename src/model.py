import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torch.nn as nn

from constants import *
from dataset import pixel_to_ray

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
            if i != MLP_DEPTH - 1:
                layers.append(nn.LeakyReLU())
        # TODO add sigmoid
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x


def render_ray(nerf: NeRF, loc, ray, clipping, steps):
    """
    Use volume rendering, from loc for distance clipping.
    """
    # Get samples at intervals.
    step_ray = ray / torch.norm(ray) * clipping / steps
    model_input = torch.empty(steps, 3, device=DEVICE, dtype=torch.float32)
    for i in range(steps):
        loc = loc + step_ray
        model_input[i] = loc
    samples = nerf(model_input)
    color = samples[:, :3]
    density = samples[:, 3]
    density_cum = torch.exp(-torch.cumsum(density, dim=0))

    # Integrate
    result = torch.zeros(3, device=DEVICE, dtype=torch.float32)
    for i in range(steps):
        result += density_cum[i] * density[i] * color[i]

    return result


def render_image(nerf: NeRF, loc, rot, fov, resolution: tuple[int, int]):
    """
    :param loc, rot, fov, resolution: Camera parameters. See dataset.py/pixel_to_ray
    """
    image = torch.zeros((*resolution, 3), device=DEVICE, dtype=torch.float32)
    for x in range(resolution[0]):
        for y in range(resolution[1]):
            ray = pixel_to_ray(*resolution, fov, rot, x, y)
            ray = torch.tensor(ray, device=DEVICE, dtype=torch.float32)
            image[y, x] = render_ray(nerf, loc, ray, CLIPPING, RENDER_STEPS)
    return image


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def dummy_nerf(x):
    """
    Unit cube.
    """
    output = torch.empty(x.size(0), 4, device=DEVICE, dtype=torch.float32)
    for i in range(output.size(0)):
        in_cube = torch.all(-0.1 <= x[i]) and torch.all(x[i] <= 0.1)
        output[i] = torch.tensor([1, 0, 0, 1]) if in_cube else torch.tensor([0, 0, 0, 0])
    return output


if __name__ == "__main__":
    """
    # Test sine embeddings
    embedding = SineEmbedding()
    data = torch.linspace(0, 1, 1000)
    embeds = embedding(data).detach().numpy()
    embeds = np.interp(embeds, [np.min(embeds), np.max(embeds)], [0, 1])
    plt.imshow(embeds.T, aspect="auto", cmap="gray")
    plt.savefig("embed.png")
    """

    # Test render image
    nerf = NeRF(3).to(DEVICE)
    nerf.apply(init_weights)
    loc = torch.tensor([0, 0, 3], device=DEVICE, dtype=torch.float32)
    rot = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
    image = render_image(nerf, loc, rot, math.radians(60), (64, 64))
    image = image.detach().cpu().numpy()
    image = np.clip(image*255, 0, 255).astype(np.uint8)
    cv2.imwrite("image.png", image)
