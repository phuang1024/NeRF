import json
from pathlib import Path

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from constants import *


class ImageDataset(Dataset):
    """
    Extracts all rays for each image.
    One ray per pixel.

    directory/
    |__ 0.jpg
    |__ 0.json
    ...
    """

    def __init__(self, directory):
        directory = Path(directory)

        # List of (path_to_image, metadata)
        self.images = []

        for file in directory.iterdir():
            if file.suffix == ".jpg" and (json_file := file.with_suffix(".json")).exists():
                meta = json.loads(json_file.read_text())
                self.images.append((file, meta))

    def __len__(self) -> int:
        length = 0
        for _, meta in self.images:
            length += meta["res"][0] * meta["res"][1]
        return length

    def __getitem__(self, idx):
        # Find corresponding ray of image.
        i = 0
        while True:
            image_path, meta = self.images[i]
            width, height = meta["res"]
            curr_size = width * height
            if idx < curr_size:
                break
            idx -= curr_size
            i += 1

        # Get ray
        px_y = idx // width
        px_x = idx % width
        ray = pixel_to_ray(width, height, meta["fov"], meta["rot"], px_x, px_y)
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        color = image[px_y, px_x] / 255

        return (torch.tensor(meta["loc"], dtype=torch.float32), ray), torch.tensor(color, dtype=torch.float32)


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def pixel_to_ray(width, height, fov_x, rot: np.ndarray, x, y):
    """
    Convert pixel on a camera setup to a normalized ray starting from the camera.
    :param width, height: Resolution of the image.
    :param fov_x: Field of view in radians in horizontal camera direction.
    :param rot: Camera quaternion rotation.
    :param x, y: Pixel coordinates.
    """
    # Get ray for camera facing in negative z direction.
    x_fac = np.interp(x, [0, width], [-1, 1])
    y_fac = np.interp(y, [0, height], [-1, 1])
    aspect = width / height
    x_near = np.tan(fov_x / 2) * x_fac
    y_near = np.tan(fov_x / 2) * y_fac / aspect
    ray = np.array([0, x_near, y_near, -1])

    # Rotate by camera's rotation.
    rot_conj = rot * np.array([1, -1, -1, -1])
    ray = quat_mult(quat_mult(rot, ray), rot_conj)
    ray = ray[1:]
    ray /= np.linalg.norm(ray)

    return ray


if __name__ == "__main__":
    """
    # Test pixel_to_ray

    width = height = 100
    fov = np.pi / 2
    loc = np.array([0, 0, 0])
    # From -Z to +Y
    rot = np.array([0.707, 0.707, 0, 0])

    print(pixel_to_ray(width, height, fov, loc, rot, 00, 00))
    """

    # Test ImageDataset
    dataset = ImageDataset(Path("data/RubberDuck"))
    print(len(dataset))
    print(dataset[0])
