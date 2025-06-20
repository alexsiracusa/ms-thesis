import random
import torch
import numpy as np
from noise import pnoise2
from perlin_noise import PerlinNoise


def random_densities(shape):
    p_random = 0.25 * random.random() + 0.01
    print(f"p_random: {p_random}")

    random_tensor = torch.rand(shape)
    mask = torch.rand(shape) < p_random

    base_tensor = torch.zeros(shape)
    base_tensor[mask] = random_tensor[mask]

    return base_tensor


def generate_perlin_noise_2d(shape):
    width, height = shape

    noise = PerlinNoise(octaves=3)
    noise_array = np.array([[noise([i/width, j/height]) for j in range(width)] for i in range(height)])

    normalized_noise = (noise_array + 1) / 2
    return normalized_noise


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    shape = (150, 150)

    densities = random_densities(shape)
    plt.imshow(densities, cmap='gray')
    plt.savefig('../images/densities.png')

    noise = generate_perlin_noise_2d(shape)
    plt.imshow(noise, cmap='gray')
    plt.savefig('../images/noise.png')


