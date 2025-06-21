import random
import torch
import numpy as np
from perlin_noise import PerlinNoise


def sparse_random(shape):
    p_random = random.random()
    print(f"p_random: {p_random}")

    random_tensor = torch.rand(shape)
    mask = torch.rand(shape) < p_random

    base_tensor = torch.zeros(shape)
    base_tensor[mask] = random_tensor[mask]

    return base_tensor


def sparse_perlin(shape):
    densities = generate_perlin_noise_2d(shape, square=True)
    densities = _normalize((densities - 0.33).clip(0, 1))

    return densities


def generate_perlin_noise_2d(shape, octaves=8, square=True):
    rows, cols = shape
    noise = PerlinNoise(octaves=octaves)

    if square:
        dim = max(rows, cols)
        noise_array = np.array([[noise([i / dim, j / dim]) for j in range(dim)] for i in range(dim)])
        noise_array = noise_array[:rows, :cols]
    else:
        noise_array = np.array([[noise([i/cols, j/rows]) for j in range(cols)] for i in range(rows)])

    normalized_noise = _normalize(noise_array)
    return normalized_noise


def _normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    shape = (45, 145)
    plt.axis('off')

    densities = sparse_random(shape)
    plt.imshow(densities, cmap='gray')
    plt.savefig('../images/sparse_random.png', bbox_inches='tight', pad_inches=0)

    noise = generate_perlin_noise_2d(shape)
    plt.imshow(noise, cmap='gray')
    plt.savefig('../images/perlin.png', bbox_inches='tight', pad_inches=0)

    noise = _normalize((noise - 0.33).clip(0, 1))
    plt.imshow(noise, cmap='gray')
    plt.savefig('../images/sparse_perlin.png', bbox_inches='tight', pad_inches=0)


