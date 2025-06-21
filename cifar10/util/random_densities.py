import random
import torch
import numpy as np
from perlin_noise import PerlinNoise


def sparse_random_densities(shape):
    p_random = 0.25 * random.random()
    print(f"p_random: {p_random}")

    random_tensor = torch.rand(shape)
    mask = torch.rand(shape) < p_random

    base_tensor = torch.zeros(shape)
    base_tensor[mask] = random_tensor[mask]

    return base_tensor


def perlin_densities(shape):
    densities = generate_perlin_noise_2d(shape)
    densities[densities < 0.5] = 0

    return densities


def generate_perlin_noise_2d(shape):
    width, height = shape

    noise = PerlinNoise(octaves=3)
    noise_array = np.array([[noise([i/width, j/height]) for j in range(width)] for i in range(height)])

    normalized_noise = normalize(noise_array)
    return normalized_noise


def normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    shape = (150, 150)

    densities = sparse_random_densities(shape)
    plt.imshow(densities, cmap='gray')
    plt.savefig('../images/densities.png')

    noise = generate_perlin_noise_2d(shape)
    plt.imshow(noise, cmap='gray')
    plt.savefig('../images/noise.png')

    noise[noise < 0.5] = 0
    plt.imshow(noise, cmap='gray')
    plt.savefig('../images/threshold.png')


