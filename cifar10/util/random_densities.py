import random
import torch
import numpy as np
from noise import pnoise2


def random_densities(shape):
    p_random = 0.25 * random.random() + 0.01
    print(f"p_random: {p_random}")

    random_tensor = torch.rand(shape)
    mask = torch.rand(shape) < p_random

    base_tensor = torch.zeros(shape)
    base_tensor[mask] = random_tensor[mask]

    return base_tensor


def generate_perlin_noise_2d(shape, scale=100.0, octaves=1, persistence=0.5, lacunarity=2.0, base=0):
    """
    Generate a 2D NumPy array filled with Perlin noise.

    Args:
        shape: Tuple (height, width) for the output array shape.
        scale: Controls the "zoom" level of the noise.
        octaves: Number of layers of noise to combine.
        persistence: Amplitude decrease per octave.
        lacunarity: Frequency increase per octave.
        base: Base offset for noise generation.

    Returns:
        A NumPy array of shape `shape` filled with Perlin noise values.
    """
    height, width = shape
    noise_array = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            x = j / scale
            y = i / scale
            noise_val = pnoise2(
                x, y,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=base
            )
            noise_array[i][j] = noise_val

    return noise_array


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    densities = random_densities((150, 150))
    plt.imshow(densities, cmap='gray')
    plt.savefig('../images/densities.png')

    noise = generate_perlin_noise_2d((150, 150))
    plt.imshow(noise, cmap='gray')
    plt.savefig('../images/noise.png')


