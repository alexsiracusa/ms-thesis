import torch
import random

def sparse_random(shape):
    p_random = 0.99 * random.random() + 0.01
    print(f"p_random: {p_random}")

    random_tensor = torch.rand(shape)
    mask = torch.rand(shape) < p_random

    base_tensor = torch.zeros(shape)
    base_tensor[mask] = random_tensor[mask]

    return base_tensor