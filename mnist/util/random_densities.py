import torch
import random


def sparse_random(shape, p_random=None):
    if p_random is None:
        p_random = 0.99 * random.random() + 0.01

    random_tensor = torch.rand(shape)
    mask = torch.rand(shape) < p_random

    base_tensor = torch.zeros(shape)
    base_tensor[mask] = random_tensor[mask]

    return base_tensor