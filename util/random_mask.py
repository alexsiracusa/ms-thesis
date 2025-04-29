import torch


def random_boolean_tensor(rows, cols, num_true):
    total_elements = rows * cols
    values = torch.tensor([True] * num_true + [False] * (total_elements - num_true))
    shuffled_values = values[torch.randperm(total_elements)]

    return shuffled_values.view(rows, cols)
