import torch
import torch.nn.functional as F
from Sequential2D import MaskedLinear, SparseLinear, SparseLinearOld, IterativeSequential2D
import numpy as np



# CREATE MODELS

def masked_model(sizes, sparsity):
    blocks = np.empty((len(sizes), len(sizes)), dtype=object)
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            if i == 0 and j == 0:
                blocks[i, j] = torch.nn.Identity()
            elif i == 0:
                blocks[i, j] = None
            else:
                blocks[i, j] = MaskedLinear.sparse_random(sizes[j], sizes[i], percent=sparsity)

    return IterativeSequential2D(blocks, len(sizes), F.relu)


def sparse_model(sizes, sparsity):
    blocks = np.empty((len(sizes), len(sizes)), dtype=object)

    for i in range(len(sizes)):
        for j in range(len(sizes)):
            if i == 0 and j == 0:
                blocks[i, j] = torch.nn.Identity()
            elif i == 0:
                blocks[i, j] = None
            else:
                blocks[i, j] = SparseLinear.sparse_random(sizes[j], sizes[i], percent=sparsity)

    return IterativeSequential2D(blocks, len(sizes), F.relu)


def old_sparse_model(sizes, sparsity, device):
    blocks = np.empty((len(sizes), len(sizes)), dtype=object)

    for i in range(len(sizes)):
        for j in range(len(sizes)):
            if i == 0 and j == 0:
                blocks[i, j] = torch.nn.Identity()
            elif i == 0:
                blocks[i, j] = None
            else:
                blocks[i, j] = SparseLinearOld.sparse_random(sizes[j], sizes[i], percent=sparsity, device=device)

    return IterativeSequential2D(blocks, len(sizes), F.relu)