import torch
from torch import nn
from Sequential2D import Sequential2D
import torch.nn.functional as F


class IterativeSequential2D(nn.Module):
    def __init__(self, blocks, num_iterations, activations=F.relu):
        super(IterativeSequential2D, self).__init__()
        self.blocks = blocks
        self.num_iterations = num_iterations
        self.sequential = Sequential2D(blocks)
        self.activations = activations

    def format_input(self, X):
        n = len(self.blocks)
        if isinstance(X, list):
            return X + [None] * (n - len(X))
        else:
            return [X] + [None] * (n - 1)

    def forward(self, X):
        X = self.format_input(X)
        activations = self.activations if type(self.activations) is list else [self.activations] * len(self.blocks)

        for _ in range(self.num_iterations):
            X = self.sequential.forward(X)
            X = [activation(x) if activation is not None else x if torch.is_tensor(x) else None for x, activation in zip(X, activations)]

        return X[-1]
