from torch import nn
from .Sequential2D import Sequential2D
import torch.nn.functional as F


class IterativeSequential2D(nn.Module):
    def __init__(self, blocks, num_iterations, activation=F.relu):
        super(IterativeSequential2D, self).__init__()
        self.num_iterations = num_iterations
        self.sequential = Sequential2D(blocks)
        self.activation = activation

    def format_input(self, X):
        n = len(self.sequential.blocks)
        if isinstance(X, list):
            return X + [None] * (n - len(X))
        else:
            return [X] + [None] * (n - 1)

    def forward(self, X):
        X = self.format_input(X)

        for _ in range(self.num_iterations):
            X = self.sequential.forward(X)
            X = [self.activation(x) for x in X]

        return X