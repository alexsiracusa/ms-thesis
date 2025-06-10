import torch
from torch import nn
from Sequential2D import Sequential2D, FlatSequential2D
import torch.nn.functional as F


class IterativeSequential2D(nn.Module):
    def __init__(
            self,
            blocks,
            num_iterations=1,
            activations=F.relu
    ):
        super(IterativeSequential2D, self).__init__()

        self.blocks = blocks
        self.num_iterations = num_iterations
        self.sequential = Sequential2D(blocks)
        self.activations = activations

    # pads input with 'None' blocks to be correct length
    def format_input(self, X):
        n = len(self.blocks)
        if isinstance(X, list):
            return X + [None] * (n - len(X))
        else:
            return [X] + [None] * (n - 1)

    """
    Parameter Values:
        'num_blocks' - number of block rows (output)
        'batch_size' - number of samples in the mini-batch
        'block_size' - number of output features in each block (inhomogeneous: may be different for each block)
    
    Args:
        X (num_blocks, batch_size, block_size): The input features

    Returns:
        y (num_blocks, batch_size, block_size): The output features
    """
    def forward(self, X):
        X = self.format_input(X)
        activations = self.activations if type(self.activations) is list else [self.activations] * len(self.blocks)

        for _ in range(self.num_iterations):
            X = self.sequential.forward(X)
            X = [
                activation(x) if activation is not None and x is not None else None
                for x, activation in zip(X, activations)
            ]

        return X


class FlatIterativeSequential2D(nn.Module):
    def __init__(
            self,
            blocks,
            sizes,
            num_iterations=1,
            activations=F.relu,
    ):
        super(FlatIterativeSequential2D, self).__init__()

        self.blocks = blocks
        self.sizes = sizes
        self.num_iterations = num_iterations
        self.sequential = FlatSequential2D(blocks, in_features=sizes, out_features=sizes)
        self.activations = activations

    # pads input with zeros to be the correct length
    def format_input(self, X):
        pad_amount = sum(self.sizes) - X.size(1)
        return F.pad(X, (0, pad_amount))

    """
    Parameter Values:
        'batch_size'   - number of samples in the mini-batch
        'num_features' - the total number of input features = total number of output features
    
    Args:
        X (batch_size, in_features): The input features

    Returns:
        y (batch_size, out_features): The output features
    """
    def forward(self, X):
        X = self.format_input(X)
        activations = self.activations if type(self.activations) is list else [self.activations] * len(self.blocks)

        for _ in range(self.num_iterations):
            X = self.sequential.forward(X)

            # do activation functions
            for i in range(len(self.sizes)):
                start = sum(self.sizes[:i])
                end = sum(self.sizes[:i + 1])
                X[:, start:end] = activations[i](X[:, start:end]) if activations[i] else X[:, start:end]

        return X
