import torch
from torch import nn
from Sequential2D import Sequential2D, FlatSequential2D
import torch.nn.functional as F


class IterativeSequential2D(nn.Module):
    def __init__(
            self,
            blocks,
            num_iterations,
            activations=F.relu,
            return_last=True
    ):
        super(IterativeSequential2D, self).__init__()

        self.blocks = blocks
        self.num_iterations = num_iterations
        self.sequential = Sequential2D(blocks)
        self.activations = activations
        self.return_last = return_last

    # pads input with 'None' blocks to be correct length
    def format_input(self, X):
        n = len(self.blocks)
        if isinstance(X, list):
            return X + [None] * (n - len(X))
        else:
            return [X] + [None] * (n - 1)

    """
    Args:
        X: The input features of shape:
            (num_blocks, batch_size, block_size)
              'num_blocks' - number of block columns (input)
              'batch_size' - number of samples in the mini-batch
              'block_size' - number of input features in each block (inhomogeneous: may be different for each block)

    Returns:
        y: The output features of shape:
            (num_blocks, batch_size, block_size)
              'num_blocks' - number of block rows (output)
              'batch_size' - number of samples in the mini-batch
              'block_size' - number of output features in each block (inhomogeneous: may be different for each block)
    """
    def forward(self, X):
        X = self.format_input(X)
        activations = self.activations if type(self.activations) is list else [self.activations] * len(self.blocks)

        for _ in range(self.num_iterations):
            X = self.sequential.forward(X)
            X = [activation(x) if activation is not None else x if torch.is_tensor(x) else None for x, activation in zip(X, activations)]

        return X[-1] if self.return_last else X


class FlatIterativeSequential2D(nn.Module):
    def __init__(
            self,
            blocks,
            sizes,
            num_iterations,
            activations=F.relu,
    ):
        super(FlatIterativeSequential2D, self).__init__()

        self.blocks = blocks
        self.sizes = sizes
        self.num_iterations = num_iterations
        self.sequential = FlatSequential2D(blocks, block_in_features=sizes, block_out_features=sizes)
        self.activations = activations

    # pads input with zeros to be the correct length
    def format_input(self, X):
        pad_amount = sum(self.sizes) - X.size(1)
        return F.pad(X, (0, pad_amount))

    """
    Args:
        X: The input features of shape:
            (batch_size, in_features)
              'batch_size'  - number of samples in the mini-batch
              'num_features' - the total number of input features = total number of output features

    Returns:
        y: The output features of shape:
            (batch_size, out_features)
              'batch_size'   - number of samples in the mini-batch
              'num_features' - the total number of output features = total number of input features
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
                X[:, start:end] = activations[i](X[:, start:end])

        return X
