from torch import nn
from Sequential2D import Sequential2D, FlatSequential2D, LinearSequential2D
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

    """
    Parameter Values:
        'num_blocks' - number of block rows (output)
        'batch_size' - number of samples in the mini-batch
        'block_size' - number of output features in each block (inhomogeneous: may be different for each block)
    
    Args:
        X (num_blocks, batch_size, block_size): The input features. Will pad input with 'None' blocks if necessary

    Returns:
        y (num_blocks, batch_size, block_size): The output features
    """
    def forward(self, X):
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

    """
    Parameter Values:
        'batch_size'   - number of samples in the mini-batch
        'num_features' - the total number of input features = total number of output features
    
    Args:
        X (batch_size, in_features): The input features. Will pad input with 0's if necessary

    Returns:
        y (batch_size, out_features): The output features
    """
    def forward(self, X):
        activations = self.activations if type(self.activations) is list else [self.activations] * len(self.sizes)

        for _ in range(self.num_iterations):
            X = self.sequential.forward(X)

            # do activation functions
            for i in range(len(self.sizes)):
                start = sum(self.sizes[:i])
                end = sum(self.sizes[:i + 1])
                X[:, start:end] = activations[i](X[:, start:end]) if activations[i] else X[:, start:end]

        return X


class LinearIterativeSequential2D(nn.Module):
    def __init__(
            self,
            sizes,
            bias=True,
            num_iterations=1,
            activations=F.relu,
            num_input_blocks=1,
            densities=1,
    ):
        super(LinearIterativeSequential2D, self).__init__()

        self.sizes = sizes
        self.num_iterations = num_iterations
        self.activations = activations

        self.sequential = LinearSequential2D(
            sizes,
            num_input_blocks=num_input_blocks,
            bias=bias,
            densities=densities,
        )

    """
    Parameter Values:
        'batch_size'   - number of samples in the mini-batch
        'num_features' - the total number of input features = total number of output features

    Args:
        X (batch_size, in_features): The input features. Will pad input with 0's if necessary

    Returns:
        y (batch_size, out_features): The output features
    """
    def forward(self, X):
        activations = self.activations if type(self.activations) is list else [self.activations] * len(self.sizes)

        for _ in range(self.num_iterations):
            X = self.sequential.forward(X)

            # do activation functions
            for i in range(len(self.sizes)):
                start = sum(self.sizes[:i])
                end = sum(self.sizes[:i + 1])
                X[:, start:end] = activations[i](X[:, start:end]) if activations[i] else X[:, start:end]

        return X
