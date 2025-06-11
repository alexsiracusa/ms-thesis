import torch
import numpy as np


class Sequential2D(torch.nn.Module):
    """
    A 2D version of the torch.nn.Sequential module.

    The idea is to have a simple class that takes a 2D list (or list like) of modules and then applies them in
    sequence.  This just does a few things:

    1)  Implements the "+" combiner (which is what it means for the sizes to match up).
    2)  Actually calls the modules and returns the result.
    3)  It is efficient, since if a block is None, then it just acts as if the block returns a 0-vector of the
        correct size.

    Also, it assumes that things like what parameters are in trainable in the model and how they are initialized
    are handled by the modules themselves.

    Args:
        blocks: A 2D list of lists of torch.nn.Module objects. If blocks[i][j] is None, then we assume that the
                output is just a 0-vector of the correct size.

    Examples:
        blocks = [[I,    None, None],
                  [f1,   None, None],
                  [None, f2,   None]]

        where:
            I = torch.Identity()
            f1 = torch.nn.Linear(x, h)
            f2 = torch.nn.Linear(h, y)
    """
    def __init__(self, blocks):
        super(Sequential2D, self).__init__()

        # ensure correct dimensions of blocks
        blocks = np.array(blocks, dtype=object)
        assert blocks.ndim == 2, "Blocks must be a 2 dimensional array"
        assert blocks.shape[0] > 0 and blocks.shape[1] > 0, "Blocks must have no zero dimensions"
        self.blocks = blocks

        # This makes sure that each of the modules parameters are registered as parameters of the Sequential2D module.
        with torch.no_grad():
            self.module_dict = torch.nn.ModuleDict({
                f'{i},{j}' : value for (i, j), value in np.ndenumerate(blocks)
            })

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
        def safe_sum(arr):
            return sum(arr) if arr else None

        return [
            safe_sum([f.forward(x) for f, x in zip(row, X) if f is not None and x is not None])
            for row in self.blocks
        ]


class FlatSequential2D(torch.nn.Module):
    """
    Args:
        blocks: A 2D list of lists of torch.nn.Module objects. The blocks[i][j] is the block that takes
                in_features[i] features and outputs out_features[j] features.  If blocks[i][j] is None,
                then we assume that the output is just a 0-vector of the correct size.
        in_features: A list containing the number of input features for each column in blocks
        out_features: A list containing the number of output features for each row in blocks
    """
    def __init__(
            self,
            blocks,
            in_features: list,
            out_features: list
    ):
        super(FlatSequential2D, self).__init__()
        self.blocks = blocks
        self.in_features = in_features
        self.out_features = out_features

        self.sequential = Sequential2D(blocks)

    """
    Parameter Values:
        'batch_size'   - number of samples in the mini-batch
        'in_features'  - the total number of input features = sum(self.in_features)
        'out_features' - the total number of output features = sum(self.out_features)
        
    Args:
        X (batch_size, in_features): The input features

    Returns:
        y (batch_size, out_features): The output features
    """
    def forward(self, X):
        in_blocks = [
            X[:, sum(self.in_features[:i]):sum(self.in_features[:i + 1])].clone()
            for i in range(len(self.in_features))
        ]

        out = self.sequential(in_blocks)                      # (num_blocks, batch_size, block_size)
        out = list(zip(*out))                                 # (batch_size, num_blocks, block_size)
        out = [torch.cat(tensors, dim=0) for tensors in out]  # (batch_size, out_features)
        out = torch.stack(out)                                # (batch_size, out_features)
        return out


# class LinearSequential2D(torch.nn.Module):
#
#     def __init__(
#             self,
#             in_features: list,
#             out_features: list,
#             num_input_blocks=1,
#             num_output_blocks=1,
#             densities=1,
#     ):
#         super(LinearSequential2D, self).__init__()
#
#         self.in_features = in_features
#         self.out_features = out_features
#




