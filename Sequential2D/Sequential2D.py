import torch

import numpy as np


class Sequential2D(torch.nn.Module):
    """
    A 2D version of the torch.nn.Sequential module.

    The idea is to have a simple class that takes a 2D list (or list like) of modules and then applies them in sequence.  This just does a few things:

    1)  Implements the "+" combiner (which is what it means for the sizes to match up).
    2)  Actually calls the modules and returns the result.
    3)  It is efficient, since if a block is None, then it just acts as if the block returns a 0-vector of the correct size.

    Also, it assumes that things like what parameters are in trainable in the model and how they are initialized are handled by the modules themselves.

    Args:
        blocks: A list of lists of torch.nn.Module objects. The blocks[i][j] is the block that takes in_features_list[i] features and
                outputs out_features_list[j] features.  If blocks[i][j] is None, then we assume that the output is
                just a 0-vector of the correct size.

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
    Args:
        X: The input features of shape:
           (num_blocks, n_samples, block_dim)
              'num_blocks' - number of blocks
              'n_samples'  - number of samples in the mini-batch
              'block_dim'  - number of features in each block (may be different for each block)
                      
    Returns:
        y: The features for the next iteration (same shape as X)
    """
    def forward(self, X):
        return [
            sum([f.forward(x) for f, x in zip(row, X) if f is not None and x is not None])
            for row in self.blocks
        ]

