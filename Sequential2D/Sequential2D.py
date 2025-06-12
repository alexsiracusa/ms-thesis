import torch
import numpy as np
import torch.nn.functional as F

from .MaskedLinear import MaskedLinear


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
                f'{i},{j}': value for (i, j), value in np.ndenumerate(blocks)
            })

    # pads input with 'None' blocks to be correct length
    def format_input(self, X):
        n = len(self.blocks[0])
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
        X (num_blocks, batch_size, block_size): The input features. Will pad input with 'None' blocks if necessary
                        
    Returns:
        y (num_blocks, batch_size, block_size): The output features      
    """
    def forward(self, X):
        X = self.format_input(X)

        def safe_sum(arr):
            return sum(arr) if arr else None

        return [
            safe_sum([f.forward(x) for f, x in zip(row, X) if f is not None and x is not None])
            for row in self.blocks
        ]


class FlatSequential2D(torch.nn.Module):
    """
    A version of Sequential2D that takes in a single flat tensor for all blocks, and instead of requiring
    them to already be seperated in a list. FlatSequential2D takes in an X of size (batch_size, in_features)
    instead of (num_blocks, batch_size, block_size) like in Sequential2D.

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

    # pads input with zeros to be the correct length
    def format_input(self, X):
        pad_amount = sum(self.in_features) - X.size(1)
        return F.pad(X, (0, pad_amount))

    """
    Parameter Values:
        'batch_size'   - number of samples in the mini-batch
        'in_features'  - the total number of input features = sum(self.in_features)
        'out_features' - the total number of output features = sum(self.out_features)
        
    Args:
        X (batch_size, in_features): The input features. Will pad input with 0's if necessary

    Returns:
        y (batch_size, out_features): The output features
    """
    def forward(self, X):
        X = self.format_input(X)

        in_blocks = [
            X[:, sum(self.in_features[:i]):sum(self.in_features[:i + 1])].clone()
            for i in range(len(self.in_features))
        ]

        out = self.sequential(in_blocks)                      # (num_blocks, batch_size, block_size)
        out = list(zip(*out))                                 # (batch_size, num_blocks, block_size)
        out = [torch.cat(tensors, dim=0) for tensors in out]  # (batch_size, out_features)
        out = torch.stack(out)                                # (batch_size, out_features)
        return out


class LinearSequential2D(torch.nn.Module):
    """
    An optimized purely linear version of FlatSequential2D by combining all linear blocks together into one
    large tensor for efficiency. It also allows for different densities for each block to be specified

    Args:
        sizes (N): A list of the sizes for each linear block such that blocks[i][j] has shape (sizes[i], sizes[j])
        num_input_blocks: The number of blocks reserved for input features.

        densities (Union[List[List[float]], float]):
            Defines the densities for each block[i, j] in blocks.

            - If a list:
                A 2D list of shape (N, N) specifying the density (from 0.0 to 1.0) for each linear layer
                in the resulting `Sequential2D` block matrix where blocks[i][j] has density densities[i][j].
                (The first `num_input_blocks` rows are ignored as they are all set to torch.Identity or None)

            - If a float:
                Sets all linear blocks to the specified density.

    Example:
        Although this does not implement 'blocks' the same way as the other Sequential2D modules, it can still
        be thought of the same way as a special case as seen below.

        sizes = [a, b, c, d, e]
        num_input_blocks = 2

        Blocks:
          a      b      c      d      e        Size
        [[I      None   None   None   None ]   a      } Input space = `num_input_blocks`
         [None   I      None   None   None ]   b      }
         [w      w      w      w      w    ]   c
         [w      w      w      w      w    ]   d
         [w      w      w      w      w    ]]  e

        Where the first `num_input_blocks` rows serve as an identity map for the first `num_input_blocks` blocks
        given, and the remaining rows are a single sparse matrix with varying densities as determined by the
        `densities` parameter.
    """
    def __init__(
            self,
            sizes,
            bias=True,
            num_input_blocks=1,
            densities=1,
    ):
        super(LinearSequential2D, self).__init__()

        self.sizes = sizes
        self.num_input_blocks = num_input_blocks

        self.linear = MaskedLinear.variable_random(
            sizes,
            sizes[num_input_blocks:],
            bias=bias,
            densities=densities[num_input_blocks:] if isinstance(densities, list) else densities
        )

    # pads input with zeros to be the correct length
    def format_input(self, X):
        pad_amount = sum(self.sizes) - X.size(1)
        return F.pad(X, (0, pad_amount))

    """
    Parameter Values:
        'batch_size'   - number of samples in the mini-batch
        'in_features'  - the total number of input features = sum(self.in_features)
        'out_features' - the total number of output features = sum(self.out_features)

    Args:
        X (batch_size, in_features): The input features. Will pad input with 0's if necessary

    Returns:
        y (batch_size, out_features): The output features
    """
    def forward(self, X):
        X = self.format_input(X)
        hidden = self.linear(X)
        input_size = sum(self.sizes[:self.num_input_blocks])
        return torch.cat((X[..., :input_size], hidden), dim=1)




