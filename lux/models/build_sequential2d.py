from Sequential2D import FlatRecurrentSequential2D, MaskedLinear
import torch.nn.functional as F
import torch
import numpy as np

"""
Args:
    sizes (N): list of integers representing the dimensions of each of the blocks
    num_input_blocks: int: How many of the first n blocks are for the input
    densities (N, N): corresponding densities (0 - 1) for the linear layers for each of the blocks in the result Sequential2D. Default is all 100% dense. The first 'num_input_blocks' rows are ignored.

Returns:
    model: A Sequential2D model of shape:

    [[I      None   None    ...    None]
     [None   I      None    ...    None]
     [...    ...    I       ...    None]
     [f      f      f       f      f   ]
     [f      f      f       f      f   ]
     [f      f      f       f      f   ]]

    where the first 'num_input_blocks' rows is an identity map for the input blocks, and everything else is a MaskedLinear block with densities determined the 'densities' parameter.

"""
def build_sequential2d(sizes, num_input_blocks=1, num_iterations=1, densities=None):
    blocks = np.empty((len(sizes), len(sizes)), dtype=object)

    for i in range(len(sizes)):
        for j in range(len(sizes)):
            if i < num_input_blocks:
                if i == j:
                    blocks[i, j] = torch.nn.Identity()
                else:
                    blocks[i, j] = None
            else:
                density = densities[i][j] if densities is not None else 1
                blocks[i, j] = MaskedLinear.sparse_random(sizes[j], sizes[i], percent=density)

    return FlatRecurrentSequential2D(
        blocks,
        sizes,
        num_iterations=num_iterations,
        activations=[F.relu] * (len(sizes) - 1) + [None]
    )
