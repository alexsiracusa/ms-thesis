import torch.nn.functional as F
import torch
import numpy as np

from ..MaskedLinear import MaskedLinear


"""
Args:
    in_features (List[int]): A list of input feature sizes, one for each column of blocks.
    out_features (List[int]): A list of output feature sizes, one for each row of blocks.
    
    activations (Union[List[Optional[nn.Module]], nn.Module]):
        Defines the activation functions applied to each row of the output blocks.

        - If a list: 
            Should contain one activation function (or `None`) for each row. If the list is 
            shorter than `len(out_features)`, it will be padded with `None`.
        - If a single nn.Module: 
            The same activation will be applied to all rows, excluding the input 
            and output blocks as defined by `num_input_blocks` and `num_output_blocks`.

        Example:
            - `activations = [nn.ReLU(), None, nn.Tanh()]`  # Per-row specification
            - `activations = nn.ReLU()`  # Shared activation, will be skipped for I/O rows

    num_input_blocks (int): The number of rows (from the top) reserved for input blocks.
    num_output_blocks (int): The number of rows (from the bottom) reserved for output blocks.

    densities (Optional[List[List[float]]]): A 2D list of shape (N, N) specifying the density (from 0.0 to 1.0)
        for each linear layer in the resulting `Sequential2D` block matrix. Defaults to all 1.0 (fully dense).
        The first `num_input_blocks` rows are ignored as they are all set to torch.Identity or None.

Returns:
    blocks: A 2D array of blocks
    
    Example:
    
          Blocks                                Activations
        [[I      None   None   ...    None]    [None            } Input space = `num_input_blocks`
        [None   I      None   ...    None]      None            }
        [...    ...    I      ...    None]      None            } 
        [f      f      f      ...    f   ]      F.relu          } Hidden space
        [f      f      f      ...    f   ]      F.relu          }
        [...    ...    ...    ...    ... ]      ...             }
        [f      f      f      ...    f   ]      F.relu          }
        [f      f      f      ...    f   ]]     None            } Output space = `num_output_blocks`
        [...    ...    ...    ...    ... ]      ...             } 
        [f      f      f      ...    f   ]]     None]           }

        where the first 'num_input_blocks' rows is an identity map for the input blocks, and everything else is a 
        MaskedLinear block with densities determined the 'densities' parameter.
"""
def build_blocks(
        in_features,
        out_features,
        activations=F.relu,
        num_input_blocks=1,
        num_output_blocks=1,
        densities=None
):
    blocks = np.empty((len(out_features), len(in_features)), dtype=object)

    for i in range(len(out_features)):
        for j in range(len(in_features)):
            if i < num_input_blocks:
                if i == j:
                    blocks[i, j] = torch.nn.Identity()
                else:
                    blocks[i, j] = None
            else:
                density = densities[i][j] if densities is not None else 1
                blocks[i, j] = MaskedLinear.sparse_random(in_features[j], out_features[i], percent=density)

    if isinstance(activations, list):
        activations = activations + [None] * (len(out_features) - len(activations))
    elif activations is not None:
        activations = (
            [None] * num_input_blocks +
            [activations] * (len(out_features) - num_input_blocks - num_output_blocks) +
            [None] * num_output_blocks
        )
    else:
        activations = None

    return blocks, activations
