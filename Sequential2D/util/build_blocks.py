import torch.nn.functional as F
import torch
import numpy as np

# from Sequential2D import MaskedLinear
from ..MaskedLinear import MaskedLinear

"""
Args:
    in_features (N): A list of input feature sizes, one for each column of blocks.
    out_features (N): A list of output feature sizes, one for each row of blocks.
    
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

    densities (Union[List[List[float]], float]): 
        Defines the densities for each block[i, j] in blocks.
        
        - If a list:
            A 2D list of shape (N, N) specifying the density (from 0.0 to 1.0) for each linear layer 
            in the resulting `Sequential2D` block matrix where blocks[i][j] has density densities[i][j]. 
            (The first `num_input_blocks` rows are ignored as they are all set to torch.nn.Identity or None)
            
        - If a float:
            Sets all linear blocks to the specified density.
            
    flat_init (Boolean): 
        Whether or not to initialize each block's weights separately or as one big matrix.  This is important
        as the default PyTorch init for nn.Linear samples from U(-√k, √k) where k = 1 / in_features.  Thus
        a 1000x1000 matrix would have different initial weights than 100 100x100 matrices. (We are often using 
        many smaller matrices instead of one large one so that we can assign different weight densities to each).
        
        This can have a very large affect on loss when training.

Returns:
    blocks: A 2D array of blocks
    activations: A list of activation functions for each row in `blocks`
    
    Example:
    
          Blocks                                Activations
       [[I      None   None   ...    None]    [None            } Input space = `num_input_blocks`
        [None   I      None   ...    ... ]     ...             }
        [...    ...    I      ...    None]     None            } 
        [f      f      f      ...    f   ]     F.relu          } Hidden space
        [...    ...    ...    ...    ... ]     ...             }
        [f      f      f      ...    f   ]     F.relu          }
        [f      f      f      ...    f   ]     None            } Output space = `num_output_blocks`
        [...    ...    ...    ...    ... ]     ...             } 
        [f      f      f      ...    f   ]]    None]           }

        where the first `num_input_blocks` rows is an identity map for the input blocks, and everything else is a 
        MaskedLinear block with densities determined the 'densities' parameter.
"""
def build_blocks(
        in_features,
        out_features,
        activations=F.relu,
        num_input_blocks=1,
        num_output_blocks=1,
        densities=1,
        flat_init=True,
):
    # Build blocks
    blocks = np.empty((len(out_features), len(in_features)), dtype=object)
    weights = torch.nn.Linear(sum(in_features), sum(out_features[num_input_blocks:])).weight.data

    for i in range(len(out_features)):
        for j in range(len(in_features)):
            if i < num_input_blocks:
                if i == j:
                    blocks[i, j] = torch.nn.Identity()
                else:
                    blocks[i, j] = None
            else:
                density = densities[i][j] if isinstance(densities, list) else densities
                # only have bias when i==j to ensure only one bias per row
                masked_linear = MaskedLinear.sparse_random(in_features[j], out_features[i], percent=density, bias=i == j)

                if flat_init:
                    with torch.no_grad():
                        masked_linear.linear.weight.data = weights[
                            sum(out_features[num_input_blocks:i]):sum(out_features[num_input_blocks:i+1]),
                            sum(in_features[:j]):sum(in_features[:j+1])
                        ]

                blocks[i, j] = masked_linear

    # Build activations
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
