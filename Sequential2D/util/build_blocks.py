import torch
import numpy as np
import math

from Sequential2D.MaskedLinear import MaskedLinear

"""
Args:
    in_features (N): A list of input feature sizes, one for each column of blocks.
    out_features (N): A list of output feature sizes, one for each row of blocks.
    bias (bool): whether or not to include a bias term.
    num_input_blocks (int): The number of rows (from the top) reserved for input blocks.

    densities (Union[List[List[float]], float]): 
        Defines the densities for each block[i, j] in blocks.
        
        - If a list:
            A 2D list of shape (N, N) specifying the density (from 0.0 to 1.0) for each linear layer 
            in the resulting `Sequential2D` block matrix where blocks[i][j] has density densities[i][j]. 
            (The first `num_input_blocks` rows are ignored as they are all set to torch.nn.Identity or None)
            
        - If a float:
            Sets all linear blocks to the specified density.
            
    weight_init: either 'standard', 'uniform', 'weighted'
        - standard:
            Initializes each linear blocks weights with PyTorch's default settings (kaiming uniform).
            This is sub-optimal when there are many small blocks, and generally should not be used.
            
        - uniform:
            Initializes each linear block's weights uniformly from (-bound, bound) where bound is equal
            to 1 / sqrt(sum(in_features)). This is the same as kaiming uniform initialization on a single
            large matrix, however when split up into many separate blocks each individual matrix has a much 
            smaller in_features[i] instead of the sum. 
            
        - weighted:
            Initializes each row of linear block's weights uniformly from (-bound, bound) where bound is
            equal to 1 / sqrt(sum(in_features * densities[row])). This is a similar concept to the `uniform`
            initialization above, however since each row can be sparse the effective amount of in_features
            is actually smaller as (1 - density)% get ignored on average. This initialization accounts for 
            this by calculating the "real" amount of in_features per row/block of output and using that instead.

Returns:
    blocks: A 2D array of blocks
    
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
        bias=True,
        num_input_blocks=1,
        densities=1,
        weight_init='uniform',
):
    # Valid parameters
    valid_weight_inits = ["standard", "uniform", "weighted"]
    if weight_init not in valid_weight_inits:
        raise ValueError(f"Invalid type '{weight_init}'. Must be one of: {valid_weight_inits}")

    # Build blocks
    blocks = np.empty((len(out_features), len(in_features)), dtype=object)

    for i in range(len(out_features)):
        for j in range(len(in_features)):
            if i < num_input_blocks:
                if i == j:
                    blocks[i, j] = torch.nn.Identity()
                else:
                    blocks[i, j] = None
            else:
                density = densities[i][j] if isinstance(densities, list) else densities

                # only have bias when i == j to ensure only one bias per row
                masked_linear = MaskedLinear.sparse_random(
                    in_features[j], out_features[i],
                    percent=density,
                    bias=i == j and bias
                )

                # Custom initialization for weights
                with torch.no_grad():
                    if weight_init == 'uniform':
                        bound = 1 / math.sqrt(sum(in_features))
                        masked_linear.linear.weight.data.uniform_(-bound, bound)

                    elif weight_init == 'weighted':
                        row_densities = densities[i] if isinstance(densities, list) else densities
                        bound = 1 / max(math.sqrt(sum(torch.tensor(in_features) * torch.tensor(row_densities))), 1e-6)
                        masked_linear.linear.weight.data.uniform_(-bound, bound)

                blocks[i, j] = masked_linear

    return blocks
