import torch.nn.functional as F

from ..IterativeSequential2D import IterativeSequential2D, FlatIterativeSequential2D, LinearIterativeSequential2D
from ..RecurrentSequential2D import RecurrentSequential2D, FlatRecurrentSequential2D
from .build_blocks import build_blocks
from .build_activations import build_activations


"""
NOTE: See `build_blocks.py` for more detailed documentation on how the blocks and activation functions are determined
NOTE: See `IterativeSequential2D` and/or `RecurrentSequential2D` for more detailed on how the `flat` and 'recurrent`
      parameters affect the model's expected input and output shapes.

Args:
    sizes (N): A list of input feature sizes, one for each column of blocks.
    bias (bool): Whether the Sequential2D model should include a trainable bias
    
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
    
    type (str): The type of Sequential2D model: "standard", "flat", or "linear"
    recurrent (Boolean): Whether the Sequential2D model should take in sequences of inputs
    num_input_blocks (int): The number of rows (from the top) reserved for input blocks.
    num_output_blocks (int): The number of rows (from the bottom) reserved for output blocks.
    num_iterations (int):   How many times to iterate the sequential2d model on each input
    
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
        a 1000x1000 matrix would have different initial weights than 100 100x100 matrices. These differences can 
        have a very large affect on loss when training (flat_init=True is generally better).
        
        This is essentially always set to `True` when `type="linear` as we are literally initializing one big
        matrix in that case, and the actual value of `flat_init` has no effect.
    
Returns:
    model: A sequential2d model
"""
def build_sequential2d(
        sizes,
        bias=True,
        activations=F.relu,
        type='standard',
        recurrent=False,
        num_input_blocks=1,
        num_output_blocks=1,
        num_iterations=1,
        densities=None,
        flat_init=True,
):
    valid_types = ["standard", "flat", "linear"]
    if type not in valid_types:
        raise ValueError(f"Invalid type '{type}'. Must be one of: {valid_types}")

    activations = build_activations(len(sizes), num_input_blocks, num_output_blocks, activations)

    if type == 'linear':
        return LinearIterativeSequential2D(
            sizes,
            bias=bias,
            num_iterations=num_iterations,
            activations=activations,
            num_input_blocks=num_input_blocks,
            densities=densities,
        )

    blocks = build_blocks(
        sizes, sizes,
        bias=bias,
        num_input_blocks=num_input_blocks,
        densities=densities,
        flat_init=flat_init,
    )

    model_type = (
        IterativeSequential2D     if not recurrent and type == 'standard' else
        FlatIterativeSequential2D if not recurrent and type == 'flat' else
        RecurrentSequential2D     if recurrent and type == 'standard' else
        FlatRecurrentSequential2D if recurrent and type == 'flat'
        else None
    )

    return model_type(
        blocks=blocks,
        sizes=sizes,
        num_iterations=num_iterations,
        activations=activations,
    )
