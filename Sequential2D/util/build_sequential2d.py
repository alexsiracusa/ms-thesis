import torch.nn.functional as F
import inspect

from Sequential2D.IterativeSequential2D import IterativeSequential2D, FlatIterativeSequential2D, LinearIterativeSequential2D
from Sequential2D.RecurrentSequential2D import RecurrentSequential2D, FlatRecurrentSequential2D
from Sequential2D.util.build_blocks import build_blocks
from Sequential2D.util.build_activations import build_activations


"""
Note: 
    - See `build_blocks.py` for more detailed documentation on how the blocks and activation functions are determined
    - See `IterativeSequential2D` and/or `RecurrentSequential2D` for more detailed on how the `flat` and 'recurrent`
      parameters affect the model's expected input and output shapes.
      
Args:
    sizes (N): A list of input feature sizes, one for each column of blocks.
    type (str): The type of Sequential2D model: "standard", "flat", or "linear"
    recurrent (Boolean): Whether the Sequential2D model should take in sequences of inputs
    
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
    
    bias (bool): Whether the Sequential2D model should include a trainable bias
    num_iterations (int):   How many times to iterate the sequential2d model on each input
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
    model: A sequential2d model
"""
def build_sequential2d(
        sizes,
        type='standard',
        recurrent=False,
        activations=F.relu,
        bias=True,
        num_iterations=1,
        num_input_blocks=1,
        num_output_blocks=1,
        densities=None,
        weight_init='uniform',
):
    # Valid parameters
    valid_types = ["standard", "flat", "linear"]
    if type not in valid_types:
        raise ValueError(f"Invalid type '{type}'. Must be one of: {valid_types}")

    valid_weight_inits = ["standard", "uniform", "weighted"]
    if weight_init not in valid_weight_inits:
        raise ValueError(f"Invalid type '{weight_init}'. Must be one of: {valid_weight_inits}")

    # build model components
    activations = build_activations(len(sizes), num_input_blocks, num_output_blocks, activations)
    blocks = build_blocks(
        sizes, sizes,
        bias=bias,
        num_input_blocks=num_input_blocks,
        densities=densities,
        weight_init=weight_init,
    )

    model_type = (
        IterativeSequential2D if not recurrent and type == 'standard' else
        RecurrentSequential2D if recurrent and type == 'standard' else

        FlatIterativeSequential2D if not recurrent and type == 'flat' else
        FlatRecurrentSequential2D if recurrent and type == 'flat' else

        LinearIterativeSequential2D if not recurrent and type == 'linear' else
        # TODO: Recurrent LinearSequential2D not implemented
        None
    )

    def create_model(model_type, **kwargs):
        if model_type is None:
            raise ValueError("Invalid model type")

        # Get valid constructor parameters
        sig = inspect.signature(model_type.__init__)
        valid_params = {k: v for k, v in kwargs.items() if k in sig.parameters}

        return model_type(**valid_params)

    return create_model(
        model_type=model_type,
        blocks=blocks,
        sizes=sizes,
        activations=activations,
        bias=bias,
        num_iterations=num_iterations,
        num_input_blocks=num_input_blocks,
        densities=densities,
        weighted_init=weight_init == 'weighted'
    )
