import torch.nn.functional as F

from ..IterativeSequential2D import IterativeSequential2D, FlatIterativeSequential2D
from ..RecurrentSequential2D import RecurrentSequential2D, FlatRecurrentSequential2D
from .build_blocks import build_blocks


"""
Args:
    sizes:                see `in_features` and `out_features` in `build_blocks.py`
    flat (Boolean):       whether the sequential2d model should take in a flat input vector or a list of blocks
    recurrent (Boolean):  whether the sequential2d model should take in sequences of inputs
    num_input_blocks:     see `build_blocks.py`
    num_output_blocks:    see `build_blocks.py`
    num_iterations (int): how many times to iterate the sequential2d model on each input
    densities:            see `build_blocks.py`
    
Returns:
    model: A sequential2d model
"""
def build_sequential2d(
        sizes,
        flat=False,
        recurrent=False,
        num_input_blocks=1,
        num_output_blocks=1,
        num_iterations=1,
        densities=None
):
    blocks, activations = build_blocks(
        sizes, sizes,
        activations=F.relu,
        num_input_blocks=num_input_blocks,
        num_output_blocks=num_output_blocks,
        densities=densities,
    )

    model_type = (
        IterativeSequential2D     if not recurrent and not flat else
        FlatIterativeSequential2D if not recurrent and flat else
        RecurrentSequential2D     if recurrent and not flat else
        FlatRecurrentSequential2D if recurrent and flat
        else None
    )

    return model_type(
        blocks=blocks,
        sizes=sizes,
        num_iterations=num_iterations,
        activations=activations,
    )
