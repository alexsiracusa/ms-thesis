from mnist.util.sizes import input_sizes, output_sizes, sizes
from Sequential2D.util import build_sequential2d

def create_model(densities):
    return build_sequential2d(
        sizes,
        type='linear',
        num_input_blocks=len(input_sizes),
        num_output_blocks=len(output_sizes),
        num_iterations=4,
        densities=densities,
        weight_init='weighted',
    )



