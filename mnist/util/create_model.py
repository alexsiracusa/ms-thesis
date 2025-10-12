from mnist.util.sizes import num_output, num_input, sizes, num_iterations
from Sequential2D.util import build_sequential2d

def create_model(densities, output_size=None):
    if output_size is not None:
        sizes[-1] = output_size

    return build_sequential2d(
        sizes,
        type='linear',
        num_input_blocks=num_input,
        num_output_blocks=num_output,
        num_iterations=num_iterations,
        densities=densities,
        weight_init='weighted',
    )

