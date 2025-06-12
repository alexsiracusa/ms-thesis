import torch.nn.functional as F


def build_activations(num_blocks, num_input_blocks, num_output_blocks, activations=F.relu):
    if isinstance(activations, list):
        return activations + [None] * (num_blocks - len(activations))

    return (
        [None] * num_input_blocks +
        [activations] * (num_blocks - num_input_blocks - num_output_blocks) +
        [None] * num_output_blocks
    )

