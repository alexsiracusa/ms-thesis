import numpy as np

from .sizes import input_sizes, sizes


def get_num_trainable(densities):

    num_parameters = np.array([
        [row_size * col_size for col_size in sizes]
        for row_size in sizes
    ])

    return (np.array(densities) * num_parameters[len(input_sizes):]).sum()