import numpy as np

from .sizes import sizes, num_input


def get_num_trainable(densities, sizes=sizes, num_input=num_input):

    num_parameters = np.array([
        [row_size * col_size for col_size in sizes]
        for row_size in sizes
    ])

    return (np.array(densities) * num_parameters[num_input:]).sum()