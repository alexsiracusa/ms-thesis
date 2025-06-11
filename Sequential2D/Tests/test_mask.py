from Sequential2D.util.mask import random_mask, variable_mask

row_blocks = [1, 2]
col_blocks = [4, 8]
densities = [
    [0.25, 0.50],
    [0.75, 1.00]
]

mask = variable_mask(row_blocks, col_blocks, densities)

print(mask)