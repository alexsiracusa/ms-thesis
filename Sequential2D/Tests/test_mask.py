from Sequential2D.util.mask import random_mask, variable_mask

row_blocks = [1, 2]
col_blocks = [1, 2, 3]
densities = [
    [0, 1, 0],
    [1, 0, 0.5]
]

mask = variable_mask(row_blocks, col_blocks, densities)

print(mask)