import torch


def random_mask(rows, cols, percent):
    total_elements = rows * cols
    num_true = int(total_elements * percent)

    values = torch.tensor([True] * num_true + [False] * (total_elements - num_true))
    shuffled_values = values[torch.randperm(total_elements)]

    return shuffled_values.view(rows, cols)


def variable_mask(row_blocks, col_blocks, densities):
    mask = torch.zeros((sum(row_blocks), sum(col_blocks)))

    for i in range(len(row_blocks)):
        for j in range(len(col_blocks)):
            r_start = sum(row_blocks[:i])
            r_end = sum(row_blocks[:i + 1])
            c_start = sum(col_blocks[:j])
            c_end = sum(col_blocks[:j + 1])

            mask_part = random_mask(row_blocks[i], col_blocks[j], densities[i][j])
            mask[r_start:r_end, c_start:c_end].copy_(mask_part)

    return mask
