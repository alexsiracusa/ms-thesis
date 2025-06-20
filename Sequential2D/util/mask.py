import torch


"""
Args:
    rows (int): The number of rows in the resulting tensor
    cols (int): The number of columns in the resulting tensor
    percent (float): The percent of 1's in the resulting tensor mask, the rest being 0's
    
Return:
    tensor (rows, cols): A tensor with the specified size and sparsity
"""
def random_mask(rows, cols, percent):
    total_elements = rows * cols
    num_true = int(total_elements * percent)

    values = torch.tensor([True] * num_true + [False] * (total_elements - num_true))
    shuffled_values = values[torch.randperm(total_elements)]

    return shuffled_values.view(rows, cols).to(torch.float32)


"""
Args:
    row_blocks (N): The sizes for each row of blocks
    col_blocks (M): The sizes for each column of blocks
    
    densities (Union[List[List[float]], float]): 
        Defines the densities for each block[i, j] in blocks.
        
        - If a list:
            A 2D list of shape (N, M) specifying the density (from 0.0 to 1.0) for each linear layer 
            in the resulting `Sequential2D` block matrix. The first `num_input_blocks` rows are ignored 
            as they are all set to torch.Identity or None.
            
        - If a float:
            Sets all linear blocks to the specified density.
    
Returns:
    tensor (n_rows, n_cols): A tensor with varying densities per block where:
        - n_rows = sum(row_blocks) 
        - n_cols = sum(col_blocks)
        
Example:
    row_blocks = [1, 2]
    col_blocks = [1, 2, 3]
    densities = [[A, B, C]
                 [D, E, F]]

    tensor = [[a, b, b, c, c, c]
              [d, e, e, f, f, f]
              [d, e, e, f, f, f]]

    such that A% of the entries labeled 'a' in `tensor` are 1's, and the rest are 0's, etc.
"""
def variable_mask(row_blocks, col_blocks, densities):
    mask = torch.zeros((sum(row_blocks), sum(col_blocks)))

    for i in range(len(row_blocks)):
        for j in range(len(col_blocks)):
            r_start = sum(row_blocks[:i])
            r_end = sum(row_blocks[:i + 1])
            c_start = sum(col_blocks[:j])
            c_end = sum(col_blocks[:j + 1])

            density = densities[i][j] if isinstance(densities, list) else densities
            mask_part = random_mask(row_blocks[i], col_blocks[j], density)
            mask[r_start:r_end, c_start:c_end].copy_(mask_part)

    return mask
