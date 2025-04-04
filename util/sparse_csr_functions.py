import torch
from typing import Callable

def mask_with_csr(tensor, sparse_csr):
    """
    Masks the input tensor so that only the indices defined in sparse_csr remain.

    Parameters:
    tensor (torch.Tensor): The input tensor (dense or sparse).
    sparse_csr (torch.Tensor): The sparse CSR tensor defining the mask (values > 0 are treated as 1).

    Returns:
    torch.Tensor: The masked tensor with undefined values set to zero.
    """
    mask_values = torch.where(sparse_csr.values() > 0, torch.tensor(1, dtype=sparse_csr.dtype),
                              torch.tensor(0, dtype=sparse_csr.dtype, device=tensor.device))
    mask = torch.sparse_csr_tensor(sparse_csr.crow_indices(), sparse_csr.col_indices(), mask_values,
                                   size=sparse_csr.shape, device=tensor.device)

    if tensor.is_sparse_csr:
        masked = tensor * mask
    else:
        masked = tensor * mask.to_dense()

    return masked.to_sparse_csr()


def csr_transform(sparse_csr: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    # in place version:
    # grad.values().copy_(grad.values() ** 2)
    return torch.sparse_csr_tensor(
        sparse_csr.crow_indices(),
        sparse_csr.col_indices(),
        f(sparse_csr.values()),  # Apply function to values
        size=sparse_csr.shape
    )

def csr_power(sparse_csr, val):
    return csr_transform(sparse_csr, lambda v: v ** val)

def csr_add(sparse_csr, val):
    return csr_transform(sparse_csr, lambda v: v + val)

def csr_multiply(sparse_csr, val):
     return csr_transform(sparse_csr, lambda v: v * val)

def csr_divide(sparse_csr, val):
    return csr_transform(sparse_csr, lambda v: v / val)

def empty_csr(size, device):
    return torch.sparse_csr_tensor(torch.tensor([0]), torch.tensor([]), torch.tensor([]), size=size, dtype=torch.float, device=device)


def sparse_csr_divide(numerator: torch.Tensor, denominator: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Element-wise division of two sparse CSR tensors, replacing zeros in the denominator with epsilon.

    Args:
        numerator (torch.Tensor): A sparse CSR tensor (PyTorch 2.0+ format).
        denominator (torch.Tensor): A sparse CSR tensor with the same shape as numerator.
        epsilon (float): Minimum value to use instead of zero in the denominator.

    Returns:
        torch.Tensor: A sparse CSR tensor containing the division results.
    """
    assert numerator.layout == torch.sparse_csr and denominator.layout == torch.sparse_csr, "Both tensors must be sparse CSR."
    assert numerator.shape == denominator.shape, "Shapes of numerator and denominator must match."

    # Extract CSR components
    crow_indices = numerator.crow_indices()
    col_indices = numerator.col_indices()
    num_values = numerator.values()
    denom_values = denominator.values()

    # Avoid division by zero by replacing zeros with epsilon
    safe_denom_values = torch.where(
        denom_values == 0,
        torch.tensor(epsilon, dtype=denom_values.dtype, device=denom_values.device),
        denom_values
    )

    # Perform element-wise division
    result_values = num_values / safe_denom_values

    # Construct the result sparse CSR tensor
    result = torch.sparse_csr_tensor(crow_indices, col_indices, result_values, size=numerator.shape,
                                     dtype=numerator.dtype, device=numerator.device)

    return result