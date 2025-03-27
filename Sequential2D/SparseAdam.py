import torch
from torch.optim.optimizer import Optimizer

print(*torch.__config__.show().split("\n"), sep="\n")


# def _mask_with_csr(dense, sparse_csr):
#     row_indices = torch.arange(sparse_csr.shape[0]).repeat_interleave(
#         sparse_csr.crow_indices().diff()
#     )
#     col_indices = sparse_csr.col_indices()
#
#     # Gather values from dense tensor
#     sparse_values = dense[row_indices, col_indices]
#
#     # Create new sparse CSR tensor
#     return torch.sparse_csr_tensor(
#         sparse_csr.crow_indices(), col_indices, sparse_values, sparse_csr.shape
#     )


def _mask_with_csr(tensor, sparse_csr):
    """
    Masks the input tensor so that only the indices defined in sparse_csr remain.

    Parameters:
    tensor (torch.Tensor): The input tensor (dense or sparse).
    sparse_csr (torch.Tensor): The sparse CSR tensor defining the mask (values > 0 are treated as 1).

    Returns:
    torch.Tensor: The masked tensor with undefined values set to zero.
    """
    mask_values = torch.where(sparse_csr.values() > 0, torch.tensor(1, dtype=sparse_csr.dtype),
                              torch.tensor(0, dtype=sparse_csr.dtype))
    mask = torch.sparse_csr_tensor(sparse_csr.crow_indices(), sparse_csr.col_indices(), mask_values,
                                   size=sparse_csr.shape)

    if tensor.is_sparse_csr:
        masked = tensor * mask
    else:
        masked = tensor * mask.to_dense()

    return masked.to_sparse_csr()

def _empty_csr(size, device):
    return torch.sparse_csr_tensor(torch.tensor([0]), torch.tensor([]), torch.tensor([]), size=size, dtype=torch.float)


class SparseAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(SparseAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in reversed(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad

                # m_t = B_1 * m_t-1 + (1 - B_1) * g_t
                # v_t = B_2 * v_t-1 + (1 - B_2) * g_t^2
                # ^m_t = m_t / (1 - B_1)
                # ^v_t = v_t / (1 - B_2_
                # p = p - (lr * m_t / sqrt(^v_t + e))

                m_is_csr = len(self.state) != 0 and torch.is_tensor(self.state['m']) and self.state[
                    'm'].layout == torch.sparse_csr
                v_is_csr = len(self.state) != 0 and torch.is_tensor(self.state['v']) and self.state[
                    'v'].layout == torch.sparse_csr

                print((
                    f'param: {p.is_sparse_csr}\n'
                    f'grad:  {grad.is_sparse_csr}\n'
                    f'm:     {m_is_csr}\n'
                    f'v:     {v_is_csr}\n'
                ))

                # Handle sparse tensors
                if p.is_sparse_csr:
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['m'] = _empty_csr(p.data.shape, device=p.device)
                        state['v'] = _empty_csr(p.data.shape, device=p.device)

                    m, v = state['m'], state['v']
                    beta1, beta2 = group['betas']
                    state['step'] += 1
                    step = state['step']

                    m = _mask_with_csr(m, p)
                    v = _mask_with_csr(v, p)
                    grad = _mask_with_csr(grad, p)

                    print(grad.is_sparse_csr)
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * grad**2

                    state['m'] = m
                    state['v'] = v

                    print(m.is_sparse_csr, v.is_sparse_csr)

                    raise Exception("not implemented yet")

                elif not p.is_sparse_csr and not grad.is_sparse_csr:
                    # Dense tensor operations (standard Adam)
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['m'] = torch.zeros_like(p)
                        state['v'] = torch.zeros_like(p)

                    m, v = state['m'], state['v']
                    beta1, beta2 = group['betas']
                    state['step'] += 1

                    # Update m and v for dense gradients
                    m = m.to_dense()
                    grad = grad.to_dense()
                    v = v.to_dense()
                    data = p.data.to_dense()

                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Compute bias correction
                    m_hat = m / (1 - beta1 ** state['step'])
                    v_hat = v / (1 - beta2 ** state['step'])

                    # Update dense gradient
                    denom = v_hat.sqrt().add_(group['eps'])
                    data.addcdiv_(m_hat, denom, value=-group['lr'])
                else:
                    raise Exception("not implemented yet")

        return loss
