import torch
from torch.optim.optimizer import Optimizer
from typing import Tuple
from util import sparse_csr_divide, csr_divide, csr_power, csr_multiply, csr_transform, empty_csr, mask_with_csr

class SparseAdam(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(SparseAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # m_t = B_1 * m_t-1 + (1 - B_1) * g_t
                # v_t = B_2 * v_t-1 + (1 - B_2) * g_t^2
                # ^m_t = m_t / (1 - B_1)
                # ^v_t = v_t / (1 - B_2)
                # p = p - (lr * m_t / sqrt(^v_t + e))

                # Handle sparse tensors
                if p.is_sparse_csr:
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['m'] = empty_csr(p.data.shape, device=p.device)
                        state['v'] = empty_csr(p.data.shape, device=p.device)

                    m, v = state['m'], state['v']
                    beta1, beta2 = group['betas']
                    state['step'] += 1

                    m = mask_with_csr(m, p)
                    v = mask_with_csr(v, p)
                    grad = mask_with_csr(grad, p)

                    # Update m and v
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * csr_power(grad, 2)

                    # Store updated values back
                    state['m'], state['v'] = m, v

                    m_hat = csr_divide(m, 1 - beta1 ** state['step'])
                    v_hat = csr_divide(v, 1 - beta2 ** state['step'])

                    denom = csr_transform(v_hat, lambda v: v ** 0.5 + group['eps'])

                    data = p.data + sparse_csr_divide(csr_multiply(m_hat, -group['lr']), denom, group['eps'])
                    p.data.copy_(data)

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

                    # Update m and v
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Store updated values back
                    state['m'], state['v'] = m, v

                    # Compute bias correction
                    m_hat = m / (1 - beta1 ** state['step'])
                    v_hat = v / (1 - beta2 ** state['step'])

                    # Update dense gradient
                    denom = v_hat.sqrt().add_(group['eps'])
                    p.data.addcdiv_(m_hat, denom, value=-group['lr'])
                else:
                    raise Exception("not implemented yet")

        return loss
