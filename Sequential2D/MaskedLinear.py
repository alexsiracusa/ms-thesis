import torch


# TODO: possibly change this to work with static non-zero masked weights at initialization
class MaskedLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            weight=None,
            mask=None,
    ):
        super(MaskedLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.register_buffer('mask', mask)

        if weight is not None:
            assert weight.shape == (out_features, in_features), 'weight dimensions must match'
            with torch.no_grad():
                self.linear.weight.copy_(weight)

        # Register the gradient hook
        self.linear.weight.register_hook(self._mask_grad)

    def forward(self, X):
        self.linear.weight.data *= self.mask  # Enforce mask in-place
        return self.linear(X)

    def _mask_grad(self, grad):
        return grad * self.mask  # Apply mask during backpropagation

    @staticmethod
    def sparse_random(in_features, out_features, bias=True, percent=0.5):
        from .util.mask import random_mask, variable_mask

        mask = random_mask(out_features, in_features, percent)
        return MaskedLinear(in_features, out_features, bias=bias, mask=mask)


