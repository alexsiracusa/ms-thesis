import torch
import math


# TODO: possibly change this to work with static non-zero masked weights at initialization
class MaskedLinear(torch.nn.Module):
    """
    Args:
        in_features (int): The number of input features
        out_features(int): The number of output features
        bias: Whether to include a bias term
        mask (out_features, in_features):
            A tensor of 0's or 1's to effectively ignore certain weights. In both the forward and backward pass
            the weights/gradient is multiplied by this mask element-wise .
        weighted_init (bool):
            Whether to use the weighted initialization described below, or the default torch.nn.Linear initialization.

            Initializes each row of the weight matrix uniformly from (-bound, bound) where bound is equal to
            1 / sqrt(sum(mask[row] == 1)). This is a similar concept to the kaiming uniform initialization
            (the default for nn.Linear), however since each row can be sparse the effective amount of in_features
            is actually smaller as (1 - sparsity)% get ignored/are multiplied by 0 on average. This initialization
            accounts for that by calculating the "real" amount of in_features per row and uses that instead.
    """
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            mask=None,
            weighted_init=False,
    ):
        super(MaskedLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.register_buffer('mask', mask)

        if weighted_init and mask is not None:
            weight = torch.empty_like(mask)
            one_counts = (self.mask != 0).sum(dim=1)

            for i in range(weight.shape[0]):
                bound = 1 / max(math.sqrt(one_counts[i]), 1)
                weight[i].uniform_(-bound, bound)

            with torch.no_grad():
                self.linear.weight.copy_(weight)

        # Register the gradient hook
        self.linear.weight.register_hook(self._mask_grad)

    def forward(self, X):
        if self.mask is not None:
            self.linear.weight.data *= self.mask  # Enforce mask in-place
        return self.linear(X)

    def _mask_grad(self, grad):
        if self.mask is not None:
            return grad * self.mask  # Apply mask during backpropagation
        return grad

    """
    See `random_mask` in `mask.py` for parameter details
    """
    @staticmethod
    def sparse_random(in_features, out_features, bias=True, percent=0.5, weighted_init=False):
        from .util.mask import random_mask

        mask = random_mask(out_features, in_features, percent)
        return MaskedLinear(in_features, out_features, bias=bias, mask=mask, weighted_init=weighted_init)

    """
    See `variable_mask` in `mask.py` for parameter details
    """
    @staticmethod
    def variable_random(in_features, out_features, bias=True, densities=1, weighted_init=False):
        from .util.mask import variable_mask

        mask = variable_mask(out_features, in_features, densities)
        return MaskedLinear(sum(in_features), sum(out_features), bias=bias, mask=mask, weighted_init=weighted_init)


