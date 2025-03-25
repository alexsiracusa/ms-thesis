import torch
import torch.nn.functional as F


class MaskedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, mask=None):
        super(MaskedLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.mask = mask if mask is not None else torch.ones((in_features, out_features), dtype=torch.bool)

    def forward(self, X):
        masked_weight = self.linear.weight * self.mask.T
        return F.linear(X, masked_weight, self.linear.bias)
