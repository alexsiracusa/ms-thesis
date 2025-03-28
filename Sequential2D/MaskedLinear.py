import torch
import torch.nn.functional as F

def _random_boolean_tensor(rows, cols, num_true):
    total_elements = rows * cols
    values = torch.tensor([True] * num_true + [False] * (total_elements - num_true))
    shuffled_values = values[torch.randperm(total_elements)]

    return shuffled_values.view(rows, cols)


class MaskedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, mask=None):
        super(MaskedLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.mask = mask if mask is not None else torch.ones((in_features, out_features), dtype=torch.bool)

    def forward(self, X):
        masked_weight = self.linear.weight * self.mask.T
        return F.linear(X, masked_weight, self.linear.bias)

    @staticmethod
    def sparse_random(in_features, out_features, bias=True, percent=0.5):
        total_elements = in_features * out_features
        mask = _random_boolean_tensor(in_features, out_features, int(percent * total_elements))
        return MaskedLinear(in_features, out_features, bias=bias, mask=mask)


