import torch


def random_mask(rows, cols, num_true):
    total_elements = rows * cols
    values = torch.tensor([True] * num_true + [False] * (total_elements - num_true))
    shuffled_values = values[torch.randperm(total_elements)]

    return shuffled_values.view(rows, cols)


class MaskedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, mask=None):
        super(MaskedLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.register_buffer('mask', mask)

    def forward(self, X):
        with torch.no_grad():
            self.linear.weight.data *= self.mask.T

        return self.linear.forward(X)

    @staticmethod
    def sparse_random(in_features, out_features, bias=True, percent=0.5):
        total_elements = in_features * out_features
        mask = random_mask(in_features, out_features, int(percent * total_elements))
        return MaskedLinear(in_features, out_features, bias=bias, mask=mask)


