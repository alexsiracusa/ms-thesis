import torch
import torch.nn.init as init
import math
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*Sparse CSR tensor support is in beta.*")

def _random_boolean_tensor(rows, cols, num_true):
    total_elements = rows * cols
    values = torch.tensor([True] * num_true + [False] * (total_elements - num_true))
    shuffled_values = values[torch.randperm(total_elements)]

    return shuffled_values.view(rows, cols)


class SparseLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, mask=None):
        super(SparseLinear, self).__init__()
        self.mask = mask
        self.in_features = in_features
        self.out_features = out_features

        weight = torch.empty((out_features, in_features)).to_sparse_csr()
        self.weight = torch.nn.Parameter(weight)

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weight = torch.empty((self.out_features, self.in_features))
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        weight = weight * self.mask
        self.weight = torch.nn.Parameter(weight.to_sparse_csr())

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, X):
        # self.weight.register_hook(lambda grad: (grad.to_dense() * self.mask))

        # print(self.weight.is_sparse_csr)
        # return X @ self.weight.T + self.bias
        return F.linear(X, self.weight, self.bias)

    @staticmethod
    def sparse_random(in_features, out_features, bias=True, percent=0.5):
        total_elements = in_features * out_features
        mask = _random_boolean_tensor(out_features, in_features, int(percent * total_elements))
        return SparseLinear(in_features, out_features, bias=bias, mask=mask)


