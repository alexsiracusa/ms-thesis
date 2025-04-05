import torch
import torch.nn.init as init
import math
import torch.nn.functional as F
import warnings
from util.random_mask import random_boolean_tensor

warnings.filterwarnings("ignore", category=UserWarning, message=".*Sparse CSR tensor support is in beta.*")


class SparseLinearOld(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, mask=None, device=None):
        super(SparseLinearOld, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        self.register_buffer("mask", mask.to(device))
        self.weight = None # Will be set in `reset_parameters()`

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, device=self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weight = torch.empty((self.out_features, self.in_features), device=self.device)
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        weight = weight * self.mask
        self.weight = torch.nn.Parameter(weight.to_sparse_csr())

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, X):
        return F.linear(X, self.weight, self.bias)

    @staticmethod
    def sparse_random(in_features, out_features, bias=True, percent=0.5, device=None):
        total_elements = in_features * out_features
        mask = random_boolean_tensor(out_features, in_features, int(percent * total_elements))
        return SparseLinearOld(in_features, out_features, bias=bias, mask=mask, device=device)

