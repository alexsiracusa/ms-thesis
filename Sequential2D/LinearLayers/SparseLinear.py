import torch
from torch import nn
import torch.nn.init as init
import math
import torch.nn.functional as F
import warnings
from Sequential2D.util.random_mask import random_boolean_tensor

warnings.filterwarnings("ignore", category=UserWarning, message=".*Sparse CSR tensor support is in beta.*")


class SparseLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, mask=None, device=None):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        self.register_buffer("mask", mask.to(device))

        # components of weights
        self.values = nn.Parameter(torch.tensor([0], dtype=torch.float), requires_grad=True)
        self.crow_indices = nn.Parameter(torch.tensor([]), requires_grad=False)
        self.col_indices = nn.Parameter(torch.tensor([]), requires_grad=False)
        self.shape = (out_features, in_features)

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, device=self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    # this is meant to be the same as the default nn.Linear initialization
    def reset_parameters(self):
        self._initialize_weights()
        self._initialize_bias()

    def _initialize_weights(self):
        w = self.mask.to_sparse_csr()

        samples = torch.empty(self.shape)
        init.kaiming_uniform_(samples, a=math.sqrt(5))

        self.values = nn.Parameter(samples.flatten()[:len(w.values())], requires_grad=True)
        self.crow_indices = nn.Parameter(w.crow_indices(), requires_grad=False)
        self.col_indices = nn.Parameter(w.col_indices(), requires_grad=False)

    def _initialize_bias(self):
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, X):
        weight = torch.sparse_csr_tensor(
            self.crow_indices, self.col_indices, self.values, size=self.shape, device=self.values.device
        )

        return F.linear(X, weight, self.bias)

    @staticmethod
    def sparse_random(in_features, out_features, bias=True, percent=0.5, device=None):
        total_elements = in_features * out_features
        mask = random_boolean_tensor(out_features, in_features, int(percent * total_elements))
        return SparseLinear(in_features, out_features, bias=bias, mask=mask, device=device)


