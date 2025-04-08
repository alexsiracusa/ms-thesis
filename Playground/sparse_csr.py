from util import random_boolean_tensor
from torch import nn
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

device = torch.device('cuda')
tensor_dim = 10000

sparsity_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
dense_times = []
sparse_times = []

dense_grad = []
sparse_grad = []

linear = nn.Linear(tensor_dim, tensor_dim, bias=False)
full_tensor = linear.weight.data.clone()

for sparsity in sparsity_values:
    mask = random_boolean_tensor(tensor_dim, tensor_dim, int(tensor_dim**2 * sparsity))
    tensor = full_tensor * mask
    tensor.requires_grad = True

    dense = tensor.to_dense().to(device)
    sparse = tensor.to_sparse_csr().to(device)
    bias = torch.normal(0, 1, size=(1, tensor_dim)).to(device)

    # DENSE
    X = torch.normal(0, 1, size=(100, tensor_dim), requires_grad=True).to(device)

    torch.cuda.synchronize()
    start = time.time()
    y = F.linear(X, dense, bias)
    torch.cuda.synchronize()
    dense_times.append(time.time() - start)
    print(y.sum().item()) # Prevent optimization

    total = torch.sum(y)
    start = time.time()
    # total.backward()
    dense_grad.append(time.time() - start)
    print(X.grad) # Prevent optimization


    # SPARSE
    X = torch.normal(0, 1, size=(100, tensor_dim), requires_grad=True).to(device)

    torch.cuda.synchronize()
    start = time.time()
    y = F.linear(X, dense, bias)
    torch.cuda.synchronize()
    sparse_times.append(time.time() - start)
    print(y.sum().item()) # Prevent optimization

    total = torch.sum(y)
    start = time.time()
    # total.backward()
    sparse_grad.append(time.time() - start)
    print(X.grad) # Prevent optimization

print("Sparse times:", sparse_times)
print("Dense times:", dense_times)
print("Sparse grad:", sparse_grad)
print("Dense grad:", dense_grad)

plt.figure(figsize=(10, 5))
plt.plot(sparsity_values, dense_times, label='Dense', color='red')
plt.plot(sparsity_values, sparse_times, label='Sparse', color='blue')
# plt.plot(sparsity_values, dense_grad, label='Dense', color='red')
# plt.plot(sparsity_values, sparse_grad, label='Sparse', color='blue')

plt.xlabel('Sparsity')
plt.ylabel('Time')
plt.title('Dot Product Time vs. Sparsity')
plt.legend()
plt.savefig('dot_times.png', dpi=300)
plt.show()
