from util import random_boolean_tensor
from torch import nn
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

device = torch.device('cpu')
tensor_dim = 2000

sparsity_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
dense_times = []
sparse_times = []

dense_grad = []
sparse_grad = []

linear = nn.Linear(tensor_dim, tensor_dim, bias=False)
full_tensor = linear.weight.data.clone()

for sparsity in sparsity_values:
    mask = random_boolean_tensor(tensor_dim, tensor_dim, int(tensor_dim**2 * sparsity))
    mask.requires_grad = True
    tensor = full_tensor * mask

    dense = tensor.to_dense().to(device)
    sparse = tensor.to_sparse_csr().to(device)
    bias = torch.normal(0, 1, size=(1, tensor_dim)).to(device)

    X_input = torch.normal(0, 1, size=(100, tensor_dim), requires_grad=True).to(device)

    # DENSE
    torch.cuda.synchronize()
    start = time.time()
    output = X_input
    for _ in range(1000):
        output = F.linear(output, dense, bias)
    torch.cuda.synchronize()
    # Prevent optimization: force some computation using the output
    dense_times.append(time.time() - start)
    print(output.sum().item())

    start = time.time()
    torch.sum(output).backward()
    dense_grad.append(time.time() - start)
    print(output.grad[0])


    # SPARSE
    X_input = torch.normal(0, 1, size=(100, tensor_dim), requires_grad=True).to(device)

    torch.cuda.synchronize()
    start = time.time()
    output = X_input
    for _ in range(1000):
        output = F.linear(output, sparse, bias)
    torch.cuda.synchronize()
    sparse_times.append(time.time() - start)
    print(output.sum().item())

    start = time.time()
    torch.sum(output).backward()
    sparse_grad.append(time.time() - start)
    print(output.grad[0])

print("Sparse times:", sparse_times)
print("Dense times:", dense_times)

plt.figure(figsize=(10, 5))
# plt.plot(sparsity_values, dense_times, label='Dense', color='red')
# plt.plot(sparsity_values, sparse_times, label='Sparse', color='blue')
plt.plot(sparsity_values, dense_grad, label='Dense', color='red')
plt.plot(sparsity_values, sparse_grad, label='Sparse', color='blue')

plt.xlabel('Sparsity')
plt.ylabel('Time')
plt.title('Dot Product Time vs. Sparsity')
plt.legend()
plt.savefig('dot_times.png', dpi=300)
plt.show()
