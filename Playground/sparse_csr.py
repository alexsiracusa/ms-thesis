from Playground.util import random_boolean_tensor
from torch import nn
import time
import matplotlib.pyplot as plt
import torch

device = torch.device('cpu')
tensor_dim = 10000

sparsity_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
dense_times = []
sparse_times = []

linear = nn.Linear(tensor_dim, tensor_dim, bias=False)
full_tensor = linear.weight.data.clone()

for sparsity in sparsity_values:
    mask = random_boolean_tensor(tensor_dim, tensor_dim, int(tensor_dim**2 * sparsity))
    tensor = full_tensor * mask
    tensor = tensor.to(device)

    dense = tensor.to_dense().to(device)
    sparse = tensor.to_sparse_csr().to(device)

    for i in range(25):
        x = torch.randn(size=(tensor_dim,), device=device)
        y = dense @ x
        print(y.sum().item())

    # DENSE
    X = torch.normal(0, 1, size=(1, tensor_dim), device=device)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        # y = F.linear(dense, X, None)
        y = dense @ X.T
    torch.cuda.synchronize()
    dense_times.append(time.time() - start)
    print(y.sum().item()) # Prevent optimization


    for i in range(25):
        x = torch.randn(size=(tensor_dim,), device=device)
        y = sparse @ x
        print(y.sum().item())

    # SPARSE
    X = torch.normal(0, 1, size=(1, tensor_dim), device=device)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        # y = F.linear(sparse, X, None)
        y = sparse @ X.T
    torch.cuda.synchronize()
    sparse_times.append(time.time() - start)
    print(y.sum().item()) # Prevent optimization


print("Sparse times:", sparse_times)
print("Dense times:", dense_times)

plt.figure(figsize=(10, 5))
plt.plot(sparsity_values, dense_times, label='Dense', color='red')
plt.plot(sparsity_values, sparse_times, label='Sparse', color='blue')

plt.xlabel('Sparsity')
plt.ylabel('Time')
plt.title('Dot Product Time vs. Sparsity')
plt.legend()
plt.savefig('dot_times.png', dpi=300)
plt.show()
