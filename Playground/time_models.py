import torch
from torch import nn
from Sequential2D import SparseAdam
from training import train, load_mnist, masked_model, sparse_model, old_sparse_model
import matplotlib.pyplot as plt


data_folder = "../data"
train_loader, test_loader = load_mnist(data_folder, flat=True)
device = torch.device('cpu')


sizes = [2500, 500, 200, 100, 10]
sparsity_list = [0, 0.25, 0.5, 0.75, 1.0]

masked_forward_times = []
sparse_forward_times = []
old_sparse_forward_times = []

masked_backward_times = []
sparse_backward_times = []
old_sparse_backward_times = []

for sparsity in sparsity_list:
    masked = masked_model(sizes, sparsity).to(device)
    sparse = sparse_model(sizes, sparsity).to(device)
    old_sparse = old_sparse_model(sizes, sparsity, device=device).to(device)

    criterion = nn.CrossEntropyLoss()
    masked_optim = SparseAdam(masked.parameters(), lr=0.0001)
    sparse_optim = SparseAdam(sparse.parameters(), lr=0.0001)
    old_sparse_optim = SparseAdam(old_sparse.parameters(), lr=0.0001)

    masked_losses, m_forward_times, m_backward_times = train(masked, train_loader, criterion, masked_optim, device=device)
    sparse_losses, s_forward_times, s_backward_times = train(sparse, train_loader, criterion, sparse_optim, device=device)
    old_sparse_losses, os_forward_times, os_backward_times = train(old_sparse, train_loader, criterion, old_sparse_optim, device=device)

    masked_forward_times.append(sum(m_forward_times) / len(m_forward_times))
    sparse_forward_times.append(sum(s_forward_times) / len(s_forward_times))
    old_sparse_forward_times.append(sum(os_forward_times) / len(os_forward_times))

    masked_backward_times.append(sum(m_backward_times) / len(m_backward_times))
    sparse_backward_times.append(sum(s_backward_times) / len(s_backward_times))
    old_sparse_backward_times.append(sum(os_backward_times) / len(os_backward_times))


plt.figure(figsize=(10, 5))
plt.plot(sparsity_list, masked_forward_times, label='Masked Forward', color='red')
plt.plot(sparsity_list, sparse_forward_times, label='Sparse Forward', color='green')
plt.plot(sparsity_list, old_sparse_forward_times, label='Old Sparse Forward', color='blue')

plt.plot(sparsity_list, masked_backward_times, label='Masked Backward', color='purple')
plt.plot(sparsity_list, sparse_backward_times, label='Sparse Backward', color='gold')
plt.plot(sparsity_list, old_sparse_backward_times, label='Old Sparse Backward', color='gray')

# Labels and title
plt.xlabel('Sparsity')
plt.ylabel('Time')
plt.title('Time vs. Sparsity')
plt.legend()

# Save as PNG
plt.savefig('training_times.png', dpi=300)
plt.show()

