import torch
from torch import nn
from Sequential2D import SparseAdam
from training import train, load_mnist, masked_model, sparse_model, old_sparse_model
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from statistics import median


data_folder = "../data"
train_loader, test_loader = load_mnist(data_folder, flat=True)
device = torch.device('cuda')

size = 1600
subset = Subset(train_loader.dataset, range(size))
subset_loader = torch.utils.data.DataLoader(subset, batch_size=train_loader.batch_size, shuffle=False)



sizes = [2500, 500, 200, 100, 10]
# sparsity_list = [0, 0.25, 0.5, 0.75, 1.0]
sparsity_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

masked_forward_times = []
sparse_forward_times = []
old_sparse_forward_times = []

masked_backward_times = []
sparse_backward_times = []
old_sparse_backward_times = []


# masked_forward_times = [0.01273928165435791, 0.013218855857849121, 0.012310237884521484, 0.012340340614318848, 0.012556028366088868, 0.01250183343887329, 0.012940738201141357, 0.012788033485412598, 0.012941372394561768, 0.01184880495071411, 0.011874573230743408]
# sparse_forward_times = [0.0030842208862304687, 0.009200596809387207, 0.009857857227325439, 0.010998451709747314, 0.011836092472076416, 0.01310117483139038, 0.0143141508102417, 0.015034770965576172, 0.016474893093109132, 0.017678136825561522, 0.01833742141723633]
#
# masked_backward_times = [0.016972372531890868, 0.01753736734390259, 0.016421446800231932, 0.015951147079467775, 0.01681307554244995, 0.016379332542419432, 0.016335954666137697, 0.015836167335510253, 0.016060047149658203, 0.015207455158233643, 0.015526809692382813]
# sparse_backward_times = [0.007974984645843506, 0.08544607639312744, 0.09742725610733033, 0.11184751272201537, 0.1781431841850281, 0.20318341970443726, 0.19278416395187378, 0.18812758922576905, 0.19808510541915894, 0.19478490114212035, 0.21523140430450438]


for sparsity in sparsity_list:
    masked = masked_model(sizes, sparsity).to(device)
    sparse = sparse_model(sizes, sparsity).to(device)
    old_sparse = old_sparse_model(sizes, sparsity, device=device).to(device)

    criterion = nn.CrossEntropyLoss()
    masked_optim = SparseAdam(masked.parameters(), lr=0.0001)
    sparse_optim = SparseAdam(sparse.parameters(), lr=0.0001)
    old_sparse_optim = SparseAdam(old_sparse.parameters(), lr=0.0001)

    masked_losses, m_forward_times, m_backward_times = train(masked, subset_loader, criterion, masked_optim, device=device)
    sparse_losses, s_forward_times, s_backward_times = train(sparse, subset_loader, criterion, sparse_optim, device=device)
    # old_sparse_losses, os_forward_times, os_backward_times = train(old_sparse, subset_loader, criterion, old_sparse_optim, device=device)

    # masked_forward_times.append(sum(m_forward_times) / len(m_forward_times))
    masked_forward_times.append(median(m_forward_times))
    # sparse_forward_times.append(sum(s_forward_times) / len(s_forward_times))
    sparse_forward_times.append(median(s_forward_times))
    # old_sparse_forward_times.append(sum(os_forward_times) / len(os_forward_times))
    # old_sparse_forward_times.append(median(os_forward_times))

    # masked_backward_times.append(sum(m_backward_times) / len(m_backward_times))
    masked_backward_times.append(median(m_backward_times))
    # sparse_backward_times.append(sum(s_backward_times) / len(s_backward_times))
    sparse_backward_times.append(median(s_backward_times))
    # old_sparse_backward_times.append(sum(os_backward_times) / len(os_backward_times))
    # old_sparse_backward_times.append(median(os_backward_times))


print('Masked Forward:    ', masked_forward_times)
print('Sparse Forward:    ', sparse_forward_times)
print('OldSparse Forward: ', old_sparse_forward_times)

print('Masked Backward:   ', masked_backward_times)
print('Sparse Backward:   ', sparse_backward_times)
print('OldSparse Backward:', old_sparse_backward_times)


plt.figure(figsize=(10, 5))
plt.plot(sparsity_list, masked_forward_times, label='Masked Forward', color='red')
plt.plot(sparsity_list, sparse_forward_times, label='Sparse Forward', color='green')
# plt.plot(sparsity_list, old_sparse_forward_times, label='Old Sparse Forward', color='blue')

plt.plot(sparsity_list, masked_backward_times, label='Masked Backward', color='purple')
plt.plot(sparsity_list, sparse_backward_times, label='Sparse Backward', color='gold')
# plt.plot(sparsity_list, old_sparse_backward_times, label='Old Sparse Backward', color='gray')

# Labels and title
plt.xlabel('Sparsity')
plt.ylabel('Time')
plt.title('Time vs. Sparsity')
plt.legend()

# Save as PNG
plt.savefig('training_times.png', dpi=300)
plt.show()

