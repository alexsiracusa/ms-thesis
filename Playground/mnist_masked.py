import torch
from torch import nn, optim
from training import train, load_mnist
from util import num_trainable_parameters
import numpy as np
import matplotlib.pyplot as plt
from training.create_models import masked_model



data_folder = "../data"
train_loader, test_loader = load_mnist(data_folder, flat=True)
device = torch.device('cuda')


sizes = [2500, 500, 200, 100, 10]

#            2500  500   200   100   10
# blocks = [[I,    None, None, None, None],
#           [f10,  f11,  f12,  f13,  f14 ],
#           [f20,  f21,  f22,  f23,  f24 ],
#           [f30,  f31,  f32,  f33,  f34 ],
#           [f40,  f41,  f42,  f43,  f44 ]]

model1 = masked_model(sizes=sizes, sparsity=0.5)
print(f'Trainable: {num_trainable_parameters(model1)}')


# train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=0.0001)

losses, forward_times, backward_times = train(model1, train_loader, criterion, optimizer, nth_batch=1, device=device)
iterations = np.arange(len(losses))

print(f'Forward:  {sum(forward_times) / len(forward_times)}')
print(f'Backward: {sum(backward_times) / len(backward_times)}')


plt.figure(figsize=(10, 5))
plt.plot(iterations, losses, label='Loss', color='blue')

# Labels and title
plt.xlabel('Batch Num.')
plt.ylabel('Loss')
plt.title('Loss over Batches')
plt.legend()

# Save as PNG
plt.savefig('training_metrics.png', dpi=300)
plt.show()

