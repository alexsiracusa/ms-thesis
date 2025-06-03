import torch
import torch.optim as optim

import random

from lux.models.sizes import input_sizes, hidden_sizes, output_sizes
from lux.models.build_sequential2d import build_sequential2d
from lux.models.compute_loss import compute_loss
from lux.util.load_action_dataset import load_action_dataset


device = torch.device('cpu')
print(f'Device: {device}')

model_sizes = input_sizes + hidden_sizes + output_sizes
model = build_sequential2d(
    model_sizes,
    num_input_blocks=len(input_sizes),
    num_output_blocks=2,
    num_iterations=2
)
model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

dataset = load_action_dataset('../../data/train')
random_index = int(random.random() * len(dataset))
obs, act, sap = dataset[random_index]

start = 80
max_seq_len = 1
obs = obs[start:start+max_seq_len].unsqueeze(0)  # (1, max_seq_len, input_dim)
act = act[start:start+max_seq_len].unsqueeze(0)  # (1, max_seq_len, target_dim=16)
sap = sap[start:start+max_seq_len].unsqueeze(0)  # (1, max_seq_len, target_dim=32)

num_epochs = 5

for epoch in range(num_epochs):

    output = model.forward(obs)
    ce_loss, mse_loss = compute_loss(output, act, sap)
    total_loss = ce_loss + mse_loss

    print(f"CE Loss: {ce_loss.item():.4f} MSE Loss: {mse_loss.item():.4f}")

    # print actions for random agent
    # agent_num = int(random.random() * 16)
    # output = output[:, :, -128:]
    # print(act[0, 0, agent_num].tolist(), output[0, 0, agent_num*6:agent_num*6+6].tolist())
    # print(sap[0, 0, agent_num*2:agent_num*2+2].tolist(), output[0, 0, 96+agent_num*2:96+agent_num*2+2].tolist())

    # Update weights
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

print('done')
