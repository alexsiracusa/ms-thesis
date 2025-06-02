from build_sequential2d import build_sequential2d
from sizes import input_sizes, hidden_sizes, output_sizes
from compute_loss import compute_loss

from lux.util import load_action_dataset, action_collate_fn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# TRAINING
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


dataset = load_action_dataset('../data')
dataloader = DataLoader(dataset, batch_size=1, collate_fn=action_collate_fn)
num_epochs = 25

for epoch in range(num_epochs):
    ce_losses = []
    mse_losses = []

    for batch_obs, batch_act, batch_sap, lengths in dataloader:
        start = 70
        max_seq_len = 1
        batch_size = batch_obs.shape[0]

        batch_obs = batch_obs.to(device)[:, start:start+max_seq_len]  # (batch_size, max_seq_len, input_dim)
        batch_act = batch_act.to(device)[:, start:start+max_seq_len]  # (batch_size, max_seq_len, target_dim=16)
        batch_sap = batch_sap.to(device)[:, start:start+max_seq_len]  # (batch_size, max_seq_len, target_dim=32)

        output = model(batch_obs, batch_first=True)  # (batch_size, max_seq_len, output_dim=48)
        ce_loss, mse_loss = compute_loss(output, batch_act, batch_sap)
        total_loss = ce_loss  # + mse_loss

        # Record loss values
        num_ce = batch_size * max_seq_len * 16
        num_mse = batch_size * max_seq_len * 32

        avg_ce_loss = ce_loss.item() / num_ce
        avg_mse_loss = mse_loss.item() / num_mse

        print(f"CE Loss: {avg_ce_loss:.4f} MSE Loss: {avg_mse_loss:.4f}")

        # Update weights
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    avg_ce_loss = sum(ce_losses) / len(ce_losses)
    avg_mse_loss = sum(ce_losses) / len(ce_losses)

    print(f"Epoch {epoch+1}/{num_epochs}, CE Loss: {avg_ce_loss:.4f} MSE Loss: {avg_mse_loss:.4f}")



