from torch.utils.data import DataLoader
from build_sequential2d import build_sequential2d
from lux.util import load_action_dataset, action_collate_fn
from lux.models.sizes import input_sizes, hidden_sizes, output_sizes

import torch
import torch.nn as nn
import torch.optim as optim


# TRAINING
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print(f'Device: {device}')

model_sizes = input_sizes + hidden_sizes + output_sizes
model = build_sequential2d(model_sizes, num_input_blocks=len(input_sizes), num_iterations=2)
model.to(device)
model.train()

ce_loss_fn = nn.CrossEntropyLoss()
mse_loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


dataset = load_action_dataset('../data')
dataloader = DataLoader(dataset, batch_size=1, collate_fn=action_collate_fn)
num_epochs = 25


def compute_loss(
        output,
        batch_act,  # (batch_size, max_seq_len, 16)
        batch_sap   # (batch_size, max_seq_len, 32)
):
    output = output[:, :, -128:]  # (batch_size, max_seq_len, output_dim=128)

    # Reshape targets
    flat_actions = batch_act.reshape(-1)         # (batch_size * max_seq_len * 16)
    flat_sap_deltas = batch_sap.reshape(-1, 32)  # (batch_size * max_seq_len, 32)

    # Reshape predictions
    actions_hat = output[:, :, :96]     # (batch_size, max_seq_len, 96)
    sap_deltas_hat = output[:, :, 96:]  # (batch_size, max_seq_len, 32)

    flat_actions_hat = torch.reshape(actions_hat, (-1, 6))         # (batch_size * max_seq_len * 16, 6)
    flat_sap_deltas_hat = torch.reshape(sap_deltas_hat, (-1, 32))  # (batch_size * max_seq_len, 32)

    ce_loss = ce_loss_fn(flat_actions_hat, flat_actions)
    mse_loss = mse_loss_fn(flat_sap_deltas_hat, flat_sap_deltas)

    return ce_loss, mse_loss


for epoch in range(num_epochs):
    total_ce_loss = 0.0
    total_mse_loss = 0.0

    for batch_obs, batch_act, batch_sap, lengths in dataloader:
        start = 100
        max_seq_len = 1
        batch_size = batch_obs.shape[0]

        batch_obs = batch_obs.to(device)[:, start:start+max_seq_len]  # (batch_size, max_seq_len, input_dim)
        batch_act = batch_act.to(device)[:, start:start+max_seq_len]  # (batch_size, max_seq_len, target_dim=16)
        batch_sap = batch_sap.to(device)[:, start:start+max_seq_len]  # (batch_size, max_seq_len, target_dim=32)

        output = model(batch_obs, batch_first=True)  # (batch_size, max_seq_len, output_dim=48)
        ce_loss, mse_loss = compute_loss(output, batch_act, batch_sap)
        total_loss = ce_loss # + mse_loss

        num_ce = batch_size * max_seq_len * 16
        num_mse = batch_size * max_seq_len * 32
        print(f"CE Loss: {ce_loss.item() / num_ce:.4f} MSE Loss: {mse_loss.item() / num_mse:.4f}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    avg_ce_loss = total_ce_loss / len(dataloader) / 16
    avg_mse_loss = total_mse_loss / len(dataloader) / 16

    print(f"Epoch {epoch+1}/{num_epochs}, CE Loss: {avg_ce_loss:.4f} MSE Loss: {avg_mse_loss:.4f}")



