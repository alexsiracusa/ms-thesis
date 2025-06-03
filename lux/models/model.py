from build_sequential2d import build_sequential2d
from sizes import input_sizes, hidden_sizes, output_sizes
from compute_loss import compute_loss

from lux.util import load_action_dataset, action_collate_fn

import torch
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
model = torch.load('model_0_505-3.pth', weights_only=False)
model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=1e-4)


dataset = load_action_dataset('../data/train')
dataloader = DataLoader(dataset, batch_size=5, collate_fn=action_collate_fn)
num_epochs = 100

for epoch in range(num_epochs):
    ce_losses = []
    mse_losses = []

    for batch_obs, batch_act, batch_sap, lengths in dataloader:
        start = 0
        max_seq_len = 505
        batch_size = batch_obs.shape[0]

        batch_obs = batch_obs.to(device)[:, start:start+max_seq_len]  # (batch_size, max_seq_len, input_dim)
        batch_act = batch_act.to(device)[:, start:start+max_seq_len]  # (batch_size, max_seq_len, target_dim=16)
        batch_sap = batch_sap.to(device)[:, start:start+max_seq_len]  # (batch_size, max_seq_len, target_dim=32)

        output = model(batch_obs, batch_first=True)  # (max_seq_len, batch_size, output_dim=128)
        output = output.transpose(1, 0)              # (batch_size, max_seq_len, output_dim=128)
        ce_loss, mse_loss = compute_loss(output, batch_act, batch_sap)
        total_loss = ce_loss  # + mse_loss

        ce_losses.append(ce_loss.item())
        mse_losses.append(mse_loss.item())

        print(f"CE Loss: {ce_losses[-1]:.4f} MSE Loss: {mse_losses[-1]:.4f}")

        # Update weights
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    avg_ce_loss = sum(ce_losses) / len(ce_losses)
    avg_mse_loss = sum(mse_losses) / len(mse_losses)

    print(f"Epoch {epoch+1}/{num_epochs}, CE Loss: {avg_ce_loss:.4f} MSE Loss: {avg_mse_loss:.4f}")

    torch.save(model, 'model.pth')

