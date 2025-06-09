from compute_loss import compute_loss
from lux.util import load_action_dataset, action_collate_fn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

dataset = load_action_dataset('../data/test')
dataloader = DataLoader(dataset, batch_size=6, collate_fn=action_collate_fn)

model = torch.load('model_0_505-4.pth', weights_only=False)
model.to(device)
model.eval()


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

    # print actions for random agent
    actions = []
    pred_actions = []
    for _ in range(100):
        agent_num = int(random.random() * 16)
        sample_num = int(random.random() * 6)
        step_num = int(random.random() * max_seq_len)

        act = batch_act[sample_num, step_num, agent_num]
        sap = batch_sap[sample_num, step_num, agent_num*2:agent_num*2+2]

        pred = output[:, :, -128:]
        pred_act = pred[sample_num, step_num, agent_num*6:agent_num*6+6]
        pred_sap = pred[sample_num, step_num, 96+agent_num*2:96+agent_num*2+2]

        actions.append(act)
        pred_actions.append(pred_act)

        print(act.tolist(), pred_act.tolist())

    ce_loss_fn = nn.CrossEntropyLoss()
    print(ce_loss_fn(torch.stack(pred_actions), torch.stack(actions)).item())

avg_ce_loss = sum(ce_losses) / len(ce_losses)
avg_mse_loss = sum(mse_losses) / len(mse_losses)

print(f"CE Loss: {avg_ce_loss:.4f} MSE Loss: {avg_mse_loss:.4f}")



