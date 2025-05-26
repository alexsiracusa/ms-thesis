from torch.utils.data import DataLoader
from build_sequential2d import build_sequential2d
from lux.util import load_action_dataset, action_collate_fn

import torch
import torch.nn as nn
import torch.optim as optim


# T is the number of teams (default is 2)          2
# N is the max number of units per team            16
# W, H are the width and height of the map         24x24
# R is the max number of relic nodes               6

# Total = 3619
input_sizes = [
            # SHAPE                    DIMS     NAME                    TYPE
    64,     # (T, N, 2) = 2 * 16 * 2  = 64       units position          coordinates
    32,     # (T, N, 1) = 2 * 16 * 1  = 32       units energy            continuous
    32,     # (T, N)    = 2 * 16      = 32       units_mask              binary
    576,    # (W, H)    = 24 * 24     = 576      sensor_mask             binary
    576,    # (W, H)    = 24 * 24     = 576      map_features energy     continuous

    2304,   # (W, H, 4) = 24 * 24 * 4 = 2304     map_features tile_type  categorical
            # UNKNOWN = -1
            # EMPTY_TILE = 0
            # NEBULA_TILE = 1
            # ASTEROID_TILE = 2

    6,      # (R)       = 6           = 6        relic_nodes_mask        binary
    12,     # (R, 2)    = 6 * 2       = 12       relic_nodes             coordinates
    2,      # (T)       = 2           = 2        team_points             continuous
    2,      # (T)       = 2           = 2        team_wins               continuous
    1,      # (1)       = 1           = 1        steps                   continuous
    1,      # (1)       = 1           = 1        match_steps             continuous
    1,      # (1)       = 1           = 1        remainingOverageTime    continuous

    1,      # (1)       = 1           = 1        max_units               CONSTANT = 16
    1,      # (1)       = 1           = 1        match_count_per_episode continuous
    1,      # (1)       = 1           = 1        max_steps_in_match      continuous
    1,      # (1)       = 1           = 1        map_height              CONSTANT = 24
    1,      # (1)       = 1           = 1        map_width               CONSTANT = 24
    1,      # (1)       = 1           = 1        num_teams               CONSTANT = 2
    1,      # (1)       = 1           = 1        unit_move_cost          continuous
    1,      # (1)       = 1           = 1        unit_sap_cost           continuous
    1,      # (1)       = 1           = 1        unit_sap_range          continuous
    1,      # (1)       = 1           = 1        unit_sensor_range       continuous
]

# total = 4620
hidden_sizes = [
    2304,   # spaces to remember map features
    576,
    576,
    64,     # spaces to remember unit information
    64,
    64,
    12,     # space to remember relic nodes

    144,    # Generic extra spaces for the model to figure out
    144,
    144,
    144,
    64,
    64,
    64,
    64,
    64,
    64,
]

# 5 possible actions
#   DO_NOTHING = 0
#   MOVE_UP = 1
#   MOVE_RIGHT = 2
#   MOVE DOWN = 3
#   MOVE_LEFT = 4
#   SAP = 5

# With two additional parameters for the SAP action
#   DELTA_X = int
#   DELTA_Y = int
output_sizes = [
    128     # (N, 8)    = 16 * 8    = 128       actions                 categorical
]

print(f'Input Total:  {sum(input_sizes)}')
print(f'Hidden Total: {sum(hidden_sizes)}')


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
        max_seq_len = 20
        batch_size = batch_obs.shape[0]

        batch_obs = batch_obs.to(device)[:, start:start+max_seq_len]  # (batch_size, max_seq_len, input_dim)
        batch_act = batch_act.to(device)[:, start:start+max_seq_len]  # (batch_size, max_seq_len, target_dim=16)
        batch_sap = batch_sap.to(device)[:, start:start+max_seq_len]  # (batch_size, max_seq_len, target_dim=32)

        output = model(batch_obs, batch_first=True)  # (batch_size, max_seq_len, output_dim=48)
        ce_loss, mse_loss = compute_loss(output, batch_act, batch_sap)
        total_loss = ce_loss # + mse_loss

        num_ce = batch_size * max_seq_len * 16
        num_mse = batch_size * max_seq_len * 32
        # print(batch_act[0][0][0], output[0][0][:6])
        print(f"CE Loss: {ce_loss.item() / num_ce:.4f} MSE Loss: {mse_loss.item() / num_mse:.4f}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    avg_ce_loss = total_ce_loss / len(dataloader) / 16
    avg_mse_loss = total_mse_loss / len(dataloader) / 16

    print(f"Epoch {epoch+1}/{num_epochs}, CE Loss: {avg_ce_loss:.4f} MSE Loss: {avg_mse_loss:.4f}")



