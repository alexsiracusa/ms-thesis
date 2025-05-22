from build_sequential2d import build_sequential2d
from lux.util import load_action_dataset

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f'Device: {device}')

model_sizes = input_sizes + hidden_sizes + output_sizes
model = build_sequential2d(model_sizes, num_input_blocks=len(input_sizes), num_iterations=3)
model.to(device)
model.train()

ce_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

dataloader = load_action_dataset('../data', device=device)
num_epochs = 10


for epoch in range(num_epochs):
    total_ce_loss = 0.0
    total_mse_loss = 0.0

    for batch_obs, batch_act in dataloader:
        optimizer.zero_grad()

        output = model(batch_obs)
        print(output)
        total_loss = torch.tensor(0.0, device=device)

        # For each agent we want a separate CE loss for action type and MSE loss for the sap actions delta x and y
        # for i in range(0, 128, 8):
        #     loss1 = ce_loss(output[i:i+6], action[i:i+6])
        #     loss2 = mse_loss(output[i+6:i+8], action[i+6:i+8])
        #
        #     total_loss += loss1 + loss2
        #
        #     total_ce_loss += loss1.item()
        #     total_mse_loss += loss2.item()
        #
        # total_loss.backward()
        # optimizer.step()  # update weights

    avg_ce_loss = total_ce_loss / len(dataloader) / 16
    avg_mse_loss = total_mse_loss / len(dataloader) / 16

    print(f"Epoch {epoch+1}/{num_epochs}, CE Loss: {avg_ce_loss:.4f} MSE Loss: {avg_mse_loss:.4f}")



