import torch
import numpy as np

"""
Observation JSON Format

{
  "obs": {
    "units": {
      "position": Array(T, N, 2),
      "energy": Array(T, N, 1)
    },
    // whether the unit exists and is visible to you. units_mask[t][i] is whether team t's unit i can be seen and exists.
    "units_mask": Array(T, N),
    // whether the tile is visible to the unit for that team
    "sensor_mask": Array(W, H),
    "map_features": {
        // amount of energy on the tile
        "energy": Array(W, H),
        // type of the tile. 0 is empty, 1 is a nebula tile, 2 is asteroid
        "tile_type": Array(W, H)
    },
    // whether the relic node exists and is visible to you.
    "relic_nodes_mask": Array(R),
    // position of the relic nodes.
    "relic_nodes": Array(R, 2),
    // points scored by each team in the current match
    "team_points": Array(T),
    // number of wins each team has in the current game/episode
    "team_wins": Array(T),
    // number of steps taken in the current game/episode
    "steps": int,
    // number of steps taken in the current match
    "match_steps": int
  },
  // number of steps taken in the current game/episode
  "remainingOverageTime": int, // total amount of time your bot can use whenever it exceeds 2s in a turn
  "player": str, // your player id
  "info": {
    "env_cfg": {
      "max_units": int,
      "match_count_per_episode": int,
      "max_steps_in_match": int,
      "map_height": int,
      "map_width": int,
      "num_teams": int,
      "unit_move_cost": int,
      "unit_sap_cost": int,
      "unit_sap_range": int,
      "unit_sensor_range": int
    }
  }
}
"""

"""
Output Tensors Format:

[
            # SHAPE                    DIMS     NAME                    TYPE
    64,     # (T, N, 2) = 2 * 16 * 2 = 64       units position          coordinates
    32,     # (T, N, 1) = 2 * 16 * 1 = 32       units energy            continuous
    32,     # (T, N)    = 2 * 16     = 32       units_mask              binary
    576,    # (W, H)    = 24 * 24    = 576      sensor_mask             binary
    576,    # (W, H)    = 24 * 24    = 576      map_features energy     continuous
    576,    # (W, H)    = 24 * 24    = 576      map_features tile_type  categorical
    6,      # (R)       = 6          = 6        relic_nodes_mask        binary
    12,     # (R, 2)    = 6 * 2      = 12       relic_nodes             coordinates
    2,      # (T)       = 2          = 2        team_points             continuous
    2,      # (T)       = 2          = 2        team_wins               continuous
    1,      # (1)       = 1          = 1        steps                   continuous
    1,      # (1)       = 1          = 1        match_steps             continuous
    1,      # (1)       = 1          = 1        remainingOverageTime    continuous

    1,      # (1)       = 1          = 1        max_units               CONSTANT = 16
    1,      # (1)       = 1          = 1        match_count_per_episode continuous
    1,      # (1)       = 1          = 1        max_steps_in_match      continuous
    1,      # (1)       = 1          = 1        map_height              CONSTANT = 24
    1,      # (1)       = 1          = 1        map_width               CONSTANT = 24
    1,      # (1)       = 1          = 1        num_teams               CONSTANT = 2
    1,      # (1)       = 1          = 1        unit_move_cost          continuous
    1,      # (1)       = 1          = 1        unit_sap_cost           continuous
    1,      # (1)       = 1          = 1        unit_sap_range          continuous
    1,      # (1)       = 1          = 1        unit_sensor_range       continuous
]
"""


"""
Args:
    obs: json of the form specified above

Returns:
   tensors: tensor representation of the json in the form specified above
"""
def obs_to_tensors(obs):
    tensors = [
        np.array(obs['obs']['units']['position']).flatten(),
        np.array(obs['obs']['units']['energy']).flatten(),
        np.array(obs['obs']['units_mask']).flatten(),
        np.array(obs['obs']['sensor_mask']).flatten(),
        np.array(obs['obs']['map_features']['energy']).flatten(),
        np.array(obs['obs']['map_features']['tile_type']).flatten(),
        np.array(obs['obs']['relic_nodes_mask']).flatten(),
        np.array(obs['obs']['relic_nodes']).flatten(),
        np.array(obs['obs']['team_points']).flatten(),
        np.array(obs['obs']['team_wins']).flatten(),
        np.array([obs['obs']['steps']]),
        np.array([obs['obs']['match_steps']]),
        np.array([obs['remainingOverageTime']]),

        np.array([obs['info']['env_cfg']['max_units']]),
        np.array([obs['info']['env_cfg']['match_count_per_episode']]),
        np.array([obs['info']['env_cfg']['max_steps_in_match']]),
        np.array([obs['info']['env_cfg']['map_height']]),
        np.array([obs['info']['env_cfg']['map_width']]),
        np.array([obs['info']['env_cfg']['num_teams']]),
        np.array([obs['info']['env_cfg']['unit_move_cost']]),
        np.array([obs['info']['env_cfg']['unit_sap_cost']]),
        np.array([obs['info']['env_cfg']['unit_sap_range']]),
        np.array([obs['info']['env_cfg']['unit_sensor_range']]),
    ]

    return [torch.tensor(tensor, dtype=torch.float32) for tensor in tensors]