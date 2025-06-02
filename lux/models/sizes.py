
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
    96,     # (N, 6)    = 16 * 6    = 96         actions                 categorical
    32,     # (N, 2)    = 16 * 2    = 32         actions                 continuous (int)
]

if __name__ == '__main__':
    print(f'Input Total:  {sum(input_sizes)}')
    print(f'Hidden Total: {sum(hidden_sizes)}')
    print(f'Output Total: {sum(output_sizes)}')
    print(f'-------------------------')
    print(f'Grand Total:  {sum(input_sizes + hidden_sizes + output_sizes)}')