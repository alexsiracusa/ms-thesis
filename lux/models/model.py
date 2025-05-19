
# T is the number of teams (default is 2)          2
# N is the max number of units per team            16
# W, H are the width and height of the map         24x24
# R is the max number of relic nodes               6

input_sizes = [
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




