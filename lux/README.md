## Setup Instructions
These instructions are partially copied and modified from the Lux-Design-S3 repository found here: https://github.com/Lux-AI-Challenge/Lux-Design-S3/

It is recommended to use a python package manager like conda/mamba to install the dependencies.
```shell
mamba create -n "lux-s3" "python==3.11"
pip install "git+https://github.com/Lux-AI-Challenge/Lux-Design-S3.git@main#egg=luxai-s3&subdirectory=src"
```
The first command creates a virtual enviornment called "lux-s3" which you should set as your interpreter. 
The second command installs the `src` folder of the [`Lux-Design-S3`](https://github.com/Lux-AI-Challenge/Lux-Design-S3/) GitHub repository as a python package which can be imported in any file with `import luxai_s3`

To verify your installation, you can run a match between two random agents:
```shell
luxai-s3 --help
```
```shell
luxai-s3 agents/kit_agent.py agents/kit_agent.py --output replay.json
```
Then upload the replay.json to the online visualizer here: https://s3vis.lux-ai.org/ (a link on the lux-ai.org website will be up soon)

This will also create two output files `actions_0.json` and `actions_1.json` which will show all of the agents observations and corresponding actions.

## Agent Observation Space
```json
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
    // some of the game's visible parameters
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
```

## Agent Action Space
The action space of the game is always a fixed (N, 3) array of integers to control up to units 0 to N-1 on your team where N is the max number of units each team can have (example code shows how to determine N). At any given point in time you might not have N units on your team so actions for those nonexistent units do not do anything.

For each unit's action, the first integer indicates the type of action, which can be 0 for doing nothing, 1 to move up, 2 to move right, 3 to move down, 4 to move left, and 5 to sap a tile. The next 2 integers are only for the sap action and indicate the location of the tile to sap from relative to the unit's position (a delta x and y value).

For implementing a model, it is often easier to represent this as an (N, 8) space instead where the second dimension of 8 represents each unique action (do nothing, move up, down, left, right, sap, sap delta x, sap delta y) as a one-hot encoding (aside from the sap delta options)



