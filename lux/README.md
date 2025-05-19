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
luxai-s3 path/to/bot/main.py path/to/bot/main.py --output replay.json
```
Then upload the replay.json to the online visualizer here: https://s3vis.lux-ai.org/ (a link on the lux-ai.org website will be up soon)

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
    "env_cfg": dict // some of the game's visible parameters
  }
}
```

Example `env_cfg` dictionary
```json
"env_cfg": {
  "max_units": 16,
  "match_count_per_episode": 5,
  "max_steps_in_match": 100,
  "map_height": 24,
  "map_width": 24,
  "num_teams": 2,
  "unit_move_cost": 2,
  "unit_sap_cost": 45,
  "unit_sap_range": 5,
  "unit_sensor_range": 3
}

```



