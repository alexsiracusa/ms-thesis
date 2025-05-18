from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams, env_params_ranges
from luxai_s3.state import EnvObs, UnitState, MapTile

import gymnasium
from gymnasium import spaces
import numpy as np
import jax
import jax.numpy as jnp
import chex
from typing import Dict, Any, Tuple, Optional, Union


class LuxEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "name": "LuxAIS3Gymnasium-v0"}

    def __init__(self,
                 auto_reset_lux: bool = False,
                 fixed_env_params: Optional[EnvParams] = None,
                 lux_env_params: Optional[EnvParams] = None,
                 render_mode: Optional[str] = None):
        """
        Gymnasium wrapper for the Lux AI Season 3 JAX environment.

        Args:
            auto_reset_lux: Whether the underlying LuxAIS3Env should auto-reset.
            fixed_env_params: Fixed parameters for map shapes, max units, etc.
                              If None, a default EnvParams() is used.
            lux_env_params: Runtime environment parameters for LuxAIS3Env.
                            If None, fixed_env_params or LuxAIS3Env's default_params are used.
            render_mode: Gymnasium render mode ('human' or None).
        """
        super().__init__()

        self.fixed_params = fixed_env_params if fixed_env_params is not None else EnvParams()
        self.env = LuxAIS3Env(auto_reset=auto_reset_lux, fixed_env_params=self.fixed_params)

        # Use provided lux_env_params or default from the underlying env for defining spaces
        self.runtime_params = lux_env_params if lux_env_params is not None else self.env.default_params

        # Ensure runtime_params has all necessary attributes from fixed_params if not overridden
        # This is a bit of a merge logic, assuming fixed_params are for static shapes
        # and runtime_params can override behavior/dynamic limits.
        # For space definition, we need consistent values.
        # Typically, space definition parameters come from fixed_params or a canonical EnvParams.
        space_def_params = self.runtime_params

        # Define action space
        # Action: For each unit: [action_type (0-5), delta_x, delta_y]
        # action_type: 0=none, 1=up, 2=right, 3=down, 4=left, 5=sap
        # delta_x, delta_y: for sap action, relative coords.
        action_shape = (self.fixed_params.max_units, 3)
        low_action = np.zeros(action_shape, dtype=np.int32)
        high_action = np.zeros(action_shape, dtype=np.int32)

        low_action[:, 0] = 0  # Min action type
        high_action[:, 0] = 5  # Max action type (0-5)

        # Sap range: e.g., if unit_sap_range is 3, deltas can be -3, -2, -1, 0, 1, 2, 3
        # The original code uses `env_params_ranges["unit_sap_range"][-1]`
        # Let's use the value from EnvParams if available, or fallback
        sap_range_val = getattr(space_def_params, 'unit_sap_range', env_params_ranges["unit_sap_range"][-1])

        low_action[:, 1:] = -sap_range_val
        high_action[:, 1:] = sap_range_val

        action_space_per_player = spaces.Box(
            low=low_action, high=high_action, dtype=np.int32
        )
        self.action_space = spaces.Dict(
            {f"player_{i}": action_space_per_player for i in range(self.fixed_params.num_teams)}
        )

        # Define observation space (this is complex)
        # Based on the structure of EnvObs and how `get_obs` tailors it per player
        num_teams = self.fixed_params.num_teams
        max_units = self.fixed_params.max_units
        map_w, map_h = self.fixed_params.map_width, self.fixed_params.map_height
        max_relics = self.fixed_params.max_relic_nodes

        # Get limits from space_def_params
        max_unit_energy = space_def_params.max_unit_energy
        min_unit_energy = space_def_params.min_unit_energy  # Assuming this exists or default to -max_unit_energy
        max_map_energy = space_def_params.max_energy_per_tile
        min_map_energy = space_def_params.min_energy_per_tile
        # Tile types: 0 (empty), 1 (asteroid), 2 (nebula). Masked obs uses -1.
        max_tile_type = 2
        min_tile_type = -1  # For masked/unknown

        obs_space_per_player = spaces.Dict({
            "units_position": spaces.Box(low=-1, high=max(map_w, map_h) - 1, shape=(num_teams, max_units, 2),
                                         dtype=np.int16),
            "units_energy": spaces.Box(low=min_unit_energy, high=max_unit_energy, shape=(num_teams, max_units),
                                       dtype=np.int16),
            "units_mask": spaces.Box(low=0, high=1, shape=(num_teams, max_units), dtype=bool),
            "sensor_mask": spaces.Box(low=0, high=1, shape=(map_h, map_w), dtype=bool),  # JAX env uses (H, W) for this
            "map_energy": spaces.Box(low=min_map_energy if min_map_energy != -1 else -1, high=max_map_energy,
                                     shape=(map_w, map_h), dtype=np.int16),  # JAX env uses (W,H)
            "map_tile_type": spaces.Box(low=min_tile_type, high=max_tile_type, shape=(map_w, map_h), dtype=np.int16),
            # JAX env uses (W,H)
            "team_points": spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(num_teams,), dtype=np.int32),
            "team_wins": spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(num_teams,), dtype=np.int32),
            "steps": spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(), dtype=np.int32),
            "match_steps": spaces.Box(low=-1, high=space_def_params.max_steps_in_match, shape=(), dtype=np.int32),
            "relic_nodes_positions": spaces.Box(low=-1, high=max(map_w, map_h) - 1, shape=(max_relics, 2),
                                                dtype=np.int16),
            "relic_nodes_mask": spaces.Box(low=0, high=1, shape=(max_relics,), dtype=bool)
        })

        self.observation_space = spaces.Dict(
            {f"player_{i}": obs_space_per_player for i in range(self.fixed_params.num_teams)}
        )

        self.render_mode = render_mode
        self.jax_prng_key: Optional[chex.PRNGKey] = None  # Initialized in reset
        self.current_state: Optional[Any] = None  # Stores the JAX EnvState

    def _convert_jax_obs_to_numpy_dict(self, jax_obs_pytree: EnvObs) -> Dict[str, np.ndarray]:
        """Converts a JAX EnvObs PyTree to a NumPy dictionary matching the defined obs space."""
        # Ensure field names match obs_space_per_player keys
        return {
            "units_position": np.asarray(jax_obs_pytree.units.position),
            "units_energy": np.asarray(jax_obs_pytree.units.energy),
            "units_mask": np.asarray(jax_obs_pytree.units_mask),
            "sensor_mask": np.asarray(jax_obs_pytree.sensor_mask),
            "map_energy": np.asarray(jax_obs_pytree.map_features.energy),
            "map_tile_type": np.asarray(jax_obs_pytree.map_features.tile_type),
            "team_points": np.asarray(jax_obs_pytree.team_points),
            "team_wins": np.asarray(jax_obs_pytree.team_wins),
            "steps": np.asarray(jax_obs_pytree.steps),
            "match_steps": np.asarray(jax_obs_pytree.match_steps),
            "relic_nodes_positions": np.asarray(jax_obs_pytree.relic_nodes),
            "relic_nodes_mask": np.asarray(jax_obs_pytree.relic_nodes_mask),
        }

    def _convert_actions_to_jax(self, actions: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
        """Converts NumPy actions from Gymnasium to JAX arrays for LuxAIS3Env."""
        # return jax.tree_map(jnp.asarray, actions)
        return jax.tree.map(jnp.asarray, actions)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
        super().reset(seed=seed, options=options)

        if seed is not None:
            self.jax_prng_key = jax.random.PRNGKey(seed)
        elif self.jax_prng_key is None:  # First reset and no seed provided
            self.jax_prng_key = jax.random.PRNGKey(0)  # Default seed

        key_reset, self.jax_prng_key = jax.random.split(self.jax_prng_key)

        # The underlying env.reset returns obs (dict player -> EnvObs_pytree), state
        jax_obs_dict, self.current_state = self.env.reset(key_reset, params=self.runtime_params)

        # Convert observations for each player
        numpy_obs_dict = {
            player_id: self._convert_jax_obs_to_numpy_dict(player_obs_pytree)
            for player_id, player_obs_pytree in jax_obs_dict.items()
        }

        # Info dict (can be extended)
        # For multi-agent, Gymnasium expects infos to be a dict player_id -> player_info
        infos = {f"player_{i}": {} for i in range(self.fixed_params.num_teams)}

        if self.render_mode == "human":
            self.render()

        return numpy_obs_dict, infos

    def step(self, actions: Dict[str, np.ndarray]
             ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, float],
    Dict[str, bool], Dict[str, bool], Dict[str, Any]]:

        if self.jax_prng_key is None:
            raise RuntimeError("Environment must be reset before calling step.")
        if self.current_state is None:
            raise RuntimeError("Environment state is not initialized. Call reset first.")

        key_step, self.jax_prng_key = jax.random.split(self.jax_prng_key)

        # Convert actions to JAX arrays
        jax_actions = self._convert_actions_to_jax(actions)

        # Underlying env.step returns:
        # obs_dict (player -> EnvObs_pytree), next_state, rewards_dict, terminated_dict, truncated_dict, infos_dict
        jax_obs_dict, self.current_state, rewards_dict, terminated_dict, truncated_dict, infos_dict = \
            self.env.step(key_step, self.current_state, jax_actions, params=self.runtime_params)

        # Convert observations for each player
        numpy_obs_dict = {
            player_id: self._convert_jax_obs_to_numpy_dict(player_obs_pytree)
            for player_id, player_obs_pytree in jax_obs_dict.items()
        }

        # Convert rewards, terminated, truncated, infos to NumPy if they are JAX types
        # (though they are likely already Python bools/floats in the mock)
        np_rewards_dict = jax.tree.map(np.asarray, rewards_dict)
        np_terminated_dict = jax.tree.map(np.asarray, terminated_dict)
        np_truncated_dict = jax.tree.map(np.asarray, truncated_dict)
        # infos_dict from JAX env is player_id -> dict. Ensure inner values are serializable if needed.
        # For now, assume they are basic Python types.

        if self.render_mode == "human":
            self.render()

        return numpy_obs_dict, np_rewards_dict, np_terminated_dict, np_truncated_dict, infos_dict

    def render(self):
        if self.render_mode == "human":
            if self.current_state is not None:
                # The original LuxAIS3Env.render takes state and params
                self.env.render(self.current_state, self.runtime_params)
            else:
                print("Cannot render: current state is None. Call reset first.")
        elif self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify a render_mode (\"human\") "
                " राजीव ( राजीव ) pass render_mode=\"human\" to the wrapper constructor."
            )
            return
        # Add "rgb_array" support if LuxAIPygameRenderer can return images

    def close(self):
        # If the underlying environment has a renderer that needs closing:
        if hasattr(self.env, 'renderer') and hasattr(self.env.renderer, 'close'):
            self.env.renderer.close()
        print("GymnasiumLuxAIS3Wrapper closed.")


if __name__ == '__main__':
    print("Testing GymnasiumLuxAIS3Wrapper with MOCKED LuxAIS3Env...")

    # Example: Using default EnvParams for fixed and runtime
    env_params = EnvParams(map_width=16, map_height=16, max_units=5, num_teams=2)

    # If you have the full LuxAI S3 code, you would pass the actual EnvParams:
    # from luxai_s3.params import EnvParams
    # env_params = EnvParams(map_width=32, map_height=32, ...)

    print(
        f"Using EnvParams: map_size=({env_params.map_width},{env_params.map_height}), max_units={env_params.max_units}")

    try:
        env = LuxEnv(
            fixed_env_params=env_params,  # For shapes
            lux_env_params=env_params,  # For runtime behavior and potential dynamic limits
            render_mode="human"
        )

        print("Action Space:", env.action_space)
        print("Observation Space (player_0):", env.observation_space["player_0"])

        # Test reset
        print("\nResetting environment...")
        obs, info = env.reset(seed=42)
        print("Initial observations received for players:", list(obs.keys()))
        print("Obs shape for player_0 units_position:", obs["player_0"]["units_position"].shape)
        print("Info from reset:", info)

        # Test step
        print("\nStepping environment with random actions...")
        # Create some dummy actions based on the action space
        # For multi-agent, actions is a dict: player_id -> action_array
        actions = {}
        for player_id in obs.keys():  # obs keys are "player_0", "player_1", ...
            actions[player_id] = env.action_space[player_id].sample()
            print(f"Sampled action for {player_id} shape: {actions[player_id].shape}")

        obs, rewards, terminated, truncated, infos = env.step(actions)

        print("\n--- Step Results ---")
        print("Observations received for players:", list(obs.keys()))
        print("Obs shape for player_0 units_position:", obs["player_0"]["units_position"].shape)
        print("Rewards:", rewards)
        print("Terminated flags:", terminated)
        print("Truncated flags:", truncated)
        print("Infos from step:", infos)

        print("\nSimulating a few steps...")
        for i in range(100):
            actions = {p_id: env.action_space[p_id].sample() for p_id in obs.keys()}
            obs, rewards, term, trunc, infos = env.step(actions)
            print(f"Step {i + 1}: Rewards: {rewards}, Term: {term['player_0']}, Trunc: {trunc['player_0']}")
            # Example of checking combined done
            done_any_player = any(term.values()) or any(trunc.values())
            if done_any_player:
                print("Episode finished.")
                break

        env.close()

    except ImportError as e:
        print(f"ImportError: {e}. Make sure LuxAI S3 environment code and its dependencies are available.")
        print(
            "The wrapper was tested with a MOCKED LuxAIS3Env. For full functionality, provide the actual environment.")
    except AttributeError as e:
        print(f"AttributeError: {e}. This might be due to differences between MOCKED components and actual ones.")
        print("Please ensure EnvParams and other data structures have the expected fields.")


gymnasium.register(
    id="gymnasium_env/Lux",
    entry_point=LuxEnv,
)