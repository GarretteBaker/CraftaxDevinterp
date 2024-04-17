#%%
import jax
import jax.numpy as jnp
from typing import NamedTuple
import orbax
import craftax
from craftax.environment_base.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)
from craftax.models.actor_critic import (
    ActorCritic,
)
import orbax.checkpoint as ocp
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
import pickle
import os
from tqdm import tqdm
from craftax.craftax.full_view_constants import *
from craftax.craftax.craftax_state import EnvState
from craftax.craftax import renderer
from craftax.craftax.util.game_logic_utils import is_boss_vulnerable
import matplotlib.pyplot as plt
import pickle

#%%
DIRECTIONS = jnp.concatenate(
    (
        jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32),
        jnp.zeros((11, 2), dtype=jnp.int32),
    ),
    axis=0,
)

def log_placement(past_state, present_state):
    placement = jnp.zeros((4))
    block_position = past_state.player_position + DIRECTIONS[past_state.player_direction]
    past_type = past_state.map[past_state.player_level, block_position[0], block_position[1]]
    present_type = present_state.map[past_state.player_level, block_position[0], block_position[1]]
    is_same = (
        past_type
        == present_type
    )
    present_is_stone = (
        present_type
        == BlockType.STONE.value
    )
    present_is_table = (
        present_type 
        == BlockType.CRAFTING_TABLE.value
    )
    present_is_furnace = (
        present_type
        == BlockType.FURNACE.value
    )
    present_is_plant = (
        present_type
        == BlockType.PLANT.value
    )

    delta_stone = jnp.logical_and(
        is_same, present_is_stone
    )
    delta_table = jnp.logical_and(
        is_same, present_is_table
    )
    delta_furnace = jnp.logical_and(
        is_same, present_is_furnace
    )
    delta_plant = jnp.logical_and(
        is_same, present_is_plant
    )

    placement = placement.at[0].set(delta_stone)
    placement = placement.at[1].set(delta_table)
    placement = placement.at[2].set(delta_furnace)
    placement = placement.at[3].set(delta_plant)

    return placement

def log_mining(past_state, env_state):
    mined = jnp.zeros((13))
    block_position = past_state.player_position + DIRECTIONS[past_state.player_direction]
    past_type = past_state.map[past_state.player_level, block_position[0], block_position[1]]
    present_type = present_state





def log_tracking(past_state, env_state, action):
    block_placements = log_placement(past_state, env_state)
    block_mining = log_mining(past_state, env_state)
    player_location = env_state.player_position
    player_movement = log_movement(past_state, env_state)
    revealed_blocks = log_discovery(past_state, env_state) # boolean array, we can OR then up later
    doings = log_do(past_state, action) # includes things like drinking and such, but not all actions, since some are easy to determine via environment & inventory changes
    craftings = log_crafting(past_state, env_state) # look at differences in inventory
    mob_kills = log_killing(past_state, env_state)




#%%
num_steps = 100
def generate_trajectory(network_params, rng, num_envs=1, num_steps=num_steps):
    env = CraftaxSymbolicEnv()
    env_params = env.default_params
    env = LogWrapper(env)
    env = OptimisticResetVecEnvWrapper(
        env,
        num_envs=num_envs,
        reset_ratio=min(16, num_envs),
    )
    network = ActorCritic(env.action_space(env_params).n, 512)

    class Transition(NamedTuple):
        state: jnp.ndarray
        done: jnp.ndarray

    # COLLECT TRAJECTORIES
    def _env_step(runner_state, unused):
        (
            past_state,
            last_obs,
            rng,
        ) = runner_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi, _ = network.apply(network_params, last_obs)
        action = pi.sample(seed=_rng)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state, _, done, _ = env.step(
            _rng, past_state, action, env_params
        )

        tracker = log_tracking(past_state, env_state, action)

        transition = Transition(
            state = env_state,
            done = done
        )
        runner_state = (
            env_state,
            obsv,
            rng,
        )
        return runner_state, transition
    rng, _rng = jax.random.split(rng)
    obsv, env_state = env.reset(_rng, env_params)

    rng, _rng = jax.random.split(rng)
    runner_state = (
        env_state,
        obsv,
        _rng,
    )

    runner_state, traj_batch = jax.lax.scan(
        _env_step, runner_state, None, num_steps
    )
    return traj_batch.state, traj_batch.done
#%%
