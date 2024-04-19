#%%
import jax
import jax.numpy as jnp
from typing import NamedTuple
import orbax
import craftax
from craftax.environment_base.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    AutoResetEnvWrapper,
    BatchEnvWrapper,
)
from craftax.models.actor_critic import (
    ActorCritic,
)
from craftax.craftax.util.game_logic_utils import *
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
import scipy as sp

#%%
DIRECTIONS = jnp.concatenate(
    (
        jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32),
        jnp.zeros((11, 2), dtype=jnp.int32),
    ),
    axis=0,
)

def log_placement(past_state, present_state): # TODO: Add torch logging
    placement = jnp.zeros((4))
    block_position = past_state.player_position + DIRECTIONS[past_state.player_direction]
    past_type = past_state.map[past_state.player_level][block_position[0], block_position[1]]
    present_type = present_state.map[past_state.player_level][block_position[0], block_position[1]]
    is_diff = (
        past_type
        != present_type
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
        is_diff, present_is_stone
    )
    delta_table = jnp.logical_and(
        is_diff, present_is_table
    )
    delta_furnace = jnp.logical_and(
        is_diff, present_is_furnace
    )
    delta_plant = jnp.logical_and(
        is_diff, present_is_plant
    )

    placement = placement.at[0].set(delta_stone)
    placement = placement.at[1].set(delta_table)
    placement = placement.at[2].set(delta_furnace)
    placement = placement.at[3].set(delta_plant)

    return placement

def log_mining(past_state, present_state):
    mined = jnp.zeros((13))
    block_position = past_state.player_position + DIRECTIONS[past_state.player_direction]
    past_type = past_state.map[past_state.player_level][block_position[0], block_position[1]]
    present_type = present_state.map[past_state.player_level][block_position[0], block_position[1]]

    is_diff = (
        past_type
        != present_type
    )
    past_is_tree = (
        past_type
        == BlockType.TREE.value
    )
    past_is_stone = (
        past_type
        == BlockType.STONE.value
    )
    past_is_furnace = (
        past_type
        == BlockType.FURNACE.value
    )
    past_is_table = (
        past_type
        == BlockType.CRAFTING_TABLE.value
    )
    past_is_coal = (
        past_type
        == BlockType.COAL.value
    )
    past_is_iron = (
        past_type
        == BlockType.IRON.value
    )
    past_is_diamond = (
        past_type
        == BlockType.DIAMOND.value
    )
    past_is_sapphire = (
        past_type
        == BlockType.SAPPHIRE.value
    )
    past_is_ruby = (
        past_type
        == BlockType.RUBY.value
    )
    # past_is_sapling = (
    #     past_type
    #     == BlockType.SAPLING.value
    # )
    past_is_unripe_plant = (
        past_type
        == BlockType.PLANT.value
    )
    past_is_ripe_plant = (
        past_type
        == BlockType.RIPE_PLANT.value
    )
    past_is_plant = jnp.logical_or(
        past_is_unripe_plant, past_is_ripe_plant
    )
    past_is_stalagmite = (
        past_type
        == BlockType.STALAGMITE.value
    )
    past_is_chest = (
        past_type
        == BlockType.CHEST.value
    )

    delta_tree = jnp.logical_and(
        is_diff, past_is_tree
    )
    delta_stone = jnp.logical_and(
        is_diff, past_is_stone
    )
    delta_furnace = jnp.logical_and(
        is_diff, past_is_furnace
    )
    delta_table = jnp.logical_and(
        is_diff, past_is_table
    )
    delta_coal = jnp.logical_and(
        is_diff, past_is_coal
    )
    delta_iron = jnp.logical_and(
        is_diff, past_is_iron
    )
    delta_diamond = jnp.logical_and(
        is_diff, past_is_diamond
    )
    delta_sapphire = jnp.logical_and(
        is_diff, past_is_sapphire
    )
    delta_ruby = jnp.logical_and(
        is_diff, past_is_ruby
    )
    delta_plant = jnp.logical_and(
        is_diff, past_is_plant
    )
    delta_stalagmite = jnp.logical_and(
        is_diff, past_is_stalagmite
    )
    delta_chest = jnp.logical_and(
        is_diff, past_is_chest
    )

    mined = mined.at[0].set(delta_tree)
    mined = mined.at[1].set(delta_stone)
    mined = mined.at[2].set(delta_furnace)
    mined = mined.at[3].set(delta_table)
    mined = mined.at[4].set(delta_coal)
    mined = mined.at[5].set(delta_iron)
    mined = mined.at[6].set(delta_diamond)
    mined = mined.at[7].set(delta_sapphire)
    mined = mined.at[8].set(delta_ruby)
    mined = mined.at[9].set(delta_plant)
    mined = mined.at[10].set(delta_stalagmite)
    mined = mined.at[11].set(delta_chest)

    return mined

def log_movement(past_state, present_state):
    movement = jnp.zeros((4))
    past_position = past_state.player_position
    present_position = present_state.player_position
    delta_up = (
        present_position[0] < past_position[0]
    )
    # print(f"movement: {movement}")
    # print(f"present position: {present_position}")
    # print(f"past position: {past_position}")
    # print(f"delta up: {delta_up}")
    delta_down = (
        present_position[0] > past_position[0]
    )
    delta_left = (
        present_position[1] < past_position[1]
    )
    delta_right = (
        present_position[1] > past_position[1]
    )

    movement = movement.at[0].set(delta_up)
    movement = movement.at[1].set(delta_down)
    movement = movement.at[2].set(delta_left)
    movement = movement.at[3].set(delta_right)

    return movement

# def log_discovery(past_state, present_state):
#     revealed_blocks = jnp.zeros_like(past_state.map, dtype=jnp.bool)
#     size = 48

#     past_position = past_state.player_position
#     present_position = present_state.player_position
#     past_level = past_state.player_level
#     present_level = present_state.player_level

#     past_startx = jnp.maximum(past_position[0] - OBS_DIM[0]//2, 0)
#     past_starty = jnp.maximum(past_position[1] - OBS_DIM[1]//2, 0)
#     past_endx = jnp.minimum(past_position[0] + OBS_DIM[0]//2, size)
#     past_endy = jnp.minimum(past_position[1] + OBS_DIM[1]//2, size)
#     ones_square = jnp.ones((OBS_DIM[0], OBS_DIM[1]), dtype=jnp.bool)
#     cut_ones_square = jax.lax.dynamic_slice(
#         ones_square, 
#         (0, 0), 
#         (past_endx - past_startx, past_endy - past_starty)
#     )

#     revealed_blocks = jax.lax.dynamic_update_slice(
#         revealed_blocks, 
#         cut_ones_square, 
#         (past_level, past_startx, past_starty)
#     )

#     present_startx = jnp.maximum(present_position[0] - OBS_DIM[0]//2, 0)
#     present_starty = jnp.maximum(present_position[1] - OBS_DIM[1]//2, 0)
#     present_endx = jnp.minimum(present_position[0] + OBS_DIM[0]//2, size)
#     present_endy = jnp.minimum(present_position[1] + OBS_DIM[1]//2, size)
#     ones_square = jnp.ones((OBS_DIM[0], OBS_DIM[1]), dtype=jnp.bool)
#     cut_ones_square = jax.lax.dynamic_slice(
#         ones_square, 
#         (0, 0), 
#         (present_endx - present_startx, present_endy - present_starty)
#     )

#     revealed_blocks = jax.lax.dynamic_update_slice(
#         revealed_blocks, 
#         cut_ones_square, 
#         (present_level, present_startx, present_starty)
#     )

#     return revealed_blocks

def log_do(past_state, action):
    results = jnp.zeros((4))

    is_noop = (
        action 
        == Action.NOOP.value
    )
    is_do = (
        action
        == Action.DO.value
    )
    is_sleep = (
        action
        == Action.SLEEP.value
    )
    is_rest = (
        action
        == Action.REST.value
    )
    
    block_position = past_state.player_position + DIRECTIONS[past_state.player_direction]
    is_drinking_water = jnp.logical_and(
        is_do, jnp.logical_or(
            past_state.map[past_state.player_level][block_position[0], block_position[1]]
            == BlockType.WATER.value,
            past_state.map[past_state.player_level][block_position[0], block_position[1]]
            == BlockType.FOUNTAIN.value,
        )
    )

    results = results.at[0].set(is_noop)
    results = results.at[1].set(is_drinking_water)
    results = results.at[2].set(is_sleep)
    results = results.at[3].set(is_rest)

    return results

def log_killing(past_state, action):
    killed = jnp.zeros((3))
    attacked = jnp.zeros((3))
    block_position = past_state.player_position + DIRECTIONS[past_state.player_direction]
    _, killed_melee, attacked_melee, _ = attack_mob_class(
        past_state, 
        past_state.melee_mobs, 
        block_position,
        get_player_damage_vector(past_state),
        True, 
        1
    )
    _, killed_ranged, attacked_ranged, _ = attack_mob_class(
        past_state, 
        past_state.ranged_mobs, 
        block_position,
        get_player_damage_vector(past_state),
        True, 
        1
    )
    _, killed_passive, attacked_passive, _ = attack_mob_class(
        past_state,
        past_state.passive_mobs,
        block_position,
        get_player_damage_vector(past_state),
        True, 
        1
    )

    did_attack = (
        action
        == Action.DO.value
    )

    delta_kmelee = jnp.logical_and(
        did_attack, killed_melee
    )
    delta_kranged = jnp.logical_and(
        did_attack, killed_ranged
    )
    delta_kpassive = jnp.logical_and(
        did_attack, killed_passive
    )
    deltaamelee = jnp.logical_and(
        did_attack, attacked_melee
    )
    deltaaranged = jnp.logical_and(
        did_attack, attacked_ranged
    )
    deltaapassive = jnp.logical_and(
        did_attack, attacked_passive
    )

    killed = killed.at[0].set(delta_kmelee)
    killed = killed.at[1].set(delta_kranged)
    killed = killed.at[2].set(delta_kpassive)
    attacked = attacked.at[0].set(deltaamelee)
    attacked = attacked.at[1].set(deltaaranged)
    attacked = attacked.at[2].set(deltaapassive)

    return killed, attacked

class Tracker(NamedTuple):
    block_placements: jnp.ndarray
    block_mining: jnp.ndarray
    player_location: jnp.ndarray
    player_movement: jnp.ndarray
    # revealed_blocks: jnp.ndarray
    doings: jnp.ndarray
    mob_kills: jnp.ndarray
    mob_attacks: jnp.ndarray
    time: jnp.ndarray

def log_tracking(past_state, env_state, action):
    block_placements = log_placement(past_state, env_state)
    block_mining = log_mining(past_state, env_state)
    player_location = env_state.player_position
    player_movement = log_movement(past_state, env_state)
    # revealed_blocks = log_discovery(past_state, env_state) # boolean array, we can OR then up later
    doings = log_do(past_state, action) # includes things like drinking and such, but not all actions, since some are easy to determine via environment & inventory changes
    # craftings = log_crafting(past_state, env_state) # look at differences in inventory, ignoring for now because there's only really arrows and torches here
    mob_kills, mob_attacks = log_killing(past_state, action)
    time = env_state.timestep

    return Tracker(
        block_placements = block_placements,
        block_mining = block_mining,
        player_location = player_location,
        player_movement = player_movement,
        # revealed_blocks = revealed_blocks,
        doings = doings,
        mob_kills = mob_kills,
        mob_attacks = mob_attacks, 
        time = time
    )

mapped_tracking = jax.vmap(log_tracking, in_axes=(0, 0, 0))


#%%
num_steps = 5e3
def generate_trajectory(network_params, rng, num_envs=64, num_steps=num_steps):
    env = CraftaxSymbolicEnv()
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs)
    env_params = env.default_params
    network = ActorCritic(env.action_space(env_params).n, 512)

    class Transition(NamedTuple):
        tracking: jnp.ndarray
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

        tracker = mapped_tracking(past_state, env_state, action)

        transition = Transition(
            tracking = tracker,
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
    return traj_batch.tracking, traj_batch.done
jit_gen_traj = jax.jit(generate_trajectory)

def get_sparse_trajectory(trajectory):
    block_placements = jax.device_get(trajectory.block_placements)
    block_mining = jax.device_get(trajectory.block_mining)
    player_location = jax.device_get(trajectory.player_location)
    player_movement = jax.device_get(trajectory.player_movement)
    doings = jax.device_get(trajectory.doings)
    mob_kills = jax.device_get(trajectory.mob_kills)
    mob_attacks = jax.device_get(trajectory.mob_attacks)
    time = jax.device_get(trajectory.time)

    block_placements = block_placements.reshape(block_placements.shape[0] * block_placements.shape[1], block_placements.shape[2])
    block_mining = block_mining.reshape(block_mining.shape[0] * block_mining.shape[1], block_mining.shape[2])
    player_location = player_location.reshape(player_location.shape[0] * player_location.shape[1], player_location.shape[2])
    player_movement = player_movement.reshape(player_movement.shape[0] * player_movement.shape[1], player_movement.shape[2])
    doings = doings.reshape(doings.shape[0] * doings.shape[1], doings.shape[2])
    mob_kills = mob_kills.reshape(mob_kills.shape[0] * mob_kills.shape[1], mob_kills.shape[2])
    mob_attacks = mob_attacks.reshape(mob_attacks.shape[0] * mob_attacks.shape[1], mob_attacks.shape[2])
    time = time.reshape(time.shape[0] * time.shape[1])

    block_placements = sp.sparse.coo_array(block_placements)
    block_mining = sp.sparse.coo_array(block_mining)
    player_location = sp.sparse.coo_array(player_location)
    player_movement = sp.sparse.coo_array(player_movement)
    doings = sp.sparse.coo_array(doings)
    mob_kills = sp.sparse.coo_array(mob_kills)
    mob_attacks = sp.sparse.coo_array(mob_attacks)
    time = sp.sparse.coo_array(time)

    return Tracker(
        block_placements = block_placements,
        block_mining = block_mining,
        player_location = player_location,
        player_movement = player_movement,
        doings = doings,
        mob_kills = mob_kills,
        mob_attacks = mob_attacks,
        time = time
    )

#%%
save_dir = "/workspace/CraftaxDevinterp/ExperimentData/trackers"
for modelno in tqdm(range(1000, 1525)):
    os.makedirs(f"{save_dir}/{modelno}", exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{modelno}"
    folder_list = os.listdir(checkpoint_directory)
    network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")

    seed = 0
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    trajectory, done = jit_gen_traj(network_params, _rng)
    trajectory = get_sparse_trajectory(trajectory)
    with open(f"{save_dir}/{modelno}/trajectory.pkl", "wb") as f:
        pickle.dump(trajectory, f)
    with open(f"{save_dir}/{modelno}/done.pkl", "wb") as f:
        pickle.dump(done, f)
#%%
# # %%
# save_dir = "/workspace/CraftaxDevinterp/ExperimentData/trackers"
# for modelno in tqdm(range(1, 1525)):
#     with open(f"{save_dir}/{modelno}/trajectory.pkl", "rb") as f:
#         trajectory = pickle.load(f)
#     block_placements = jax.device_get(trajectory.block_placements)
#     block_mining = jax.device_get(trajectory.block_mining)
#     player_location = jax.device_get(trajectory.player_location)
#     player_movement = jax.device_get(trajectory.player_movement)
#     doings = jax.device_get(trajectory.doings)
#     mob_kills = jax.device_get(trajectory.mob_kills)
#     mob_attacks = jax.device_get(trajectory.mob_attacks)
#     time = jax.device_get(trajectory.time)

#     block_placements = block_placements.reshape(block_placements.shape[0] * block_placements.shape[1], block_placements.shape[2])
#     block_mining = block_mining.reshape(block_mining.shape[0] * block_mining.shape[1], block_mining.shape[2])
#     player_location = player_location.reshape(player_location.shape[0] * player_location.shape[1], player_location.shape[2])
#     player_movement = player_movement.reshape(player_movement.shape[0] * player_movement.shape[1], player_movement.shape[2])
#     doings = doings.reshape(doings.shape[0] * doings.shape[1], doings.shape[2])
#     mob_kills = mob_kills.reshape(mob_kills.shape[0] * mob_kills.shape[1], mob_kills.shape[2])
#     mob_attacks = mob_attacks.reshape(mob_attacks.shape[0] * mob_attacks.shape[1], mob_attacks.shape[2])
#     time = time.reshape(time.shape[0] * time.shape[1])

#     block_placements = sp.sparse.coo_array(block_placements)
#     block_mining = sp.sparse.coo_array(block_mining)
#     player_location = sp.sparse.coo_array(player_location)
#     player_movement = sp.sparse.coo_array(player_movement)
#     doings = sp.sparse.coo_array(doings)
#     mob_kills = sp.sparse.coo_array(mob_kills)
#     mob_attacks = sp.sparse.coo_array(mob_attacks)
#     time = sp.sparse.coo_array(time)
    
#     new_trajectory = Tracker(
#         block_placements = block_placements,
#         block_mining = block_mining,
#         player_location = player_location,
#         player_movement = player_movement,
#         doings = doings,
#         mob_kills = mob_kills,
#         mob_attacks = mob_attacks,
#         time = time
#     )

#     with open(f"{save_dir}/{modelno}/trajectory.pkl", "wb") as f:
#         pickle.dump(new_trajectory, f)
# # %%
