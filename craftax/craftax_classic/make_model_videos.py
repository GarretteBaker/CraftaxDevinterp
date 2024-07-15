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
import orbax.checkpoint as ocp
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
import pickle
import os
from tqdm import tqdm
from craftax.craftax_classic.envs.craftax_state import EnvState, Mobs, Inventory
from craftax.craftax_classic import renderer
import matplotlib.pyplot as plt
import pickle
import scipy as sp


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


#%%
num_envs = 8
num_steps = 1e3
def generate_trajectory(network_params, rng, num_envs=num_envs, num_steps=num_steps):
    env = CraftaxClassicSymbolicEnv()
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs)
    env_params = env.default_params
    network = ActorCritic(env.action_space(env_params).n, 512, activation='relu')

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

        tracker = past_state

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

#%%
def unbatch_mobs(mobs: Mobs, index1: int, index2: int) -> Mobs:
    """
    Helper function to unbatch Mobs data structure.
    
    Args:
    mobs (Mobs): The batched mobs structure.
    index1 (int): The first index in the batch.
    index2 (int): The second index in the batch.
    
    Returns:
    Mobs: The unbatched mobs data for the given indices.
    """
    return Mobs(
        position=mobs.position[index1, index2, ...],
        health=mobs.health[index1, index2],
        mask=mobs.mask[index1, index2],
        attack_cooldown=mobs.attack_cooldown[index1, index2]
    )

def get_unbatched_env_state(env_state: EnvState, index1: int, index2: int) -> EnvState:
    """
    Extracts an unbatched environment state from a batched environment state.
    
    Args:
    env_state (EnvState): The batched environment state.
    index1 (int): The first index in the batch.
    index2 (int): The second index in the batch.
    
    Returns:
    EnvState: The unbatched environment state for the given indices.
    """
    return EnvState(
        map=env_state.map[index1, index2, ...],
        mob_map=env_state.mob_map[index1, index2, ...],
        player_position=env_state.player_position[index1, index2, ...],
        player_direction=env_state.player_direction[index1, index2],
        player_health=env_state.player_health[index1, index2],
        player_food=env_state.player_food[index1, index2],
        player_drink=env_state.player_drink[index1, index2],
        player_energy=env_state.player_energy[index1, index2],
        is_sleeping=env_state.is_sleeping[index1, index2],
        player_recover=env_state.player_recover[index1, index2],
        player_hunger=env_state.player_hunger[index1, index2],
        player_thirst=env_state.player_thirst[index1, index2],
        player_fatigue=env_state.player_fatigue[index1, index2],
        inventory=Inventory(
            wood=env_state.inventory.wood[index1, index2],
            stone=env_state.inventory.stone[index1, index2],
            coal=env_state.inventory.coal[index1, index2],
            iron=env_state.inventory.iron[index1, index2],
            diamond=env_state.inventory.diamond[index1, index2],
            sapling=env_state.inventory.sapling[index1, index2],
            wood_pickaxe=env_state.inventory.wood_pickaxe[index1, index2],
            stone_pickaxe=env_state.inventory.stone_pickaxe[index1, index2],
            iron_pickaxe=env_state.inventory.iron_pickaxe[index1, index2],
            wood_sword=env_state.inventory.wood_sword[index1, index2],
            stone_sword=env_state.inventory.stone_sword[index1, index2],
            iron_sword=env_state.inventory.iron_sword[index1, index2]
        ),
        zombies=unbatch_mobs(env_state.zombies, index1, index2),
        cows=unbatch_mobs(env_state.cows, index1, index2),
        skeletons=unbatch_mobs(env_state.skeletons, index1, index2),
        arrows=unbatch_mobs(env_state.arrows, index1, index2),
        arrow_directions=env_state.arrow_directions[index1, index2, ...],
        growing_plants_positions=env_state.growing_plants_positions[index1, index2, ...],
        growing_plants_age=env_state.growing_plants_age[index1, index2, ...],
        growing_plants_mask=env_state.growing_plants_mask[index1, index2, ...],
        light_level=env_state.light_level[index1, index2],
        achievements=env_state.achievements[index1, index2, ...],
        state_rng=env_state.state_rng[index1, index2],
        timestep=env_state.timestep[index1, index2],
        fractal_noise_angles=env_state.fractal_noise_angles
    )


jit_ren = jax.jit(lambda unbatched: renderer.render_craftax_pixels(unbatched, 64))
# %%
checkpoint_directory = "/workspace/CraftaxDevinterp/intermediate"
checkpoint_list = os.listdir(checkpoint_directory)
num_models = len(checkpoint_list)

# selected_models = [0, 200, 822, 1100, 1600]
# selected_models = [200, 822, 1100, 1524]
selected_models = [1524]
for modelno in tqdm(selected_models, desc="Model"):
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{checkpoint_list[modelno]}"
    folder_list = os.listdir(checkpoint_directory)
    network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
    seed = 0
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    trajectory, done = jit_gen_traj(network_params, _rng)

    frameno = 0
    savedir = f"/workspace/CraftaxDevinterp/ExperimentData/videos/frames/{modelno}"
    os.makedirs(savedir, exist_ok=True)
    for batch in tqdm(range(num_envs), desc="batch"):
        for t in tqdm(range(int(num_steps)), desc="time"):
            obs = jit_ren(get_unbatched_env_state(trajectory, t, batch))
            plt.imshow(obs/256)
            plt.savefig(f"{savedir}/{frameno}.png")
            plt.close()
            frameno += 1
