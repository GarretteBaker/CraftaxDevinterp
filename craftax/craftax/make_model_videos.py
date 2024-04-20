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
def unbatch_env_state(env_state_batch: EnvState, batch_idx1: int, batch_idx2: int) -> EnvState:
    # This function slices a single environment state from a batched environment state.
    return EnvState(
        map=env_state_batch.map[batch_idx1, batch_idx2, ...],
        item_map=env_state_batch.item_map[batch_idx1, batch_idx2, ...],
        mob_map=env_state_batch.mob_map[batch_idx1, batch_idx2, ...],
        light_map=env_state_batch.light_map[batch_idx1, batch_idx2, ...],
        down_ladders=env_state_batch.down_ladders[batch_idx1, batch_idx2, ...],
        up_ladders=env_state_batch.up_ladders[batch_idx1, batch_idx2, ...],
        chests_opened=env_state_batch.chests_opened[batch_idx1, batch_idx2, ...],
        monsters_killed=env_state_batch.monsters_killed[batch_idx1, batch_idx2, ...],
        player_position=env_state_batch.player_position[batch_idx1, batch_idx2, ...],
        player_level=env_state_batch.player_level[batch_idx1, batch_idx2],
        player_direction=env_state_batch.player_direction[batch_idx1, batch_idx2],
        player_health=env_state_batch.player_health[batch_idx1, batch_idx2],
        player_food=env_state_batch.player_food[batch_idx1, batch_idx2],
        player_drink=env_state_batch.player_drink[batch_idx1, batch_idx2],
        player_energy=env_state_batch.player_energy[batch_idx1, batch_idx2],
        player_mana=env_state_batch.player_mana[batch_idx1, batch_idx2],
        is_sleeping=env_state_batch.is_sleeping[batch_idx1, batch_idx2],
        is_resting=env_state_batch.is_resting[batch_idx1, batch_idx2],
        player_recover=env_state_batch.player_recover[batch_idx1, batch_idx2],
        player_hunger=env_state_batch.player_hunger[batch_idx1, batch_idx2],
        player_thirst=env_state_batch.player_thirst[batch_idx1, batch_idx2],
        player_fatigue=env_state_batch.player_fatigue[batch_idx1, batch_idx2],
        player_recover_mana=env_state_batch.player_recover_mana[batch_idx1, batch_idx2],
        player_xp=env_state_batch.player_xp[batch_idx1, batch_idx2],
        player_dexterity=env_state_batch.player_dexterity[batch_idx1, batch_idx2],
        player_strength=env_state_batch.player_strength[batch_idx1, batch_idx2],
        player_intelligence=env_state_batch.player_intelligence[batch_idx1, batch_idx2],
        inventory=Inventory(
            wood=env_state_batch.inventory.wood[batch_idx1, batch_idx2],
            stone=env_state_batch.inventory.stone[batch_idx1, batch_idx2],
            coal=env_state_batch.inventory.coal[batch_idx1, batch_idx2],
            iron=env_state_batch.inventory.iron[batch_idx1, batch_idx2],
            diamond=env_state_batch.inventory.diamond[batch_idx1, batch_idx2],
            sapling=env_state_batch.inventory.sapling[batch_idx1, batch_idx2],
            pickaxe=env_state_batch.inventory.pickaxe[batch_idx1, batch_idx2],
            sword=env_state_batch.inventory.sword[batch_idx1, batch_idx2],
            bow=env_state_batch.inventory.bow[batch_idx1, batch_idx2],
            arrows=env_state_batch.inventory.arrows[batch_idx1, batch_idx2],
            armour=env_state_batch.inventory.armour[batch_idx1, batch_idx2, ...],
            torches=env_state_batch.inventory.torches[batch_idx1, batch_idx2],
            ruby=env_state_batch.inventory.ruby[batch_idx1, batch_idx2],
            sapphire=env_state_batch.inventory.sapphire[batch_idx1, batch_idx2],
            potions=env_state_batch.inventory.potions[batch_idx1, batch_idx2, ...],
            books=env_state_batch.inventory.books[batch_idx1, batch_idx2]
        ),
        melee_mobs=Mobs(
            position=env_state_batch.melee_mobs.position[batch_idx1, batch_idx2, ...],
            health=env_state_batch.melee_mobs.health[batch_idx1, batch_idx2, ...],
            mask=env_state_batch.melee_mobs.mask[batch_idx1, batch_idx2, ...],
            attack_cooldown=env_state_batch.melee_mobs.attack_cooldown[batch_idx1, batch_idx2, ...],
            type_id=env_state_batch.melee_mobs.type_id[batch_idx1, batch_idx2, ...]
        ),
        passive_mobs=Mobs(
            position=env_state_batch.passive_mobs.position[batch_idx1, batch_idx2, ...],
            health=env_state_batch.passive_mobs.health[batch_idx1, batch_idx2, ...],
            mask=env_state_batch.passive_mobs.mask[batch_idx1, batch_idx2, ...],
            attack_cooldown=env_state_batch.passive_mobs.attack_cooldown[batch_idx1, batch_idx2, ...],
            type_id=env_state_batch.passive_mobs.type_id[batch_idx1, batch_idx2, ...]
        ),
        ranged_mobs=Mobs(
            position=env_state_batch.ranged_mobs.position[batch_idx1, batch_idx2, ...],
            health=env_state_batch.ranged_mobs.health[batch_idx1, batch_idx2, ...],
            mask=env_state_batch.ranged_mobs.mask[batch_idx1, batch_idx2, ...],
            attack_cooldown=env_state_batch.ranged_mobs.attack_cooldown[batch_idx1, batch_idx2, ...],
            type_id=env_state_batch.ranged_mobs.type_id[batch_idx1, batch_idx2, ...]
        ),
        mob_projectiles=Mobs(
            position=env_state_batch.mob_projectiles.position[batch_idx1, batch_idx2, ...],
            health=env_state_batch.mob_projectiles.health[batch_idx1, batch_idx2, ...],
            mask=env_state_batch.mob_projectiles.mask[batch_idx1, batch_idx2, ...],
            attack_cooldown=env_state_batch.mob_projectiles.attack_cooldown[batch_idx1, batch_idx2, ...],
            type_id=env_state_batch.mob_projectiles.type_id[batch_idx1, batch_idx2, ...]
        ),
        mob_projectile_directions=env_state_batch.mob_projectile_directions[batch_idx1, batch_idx2, ...],
        player_projectiles=Mobs(
            position=env_state_batch.player_projectiles.position[batch_idx1, batch_idx2, ...],
            health=env_state_batch.player_projectiles.health[batch_idx1, batch_idx2, ...],
            mask=env_state_batch.player_projectiles.mask[batch_idx1, batch_idx2, ...],
            attack_cooldown=env_state_batch.player_projectiles.attack_cooldown[batch_idx1, batch_idx2, ...],
            type_id=env_state_batch.player_projectiles.type_id[batch_idx1, batch_idx2, ...]
        ),
        player_projectile_directions=env_state_batch.player_projectile_directions[batch_idx1, batch_idx2, ...],
        growing_plants_positions=env_state_batch.growing_plants_positions[batch_idx1, batch_idx2, ...],
        growing_plants_age=env_state_batch.growing_plants_age[batch_idx1, batch_idx2, ...],
        growing_plants_mask=env_state_batch.growing_plants_mask[batch_idx1, batch_idx2, ...],
        potion_mapping=env_state_batch.potion_mapping[batch_idx1, batch_idx2, ...],
        learned_spells=env_state_batch.learned_spells[batch_idx1, batch_idx2, ...],
        sword_enchantment=env_state_batch.sword_enchantment[batch_idx1, batch_idx2],
        bow_enchantment=env_state_batch.bow_enchantment[batch_idx1, batch_idx2],
        armour_enchantments=env_state_batch.armour_enchantments[batch_idx1, batch_idx2, ...],
        boss_progress=env_state_batch.boss_progress[batch_idx1, batch_idx2],
        boss_timesteps_to_spawn_this_round=env_state_batch.boss_timesteps_to_spawn_this_round[batch_idx1, batch_idx2],
        light_level=env_state_batch.light_level[batch_idx1, batch_idx2],
        achievements=env_state_batch.achievements[batch_idx1, batch_idx2, ...],
        state_rng=env_state_batch.state_rng[batch_idx1, batch_idx2],
        timestep=env_state_batch.timestep[batch_idx1, batch_idx2],
    )

jit_ren = jax.jit(lambda unbatched: renderer.render_craftax_pixels(unbatched, 64))
# %%
# selected_models = [18, 414, 762, 924, 1122, 1512, 1524]
selected_models = [1512, 1524]
for modelno in tqdm(selected_models, desc="Model"):
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{modelno}"
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
            obs = jit_ren(unbatch_env_state(trajectory, t, batch))
            plt.imshow(obs/256)
            plt.savefig(f"{savedir}/{frameno}.png")
            plt.close()
            frameno += 1
