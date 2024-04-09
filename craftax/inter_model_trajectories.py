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
from craftax.craftax.util.game_logic_utils import is_boss_vulnerable
import matplotlib.pyplot as plt
import pickle


#%%
layer_size = 512
seed = 0
num_trajectories = 1525
rng = jax.random.PRNGKey(seed)
rng, _rng = jax.random.split(rng)
num_frames = 100
frames_dir = f"/workspace/CraftaxDevinterp/frames/{seed}"

def generate_trajectory(network_params, rng, num_envs=1, num_steps=num_frames):
    env = CraftaxSymbolicEnv()
    env_params = env.default_params
    env = LogWrapper(env)
    env = OptimisticResetVecEnvWrapper(
        env,
        num_envs=num_envs,
        reset_ratio=min(16, num_envs),
    )
    network = ActorCritic(env.action_space(env_params).n, layer_size)

    class Transition(NamedTuple):
        state: jnp.ndarray
        done: jnp.ndarray

    # COLLECT TRAJECTORIES
    def _env_step(runner_state, unused):
        (
            env_state,
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
            _rng, env_state, action, env_params
        )

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

jit_gen_traj = jax.jit(generate_trajectory)
#%%
def render_craftax_pixels(state, do_night_noise=False):
    block_pixel_size = 7
    textures = TEXTURES[block_pixel_size]
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # RENDER MAP
    # Get view of map
    map = state.map[state.player_level]
    padded_grid = jnp.pad(
        map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = jnp.array([24, 24]) - obs_dim_array // 2 + MAX_OBS_DIM + 2

    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

    # Boss
    boss_block = jax.lax.select(
        is_boss_vulnerable(state),
        BlockType.NECROMANCER_VULNERABLE.value,
        BlockType.NECROMANCER.value,
    )

    map_view_boss = map_view == BlockType.NECROMANCER.value
    map_view = map_view_boss * boss_block + (1 - map_view_boss) * map_view

    # Render map tiles
    map_pixels_indexes = jnp.repeat(
        jnp.repeat(map_view, repeats=block_pixel_size, axis=0),
        repeats=block_pixel_size,
        axis=1,
    )
    map_pixels_indexes = jnp.expand_dims(map_pixels_indexes, axis=-1)
    map_pixels_indexes = jnp.repeat(map_pixels_indexes, repeats=3, axis=2)

    map_pixels = jnp.zeros(
        (OBS_DIM[0] * block_pixel_size, OBS_DIM[1] * block_pixel_size, 3),
        dtype=jnp.float32,
    )

    def _add_block_type_to_pixels(pixels, block_index):
        return (
            pixels
            + textures["full_map_block_textures"][block_index]
            * (map_pixels_indexes == block_index),
            None,
        )

    map_pixels, _ = jax.lax.scan(
        _add_block_type_to_pixels, map_pixels, jnp.arange(len(BlockType))
    )

    # Items
    padded_item_map = jnp.pad(
        state.item_map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=ItemType.NONE.value,
    )

    item_map_view = jax.lax.dynamic_slice(padded_item_map, tl_corner, OBS_DIM)

    # Insert blocked ladders
    is_ladder_down_open = (
        state.monsters_killed[state.player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL
    )
    ladder_down_item = jax.lax.select(
        is_ladder_down_open,
        ItemType.LADDER_DOWN.value,
        ItemType.LADDER_DOWN_BLOCKED.value,
    )

    item_map_view_is_ladder_down = item_map_view == ItemType.LADDER_DOWN.value
    item_map_view = (
        item_map_view_is_ladder_down * ladder_down_item
        + (1 - item_map_view_is_ladder_down) * item_map_view
    )

    map_pixels_item_indexes = jnp.repeat(
        jnp.repeat(item_map_view, repeats=block_pixel_size, axis=0),
        repeats=block_pixel_size,
        axis=1,
    )
    map_pixels_item_indexes = jnp.expand_dims(map_pixels_item_indexes, axis=-1)
    map_pixels_item_indexes = jnp.repeat(map_pixels_item_indexes, repeats=3, axis=2)

    def _add_item_type_to_pixels(pixels, item_index):
        full_map_texture = textures["full_map_item_textures"][item_index]
        mask = map_pixels_item_indexes == item_index

        pixels = pixels * (1 - full_map_texture[:, :, 3] * mask[:, :, 0])[:, :, None]
        pixels = (
            pixels
            + full_map_texture[:, :, :3] * mask * full_map_texture[:, :, 3][:, :, None]
        )

        return pixels, None

    map_pixels, _ = jax.lax.scan(
        _add_item_type_to_pixels, map_pixels, jnp.arange(1, len(ItemType))
    )

    # Render player
    player_texture_index = jax.lax.select(
        state.is_sleeping, 4, state.player_direction - 1
    )
    # map_pixels = (
    #     map_pixels
    #     * (1 - textures["full_map_player_textures_alpha"][player_texture_index])
    #     + textures["full_map_player_textures"][player_texture_index]
    #     * textures["full_map_player_textures_alpha"][player_texture_index]
    # )
    map_pixels = jax.lax.dynamic_update_slice(
        map_pixels, 
        jnp.float32(textures["player_textures"][0][:, :, :3]), 
        (
            jnp.int32(state.player_position[0] * block_pixel_size), 
            jnp.int32(state.player_position[1] * block_pixel_size), 
            jnp.int32(0)
        )
    )

    # Render mobs
    # Zombies

    def _add_mob_to_pixels(carry, mob_index):
        pixels, mobs, texture_name, alpha_texture_name = carry
        # local_position = (
        #     mobs.position[state.player_level, mob_index]
        #     - state.player_position
        #     + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        # )

        local_position = (
            mobs.position[state.player_level, mob_index]
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all()
        on_screen *= mobs.mask[state.player_level, mob_index]

        melee_mob_texture = texture_name[mobs.type_id[state.player_level, mob_index]]
        melee_mob_texture_alpha = alpha_texture_name[
            mobs.type_id[state.player_level, mob_index]
        ]

        melee_mob_texture = melee_mob_texture * on_screen

        melee_mob_texture_with_background = 1 - melee_mob_texture_alpha * on_screen

        melee_mob_texture_with_background = (
            melee_mob_texture_with_background
            * jax.lax.dynamic_slice(
                pixels,
                (
                    local_position[0] * block_pixel_size,
                    local_position[1] * block_pixel_size,
                    0,
                ),
                (block_pixel_size, block_pixel_size, 3),
            )
        )

        melee_mob_texture_with_background = (
            melee_mob_texture_with_background
            + melee_mob_texture * melee_mob_texture_alpha
        )

        pixels = jax.lax.dynamic_update_slice(
            pixels,
            melee_mob_texture_with_background,
            (
                local_position[0] * block_pixel_size,
                local_position[1] * block_pixel_size,
                0,
            ),
        )

        return (pixels, mobs, texture_name, alpha_texture_name), None

    (map_pixels, _, _, _), _ = jax.lax.scan(
        _add_mob_to_pixels,
        (
            map_pixels,
            state.melee_mobs,
            textures["melee_mob_textures"],
            textures["melee_mob_texture_alphas"],
        ),
        jnp.arange(state.melee_mobs.mask.shape[1]),
    )

    (map_pixels, _, _, _), _ = jax.lax.scan(
        _add_mob_to_pixels,
        (
            map_pixels,
            state.passive_mobs,
            textures["passive_mob_textures"],
            textures["passive_mob_texture_alphas"],
        ),
        jnp.arange(state.passive_mobs.mask.shape[1]),
    )

    (map_pixels, _, _, _), _ = jax.lax.scan(
        _add_mob_to_pixels,
        (
            map_pixels,
            state.ranged_mobs,
            textures["ranged_mob_textures"],
            textures["ranged_mob_texture_alphas"],
        ),
        jnp.arange(state.ranged_mobs.mask.shape[1]),
    )

    def _add_projectile_to_pixels(carry, projectile_index):
        pixels, projectiles, projectile_directions = carry
        local_position = (
            projectiles.position[state.player_level, projectile_index]
            - state.player_position
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all()
        on_screen *= projectiles.mask[state.player_level, projectile_index]

        projectile_texture = textures["projectile_textures"][
            projectiles.type_id[state.player_level, projectile_index]
        ]
        projectile_texture_alpha = textures["projectile_texture_alphas"][
            projectiles.type_id[state.player_level, projectile_index]
        ]

        flipped_projectile_texture = jnp.flip(projectile_texture, axis=0)
        flipped_projectile_texture_alpha = jnp.flip(projectile_texture_alpha, axis=0)
        flip_projectile = jnp.logical_or(
            projectile_directions[state.player_level, projectile_index, 0] > 0,
            projectile_directions[state.player_level, projectile_index, 1] > 0,
        )

        projectile_texture = jax.lax.select(
            flip_projectile,
            flipped_projectile_texture,
            projectile_texture,
        )
        projectile_texture_alpha = jax.lax.select(
            flip_projectile,
            flipped_projectile_texture_alpha,
            projectile_texture_alpha,
        )

        transposed_projectile_texture = jnp.transpose(projectile_texture, (1, 0, 2))
        transposed_projectile_texture_alpha = jnp.transpose(
            projectile_texture_alpha, (1, 0, 2)
        )

        projectile_texture = jax.lax.select(
            projectile_directions[state.player_level, projectile_index, 1] != 0,
            transposed_projectile_texture,
            projectile_texture,
        )
        projectile_texture_alpha = jax.lax.select(
            projectile_directions[state.player_level, projectile_index, 1] != 0,
            transposed_projectile_texture_alpha,
            projectile_texture_alpha,
        )

        projectile_texture = projectile_texture * on_screen
        projectile_texture_with_background = 1 - projectile_texture_alpha * on_screen

        projectile_texture_with_background = (
            projectile_texture_with_background
            * jax.lax.dynamic_slice(
                pixels,
                (
                    local_position[0] * block_pixel_size,
                    local_position[1] * block_pixel_size,
                    0,
                ),
                (block_pixel_size, block_pixel_size, 3),
            )
        )

        projectile_texture_with_background = (
            projectile_texture_with_background
            + projectile_texture * projectile_texture_alpha
        )

        pixels = jax.lax.dynamic_update_slice(
            pixels,
            projectile_texture_with_background,
            (
                local_position[0] * block_pixel_size,
                local_position[1] * block_pixel_size,
                0,
            ),
        )

        return (pixels, projectiles, projectile_directions), None

    (map_pixels, _, _), _ = jax.lax.scan(
        _add_projectile_to_pixels,
        (map_pixels, state.mob_projectiles, state.mob_projectile_directions),
        jnp.arange(state.mob_projectiles.mask.shape[1]),
    )

    (map_pixels, _, _), _ = jax.lax.scan(
        _add_projectile_to_pixels,
        (map_pixels, state.player_projectiles, state.player_projectile_directions),
        jnp.arange(state.player_projectiles.mask.shape[1]),
    )

    # Apply darkness (underground)
    light_map = state.light_map[state.player_level]
    padded_light_map = jnp.pad(
        light_map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=False,
    )

    light_map_view = jax.lax.dynamic_slice(padded_light_map, tl_corner, OBS_DIM)
    light_map_pixels = light_map_view.repeat(block_pixel_size, axis=0).repeat(
        block_pixel_size, axis=1
    )

    map_pixels = (light_map_pixels)[:, :, None] * map_pixels

    # Apply night (turned off for inter model traj)
    # night_pixels = textures["night_texture"]
    # daylight = state.light_level
    # daylight = jax.lax.select(state.player_level == 0, daylight, 1.0)

    # if do_night_noise:
    #     night_noise = (
    #         jax.random.uniform(state.state_rng, night_pixels.shape[:2]) * 95 + 32
    #     )
    #     night_noise = jnp.expand_dims(night_noise, axis=-1).repeat(3, axis=-1)

    #     night_intensity = 2 * (0.5 - daylight)
    #     night_intensity = jnp.maximum(night_intensity, 0.0)
    #     night_mask = textures["night_noise_intensity_texture"] * night_intensity
    #     night = (1.0 - night_mask) * map_pixels + night_mask * night_noise

    #     night = night_pixels * 0.5 + 0.5 * night
    #     map_pixels = daylight * map_pixels + (1 - daylight) * night
    # else:
    #     night_noise = jnp.ones(night_pixels.shape[:2]) * 64
    #     night_noise = jnp.expand_dims(night_noise, axis=-1).repeat(3, axis=-1)

    #     night_intensity = 2 * (0.5 - daylight)
    #     night_intensity = jnp.maximum(night_intensity, 0.0)
    #     night_mask = (
    #         jnp.ones_like(textures["night_noise_intensity_texture"])
    #         * night_intensity
    #         * 0.5
    #     )
    #     night = (1.0 - night_mask) * map_pixels + night_mask * night_noise

    #     night = night_pixels * 0.5 + 0.5 * night
    #     map_pixels = daylight * map_pixels + (1 - daylight) * night
    #     # map_pixels = daylight * map_pixels
    #     # night_noise = jnp.ones(night_pixels.shape[:2]) * 64

    # Apply sleep (turned off for inter model traj)
    # sleep_pixels = jnp.zeros_like(map_pixels)
    # sleep_level = 1.0 - state.is_sleeping * 0.5
    # map_pixels = sleep_level * map_pixels + (1 - sleep_level) * sleep_pixels

    # Render mob map
    # mob_map_pixels = (
    #     jnp.array([[[128, 0, 0]]]).repeat(OBS_DIM[0], axis=0).repeat(OBS_DIM[1], axis=1)
    # )
    # padded_mob_map = jnp.pad(
    #     state.mob_map[state.player_level], MAX_OBS_DIM + 2, constant_values=False
    # )
    # mob_map_view = jax.lax.dynamic_slice(padded_mob_map, tl_corner, OBS_DIM)
    # mob_map_pixels = mob_map_pixels * jnp.expand_dims(mob_map_view, axis=-1)
    # mob_map_pixels = mob_map_pixels.repeat(block_pixel_size, axis=0).repeat(
    #     block_pixel_size, axis=1
    # )
    # map_pixels = map_pixels + mob_map_pixels

    # RENDER INVENTORY
    inv_pixel_left_space = (block_pixel_size - int(0.8 * block_pixel_size)) // 2
    inv_pixel_right_space = (
        block_pixel_size - int(0.8 * block_pixel_size) - inv_pixel_left_space
    )

    inv_pixels = jnp.zeros(
        (INVENTORY_OBS_HEIGHT * block_pixel_size, OBS_DIM[1] * block_pixel_size, 3),
        dtype=jnp.float32,
    )

    number_size = int(block_pixel_size * 0.4)
    number_offset = block_pixel_size - number_size
    number_double_offset = block_pixel_size - 2 * number_size

    def _render_digit(pixels, number, x, y):
        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size + number_offset : (x + 1) * block_pixel_size,
        ].mul(1 - textures["number_textures_alpha"][number])

        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size + number_offset : (x + 1) * block_pixel_size,
        ].add(textures["number_textures"][number])

        return pixels

    def _render_two_digit_number(pixels, number, x, y):
        tens = number // 10
        ones = number % 10

        ones_textures = jax.lax.select(
            number == 0,
            textures["number_textures"],
            textures["number_textures_with_zero"],
        )

        ones_textures_alpha = jax.lax.select(
            number == 0,
            textures["number_textures_alpha"],
            textures["number_textures_alpha_with_zero"],
        )

        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size + number_offset : (x + 1) * block_pixel_size,
        ].mul(1 - ones_textures_alpha[ones])

        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size + number_offset : (x + 1) * block_pixel_size,
        ].add(ones_textures[ones])

        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size
            + number_double_offset : x * block_pixel_size
            + number_offset,
        ].mul(1 - textures["number_textures_alpha"][tens])

        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size
            + number_double_offset : x * block_pixel_size
            + number_offset,
        ].add(textures["number_textures"][tens])

        return pixels

    def _render_icon(pixels, texture, x, y):
        return pixels.at[
            block_pixel_size * y
            + inv_pixel_left_space : block_pixel_size * (y + 1)
            - inv_pixel_right_space,
            block_pixel_size * x
            + inv_pixel_left_space : block_pixel_size * (x + 1)
            - inv_pixel_right_space,
        ].set(texture)

    def _render_icon_with_alpha(pixels, texture, x, y):
        existing_slice = pixels[
            block_pixel_size * y
            + inv_pixel_left_space : block_pixel_size * (y + 1)
            - inv_pixel_right_space,
            block_pixel_size * x
            + inv_pixel_left_space : block_pixel_size * (x + 1)
            - inv_pixel_right_space,
        ]

        new_slice = (
            existing_slice * (1 - texture[:, :, 3][:, :, None])
            + texture[:, :, :3] * texture[:, :, 3][:, :, None]
        )

        return pixels.at[
            block_pixel_size * y
            + inv_pixel_left_space : block_pixel_size * (y + 1)
            - inv_pixel_right_space,
            block_pixel_size * x
            + inv_pixel_left_space : block_pixel_size * (x + 1)
            - inv_pixel_right_space,
        ].set(new_slice)

    # Render player stats
    player_health = jnp.maximum(jnp.floor(state.player_health), 1).astype(int)
    health_texture = jax.lax.select(
        player_health > 0,
        textures["health_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, health_texture, 0, 0)
    inv_pixels = _render_two_digit_number(inv_pixels, player_health, 0, 0)

    hunger_texture = jax.lax.select(
        state.player_food > 0,
        textures["hunger_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, hunger_texture, 1, 0)
    inv_pixels = _render_two_digit_number(inv_pixels, state.player_food, 1, 0)

    thirst_texture = jax.lax.select(
        state.player_drink > 0,
        textures["thirst_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, thirst_texture, 2, 0)
    inv_pixels = _render_two_digit_number(inv_pixels, state.player_drink, 2, 0)

    energy_texture = jax.lax.select(
        state.player_energy > 0,
        textures["energy_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, energy_texture, 3, 0)
    inv_pixels = _render_two_digit_number(inv_pixels, state.player_energy, 3, 0)

    mana_texture = jax.lax.select(
        state.player_mana > 0,
        textures["mana_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, mana_texture, 4, 0)
    inv_pixels = _render_two_digit_number(inv_pixels, state.player_mana, 4, 0)

    # Render inventory

    inv_wood_texture = jax.lax.select(
        state.inventory.wood > 0,
        textures["smaller_block_textures"][BlockType.WOOD.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, inv_wood_texture, 0, 2)
    inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.wood, 0, 2)

    inv_stone_texture = jax.lax.select(
        state.inventory.stone > 0,
        textures["smaller_block_textures"][BlockType.STONE.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, inv_stone_texture, 1, 2)
    inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.stone, 1, 2)

    inv_coal_texture = jax.lax.select(
        state.inventory.coal > 0,
        textures["smaller_block_textures"][BlockType.COAL.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, inv_coal_texture, 0, 1)
    inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.coal, 0, 1)

    inv_iron_texture = jax.lax.select(
        state.inventory.iron > 0,
        textures["smaller_block_textures"][BlockType.IRON.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, inv_iron_texture, 1, 1)
    inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.iron, 1, 1)

    inv_diamond_texture = jax.lax.select(
        state.inventory.diamond > 0,
        textures["smaller_block_textures"][BlockType.DIAMOND.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, inv_diamond_texture, 2, 1)
    inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.diamond, 2, 1)

    inv_sapphire_texture = jax.lax.select(
        state.inventory.sapphire > 0,
        textures["smaller_block_textures"][BlockType.SAPPHIRE.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, inv_sapphire_texture, 3, 1)
    inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.sapphire, 3, 1)

    inv_ruby_texture = jax.lax.select(
        state.inventory.ruby > 0,
        textures["smaller_block_textures"][BlockType.RUBY.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, inv_ruby_texture, 4, 1)
    inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.ruby, 4, 1)

    inv_sapling_texture = jax.lax.select(
        state.inventory.sapling > 0,
        textures["sapling_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, inv_sapling_texture, 5, 1)
    inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.sapling, 5, 1)

    # Render tools
    # Pickaxe
    pickaxe_texture = textures["pickaxe_textures"][state.inventory.pickaxe]
    inv_pixels = _render_icon(inv_pixels, pickaxe_texture, 8, 2)

    # Sword
    sword_texture = textures["sword_textures"][state.inventory.sword]
    inv_pixels = _render_icon(inv_pixels, sword_texture, 8, 1)

    # Bow and arrows
    bow_texture = textures["bow_textures"][state.inventory.bow]
    inv_pixels = _render_icon(inv_pixels, bow_texture, 6, 1)

    arrow_texture = jax.lax.select(
        state.inventory.arrows > 0,
        textures["player_projectile_textures"][0],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, arrow_texture, 6, 2)
    inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.arrows, 6, 2)

    # Armour
    for i in range(4):
        armour_texture = textures["armour_textures"][state.inventory.armour[i], i]
        inv_pixels = _render_icon(inv_pixels, armour_texture, 7, i)

    # Torch
    torch_texture = jax.lax.select(
        state.inventory.torches > 0,
        textures["torch_inv_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, torch_texture, 2, 2)
    inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.torches, 2, 2)

    # Potions
    potion_names = ["red", "green", "blue", "pink", "cyan", "yellow"]
    for potion_index, potion_name in enumerate(potion_names):
        potion_texture = jax.lax.select(
            state.inventory.potions[potion_index] > 0,
            textures["potion_textures"][potion_index],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, potion_texture, potion_index, 3)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.potions[potion_index], potion_index, 3
        )

    # Books
    book_texture = jax.lax.select(
        state.inventory.books > 0,
        textures["book_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, book_texture, 3, 2)
    inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.books, 3, 2)

    # Learned spells
    fireball_texture = jax.lax.select(
        state.learned_spells[0],
        textures["fireball_inv_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, fireball_texture, 4, 2)

    iceball_texture = jax.lax.select(
        state.learned_spells[1],
        textures["iceball_inv_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, iceball_texture, 5, 2)

    # Enchantments
    sword_enchantment_texture = textures["sword_enchantment_textures"][
        state.sword_enchantment
    ]
    inv_pixels = _render_icon_with_alpha(inv_pixels, sword_enchantment_texture, 8, 1)

    arrow_enchantment_level = state.bow_enchantment * (state.inventory.arrows > 0)
    arrow_enchantment_texture = textures["arrow_enchantment_textures"][
        arrow_enchantment_level
    ]
    inv_pixels = _render_icon_with_alpha(inv_pixels, arrow_enchantment_texture, 6, 2)

    for i in range(4):
        armour_enchantment_texture = textures["armour_enchantment_textures"][
            state.armour_enchantments[i], i
        ]
        inv_pixels = _render_icon_with_alpha(
            inv_pixels, armour_enchantment_texture, 7, i
        )

    # Dungeon level
    inv_pixels = _render_digit(inv_pixels, state.player_level, 6, 0)

    # Attributes
    xp_texture = jax.lax.select(
        state.player_xp > 0, textures["xp_texture"], textures["smaller_empty_texture"]
    )
    inv_pixels = _render_icon(inv_pixels, xp_texture, 9, 0)
    inv_pixels = _render_digit(inv_pixels, state.player_xp, 9, 0)

    inv_pixels = _render_icon(inv_pixels, textures["dex_texture"], 9, 1)
    inv_pixels = _render_digit(inv_pixels, state.player_dexterity, 9, 1)

    inv_pixels = _render_icon(inv_pixels, textures["str_texture"], 9, 2)
    inv_pixels = _render_digit(inv_pixels, state.player_strength, 9, 2)

    inv_pixels = _render_icon(inv_pixels, textures["int_texture"], 9, 3)
    inv_pixels = _render_digit(inv_pixels, state.player_intelligence, 9, 3)

    # Combine map and inventory
    pixels = jnp.concatenate([map_pixels, inv_pixels], axis=0)

    # # Downscale by 2
    # pixels = pixels[::downscale, ::downscale]

    return pixels

_jitted_render_pixels = jax.jit(render_craftax_pixels)

#%%
for trajectory_no in tqdm(range(0, num_trajectories, 20), desc="Checkpoint progress"):
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{trajectory_no}"
    folder_list = os.listdir(checkpoint_directory)
    network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")

    trajectory, done = jit_gen_traj(network_params, _rng)
    os.makedirs(f"/workspace/CraftaxDevinterp/frames/pixels/trajectory_{trajectory_no}", exist_ok=True)
    os.makedirs(f"/workspace/CraftaxDevinterp/frames/dones/trajectory_{trajectory_no}", exist_ok=True)
    with open(f"/workspace/CraftaxDevinterp/frames/dones/trajectory_{trajectory_no}.pkl", "wb") as f:
        pickle.dump(done, f)

    for frame in tqdm(range(num_frames), desc="Frame progress"):
        state = jax.tree_util.tree_map(lambda x: x[frame, 0, ...], trajectory.env_state)
        pixels = _jitted_render_pixels(state)/256
        with open(f"/workspace/CraftaxDevinterp/frames/pixels/trajectory_{trajectory_no}/frame_{frame}.pkl", "wb") as f:
            pickle.dump(pixels, f)

# TODO: Figure out mobs in in ocean
# TODO: Use initial map as "mean"
# TODO: Smooth trainshift redshift

def rgb_to_yiq(rgb):
    """
    Convert an RGB image to the YIQ color space.
    Assumes the input RGB image has shape (n, m, 3).
    """
    matrix = np.array([
        [0.299, 0.587, 0.114],
        [0.59590059, -0.27455667, -0.32134392],
        [0.21153661, -0.52273617, 0.31119955]
    ])
    # Apply the transformation matrix to the last dimension (color channels)
    return np.tensordot(rgb, matrix, axes=([-1], [1]))

def yiq_to_rgb(yiq):
    """
    Convert a YIQ image back to the RGB color space.
    Assumes the input YIQ image has shape (n, m, 3).
    """
    matrix = np.array([
        [1.0, 0.956, 0.621],
        [1.0, -0.272, -0.647],
        [1.0, -1.106, 1.703]
    ])
    # Apply the inverse transformation matrix to the last dimension
    rgb = np.tensordot(yiq, matrix, axes=([-1], [1]))
    # Ensure RGB values are within the valid range
    return np.clip(rgb, 0, 1)

def redshift_image(rgb_image, shift_intensity):
    """
    Apply a redshift effect to an RGB image while maintaining brightness.
    The input image is assumed to have shape (n, m, 3).
    """
    # Convert RGB to YIQ
    yiq = rgb_to_yiq(rgb_image)
    
    # Increase the I component across the image
    yiq[..., 1] += shift_intensity * (1 - yiq[..., 1])
    
    # Convert back to RGB
    return yiq_to_rgb(yiq)

def shift_scheduler(modelno, total_trajectories):
    return -2/total_trajectories * modelno + 1

def create_composite(frame, num_trajectories = 1525):
    with open(f"{frames_dir}/pixels/trajectory_{0}/frame_{0}.pkl", "rb") as f:
        dummy_pixels = pickle.load(f)

    full_pixels = list()
    for model in range(0, num_trajectories, 80):
        with open(f"{frames_dir}/pixels/trajectory_{model}/frame_{frame}.pkl", "rb") as f:
            model_pixels = pickle.load(f)
        full_pixels.append(model_pixels)
    images = np.array(full_pixels)

    average_image = np.mean(images, axis=0)
    std_image = np.std(images, axis=0)
    # enhanced_std_image = np.clip(std_image * 3, 0, 1)  # Example enhancement
    enhanced_std_image = std_image
    composite_image = average_image * enhanced_std_image + average_image
    color_corrected_composite = (composite_image - np.min(composite_image))/(np.max(composite_image) - np.min(composite_image))
    return color_corrected_composite

def create_composite_range(frame, num_trajectories):
    with open(f"{frames_dir}/pixels/trajectory_{0}/frame_{0}.pkl", "rb") as f:
        dummy_pixels = pickle.load(f)

    full_pixels = list()
    for model in range(0, num_trajectories, 80):
        with open(f"{frames_dir}/pixels/trajectory_{model}/frame_{frame}.pkl", "rb") as f:
            model_pixels = pickle.load(f)
        full_pixels.append(model_pixels)
    images = np.array(full_pixels)
    min_image = np.min(images, axis=0)
    max_image = np.max(images, axis=0)

    # Step 2: Compute the range (max - min) for each pixel
    range_image = max_image - min_image

    # Step 3: Normalize the range image
    normalized_range_image = (range_image - np.min(range_image)) / (np.max(range_image) - np.min(range_image))

    # Step 4: Combine the normalized range image with the average image
    # First, normalize the average image
    average_image = np.mean(images, axis=0)
    normalized_avg_image = (average_image - np.min(average_image)) / (np.max(average_image) - np.min(average_image))

    # Create the composite image
    # For illustration, we simply add the normalized range to the average image
    # You can experiment with different methods of combining these two
    composite_image = normalized_avg_image * normalized_range_image + normalized_avg_image
    composite_image = np.clip(composite_image, 0, 1)
    return composite_image

def create_composite_furthest_mean(frame, num_trajectories):
    with open(f"{frames_dir}/pixels/trajectory_{0}/frame_{0}.pkl", "rb") as f:
        dummy_pixels = pickle.load(f)

    full_pixels = list()
    for model in range(0, num_trajectories, 80):
        with open(f"{frames_dir}/pixels/trajectory_{model}/frame_{frame}.pkl", "rb") as f:
            model_pixels = pickle.load(f)
        full_pixels.append(model_pixels)
    images_stack = np.array(full_pixels)
    mean_image = np.mean(images_stack, axis=0)

    # Initialize an array to hold the output image
    output_image = np.zeros_like(mean_image)

    # Iterate over each pixel
    for i in range(images_stack.shape[1]):  # height
        for j in range(images_stack.shape[2]):  # width
            # Calculate the Euclidean distance from the mean for each pixel in the stack
            distances = np.linalg.norm(images_stack[:, i, j, :] - mean_image[i, j, :], axis=1)
            # Find the index of the image with the maximum distance
            max_distance_idx = np.argmax(distances)
            # Assign the pixel with the maximum distance to the output image
            output_image[i, j, :] = images_stack[max_distance_idx, i, j, :]
    return output_image

def get_image_stack(frame, num_trajectories):
    full_pixels = list()
    for model in range(0, num_trajectories, 80):
        with open(f"{frames_dir}/pixels/trajectory_{model}/frame_{frame}.pkl", "rb") as f:
            model_pixels = pickle.load(f)
        with open(f"{frames_dir}/dones/trajectory_{model}.pkl", "rb") as f:
            done = pickle.load(f)
        if True not in done[:frame+1, 0]:
            full_pixels.append(model_pixels)
    images_stack = np.array(full_pixels)
    return images_stack

def create_composite_furthest_mean_shifted(images_stack, default_image):
    default_image = default_image[0, ...]

    # Initialize an array to hold the output image
    output_image_shifted = np.zeros_like(default_image)

    # Iterate over each pixel
    for i in range(images_stack.shape[1]):  # height
        for j in range(images_stack.shape[2]):  # width
            # Calculate the Euclidean distance from the mean for each pixel in the stack
            distances = np.linalg.norm(images_stack[:, i, j, :] - default_image[i, j, :], axis=1)
            # Find the index of the image with the maximum distance
            max_distance_idx = np.argmax(distances)

            shift_intensity = distances[max_distance_idx]/2


            # Assign the pixel with the maximum distance to the output image
            if max_distance_idx < len(images_stack) / 2:
                # For earlier images, apply redshift
                output_image_shifted[i, j, :] = redshift_image(images_stack[max_distance_idx, i, j, :].reshape(1, 1, 3), shift_intensity).reshape(3,)
            else:
                # For later images, apply blueshift (negative redshift)
                output_image_shifted[i, j, :] = redshift_image(images_stack[max_distance_idx, i, j, :].reshape(1, 1, 3), -shift_intensity).reshape(3,)
    return output_image_shifted

# import cProfile

# cProfile.run("create_composite_furthest_mean_shifted(images_stack)",sort=1)
images_stack = get_image_stack(199, num_trajectories=num_trajectories)
default_image = get_image_stack(0, num_trajectories=0)
composite_image = create_composite_furthest_mean_shifted(images_stack, default_image)
plt.imshow(composite_image)
plt.show()
# #%%

# num_trajectories = 1525
# num_frames = 200

#%%

for trajectory_no in tqdm(range(0, num_trajectories, 20), desc="Checkpoint progress"):
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{trajectory_no}"
    folder_list = os.listdir(checkpoint_directory)
    network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")

    trajectory, done = jit_gen_traj(network_params, _rng)
    os.makedirs(f"{frames_dir}/pixels/trajectory_{trajectory_no}", exist_ok=True)
    os.makedirs(f"{frames_dir}/dones/trajectory_{trajectory_no}", exist_ok=True)
    with open(f"{frames_dir}/dones/trajectory_{trajectory_no}.pkl", "wb") as f:
        pickle.dump(done, f)

    for frame in tqdm(range(num_frames), desc="Frame progress"):
        state = jax.tree_util.tree_map(lambda x: x[frame, 0, ...], trajectory.env_state)
        pixels = _jitted_render_pixels(state)/256
        with open(f"{frames_dir}/pixels/trajectory_{trajectory_no}/frame_{frame}.pkl", "wb") as f:
            pickle.dump(pixels, f)


os.makedirs(f"{frames_dir}/composite", exist_ok=True)
for frame in tqdm(range(num_frames)):
    images_stack = get_image_stack(frame, num_trajectories=num_trajectories)
    composite_image = create_composite_furthest_mean_shifted(images_stack)
    plt.imshow(composite_image)
    plt.savefig(f"{frames_dir}/composite/{frame}.png")
    plt.close()

#%%

# model = 0
# frame=100
# with open(f"/workspace/CraftaxDevinterp/frames/pixels/trajectory_{model}/frame_{frame}.pkl", "rb") as f:
#     model_pixels = pickle.load(f)
# color_corrected = redshift_image(model_pixels, shift_intensity=shift_scheduler(model, 1520))
# plt.imshow(color_corrected)


# #%%
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# #%%
# num_trajectories = 15
# num_frames = 200
# for frame in range(num_frames):
#     with open(f"/workspace/CraftaxDevinterp/frames/pixels/trajectory_{0}/frame_{0}.pkl", "rb") as f:
#         dummy_pixels = pickle.load(f)
#     full_pixels = np.zeros_like(dummy_pixels)
#     for model in range(0, num_trajectories, 20):
#         with open(f"/workspace/CraftaxDevinterp/frames/pixels/trajectory_{model}/frame_{frame}.pkl", "rb") as f:
#             model_pixels = pickle.load(f)
#         color_corrected = redshift_image(model_pixels, shift_intensity=shift_scheduler(model, num_trajectories))
#         full_pixels = model_pixels + full_pixels
#     plt.imshow(full_pixels)
#     plt.show()
#     break
        
# # %%
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# frame=100
# num_trajectories = 1525
# num_frames = 200
# with open(f"/workspace/CraftaxDevinterp/frames/pixels/trajectory_{0}/frame_{0}.pkl", "rb") as f:
#     dummy_pixels = pickle.load(f)
# full_pixels = np.zeros_like(dummy_pixels)
# for model in range(0, num_trajectories, 80):
#     with open(f"/workspace/CraftaxDevinterp/frames/pixels/trajectory_{model}/frame_{frame}.pkl", "rb") as f:
#         model_pixels = pickle.load(f)
#     #color_corrected = redshift_image(model_pixels, shift_intensity=shift_scheduler(model, num_trajectories))
#     full_pixels = full_pixels + model_pixels
# average = full_pixels / (num_trajectories//80 + 1)

# full_pixels = np.zeros_like(average)
# for model in range(0, num_trajectories, 80):
#     with open(f"/workspace/CraftaxDevinterp/frames/pixels/trajectory_{model}/frame_{frame}.pkl", "rb") as f:
#         model_pixels = pickle.load(f)
#     #color_corrected = redshift_image(model_pixels, shift_intensity=shift_scheduler(model, num_trajectories))
#     model_pixels = model_pixels - average
#     full_pixels += model_pixels*10

# plt.imshow( full_pixels )
# plt.show()


# # %%
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# frame=100
# num_trajectories = 1525
# def create_composite(frame, num_trajectories = 1525):
#     with open(f"/workspace/CraftaxDevinterp/frames/pixels/trajectory_{0}/frame_{0}.pkl", "rb") as f:
#         dummy_pixels = pickle.load(f)

#     full_pixels = list()
#     for model in range(0, num_trajectories, 80):
#         with open(f"/workspace/CraftaxDevinterp/frames/pixels/trajectory_{model}/frame_{frame}.pkl", "rb") as f:
#             model_pixels = pickle.load(f)
#         full_pixels.append(model_pixels)
#     images = np.array(full_pixels)

#     average_image = np.mean(images, axis=0)
#     std_image = np.std(images, axis=0)
#     enhanced_std_image = np.clip(std_image * 3, 0, 1)  # Example enhancement
#     composite_image = np.clip(average_image + enhanced_std_image, 0, 1)
#     return composite_image




# # Step 2: Variance highlighting
# # Using standard deviation as a proxy for variance here for visualization purposes

# # Enhance the standard deviation image for better visibility
# # This step is adjustable based on how much emphasis you want on the differences
# enhanced_std_image = np.clip(std_image * 3, 0, 1)  # Example enhancement

# # Step 3: Overlaying variance
# # Combine the average image and the enhanced std image
# # This can be a simple addition, or you might want to use a more complex method to merge them
# composite_image = np.clip(average_image + enhanced_std_image, 0, 1)

# # Display the results
# plt.imshow(composite_image)
# plt.show()