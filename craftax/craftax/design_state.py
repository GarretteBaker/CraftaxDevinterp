"""
Design custom states for craftax environment.

Currently only supports a single example `generate test world`. In the
future, when we have specific ideas for test states we want to try, we can
refactor to one method that generates a default state and then separate
methods that generate specific types of test levels according to whatever
parameters are relevant.
"""

import jax
import jax.numpy as jnp

import craftax.craftax.constants as constants
from craftax.craftax.craftax_state import EnvState, Inventory, Mobs
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax.util.game_logic_utils import calculate_light_level


def generate_test_world(
    rng,
    params : EnvParams = EnvParams(),
    static_params: StaticEnvParams = StaticEnvParams(),
):
    """
    Parameters:

    * rng : PRNGKey
            Random state (consumed)
    * params : EnvParams
            Contains dynamic parameters influencing the environment (such as
            controlling world generation and 
    * static_params : StaticEnvParams
            Contains static parameters influencing the environment (such as
            size of the map, number of enemies)

    Returns:

    * state : EnvState
            A simplified world state with a controlled stimulus.
    """
    rng, rng_layers = jax.random.split(rng)
    rng, rng_potions = jax.random.split(rng)
    rng, rng_state = jax.random.split(rng)

    def generate_empty_map_layer(map_rng):
        # hardcode world structure
        block_map = jnp.zeros(static_params.map_size)
        block_map = block_map.at[:,:].set(
            constants.BlockType.STONE.value
        )
        block_map = block_map.at[21:28,21:28].set(
            constants.BlockType.GRASS.value
        )
        # hardcode full light
        light_map = jnp.ones(
            static_params.map_size,
            dtype=jnp.float32,
        )
        # hardcode ladder positions
        ladder_down_position = jnp.array((0,0))
        ladder_up_position = jnp.array((0,1))
        # hard code empty item map
        item_map = (
            jnp.zeros(static_params.map_size, dtype=jnp.int32)
                .at[
                    ladder_down_position[0],
                    ladder_down_position[1]
                ].set(constants.ItemType.LADDER_UP.value)
                .at[
                    ladder_down_position[0],
                    ladder_down_position[1],
                ].set(constants.ItemType.LADDER_DOWN.value)
        )
        return (
            block_map,
            item_map,
            light_map,
            ladder_down_position,
            ladder_up_position,
        )

    block_map, item_map, light_map, down_ladders, up_ladders = jax.vmap(
        generate_empty_map_layer,
        in_axes=(0,),
        out_axes=(0,0,0,0,0,),
    )(
        jax.random.split(rng_layers, static_params.num_levels),
    )


    def generate_empty_mobs(max_mobs):
        return Mobs(
            position=jnp.zeros(
                (static_params.num_levels, max_mobs, 2),
                dtype=jnp.int32,
            ),
            health=jnp.ones(
                (static_params.num_levels, max_mobs),
                dtype=jnp.float32,
            ),
            mask=jnp.zeros(
                (static_params.num_levels, max_mobs),
                dtype=bool,
            ),
            attack_cooldown=jnp.zeros(
                (static_params.num_levels, max_mobs),
                dtype=jnp.int32,
            ),
            type_id=jnp.zeros(
                (static_params.num_levels, max_mobs),
                dtype=jnp.int32,
            ),
        )

    state = EnvState(
        # world
        map=block_map,
        item_map=item_map,
        light_map=light_map,
        down_ladders=down_ladders,
        up_ladders=up_ladders,

        # player
        player_position=jnp.array((
            static_params.map_size[0] // 2,
            static_params.map_size[1] // 2,
        )),
        player_direction=jnp.asarray(
            constants.Action.UP.value,
            dtype=jnp.int32,
        ),
        player_level=jnp.asarray(0, dtype=jnp.int32),
        player_health=jnp.asarray(9.0, dtype=jnp.float32),
        player_food=jnp.asarray(9, dtype=jnp.int32),
        player_drink=jnp.asarray(9, dtype=jnp.int32),
        player_energy=jnp.asarray(9, dtype=jnp.int32),
        player_mana=jnp.asarray(9, dtype=jnp.int32),
        player_recover=jnp.asarray(0.0, dtype=jnp.float32),
        player_hunger=jnp.asarray(0.0, dtype=jnp.float32),
        player_thirst=jnp.asarray(0.0, dtype=jnp.float32),
        player_fatigue=jnp.asarray(0.0, dtype=jnp.float32),
        player_recover_mana=jnp.asarray(0.0, dtype=jnp.float32),
        is_sleeping=False,
        is_resting=False,
        player_xp=jnp.asarray(0, dtype=jnp.int32),
        player_dexterity=jnp.asarray(1, dtype=jnp.int32),
        player_strength=jnp.asarray(1, dtype=jnp.int32),
        player_intelligence=jnp.asarray(1, dtype=jnp.int32),
        learned_spells=jnp.array([False, False], dtype=bool),
        player_projectiles=generate_empty_mobs(
            static_params.max_player_projectiles,
        ),
        player_projectile_directions=jnp.ones(
            (
                static_params.num_levels,
                static_params.max_player_projectiles,
                2,
            ),
            dtype=jnp.int32,
        ),

        # inventory
        inventory=Inventory(
            wood=jnp.asarray(0, dtype=jnp.int32),
            stone=jnp.asarray(0, dtype=jnp.int32),
            coal=jnp.asarray(0, dtype=jnp.int32),
            iron=jnp.asarray(0, dtype=jnp.int32),
            diamond=jnp.asarray(0, dtype=jnp.int32),
            sapling=jnp.asarray(0, dtype=jnp.int32),
            pickaxe=jnp.asarray(0, dtype=jnp.int32),
            sword=jnp.asarray(0, dtype=jnp.int32),
            bow=jnp.asarray(0, dtype=jnp.int32),
            arrows=jnp.asarray(0, dtype=jnp.int32),
            torches=jnp.asarray(0, dtype=jnp.int32),
            ruby=jnp.asarray(0, dtype=jnp.int32),
            sapphire=jnp.asarray(0, dtype=jnp.int32),
            books=jnp.asarray(0, dtype=jnp.int32),
            potions=jnp.zeros(6, dtype=jnp.int32),
            armour=jnp.zeros(4, dtype=jnp.int32),
        ),
        sword_enchantment=jnp.asarray(0, dtype=jnp.int32),
        bow_enchantment=jnp.asarray(0, dtype=jnp.int32),
        armour_enchantments=jnp.zeros(4, dtype=jnp.int32),
    
        # mobs
        mob_map = jnp.zeros(
            (static_params.num_levels, *static_params.map_size),
            dtype=bool,
        ),
        melee_mobs=generate_empty_mobs(static_params.max_melee_mobs),
        ranged_mobs=generate_empty_mobs(static_params.max_ranged_mobs),
        passive_mobs=generate_empty_mobs(static_params.max_passive_mobs),
        mob_projectiles=generate_empty_mobs(static_params.max_mob_projectiles),
        mob_projectile_directions=jnp.ones(
            (static_params.num_levels, static_params.max_mob_projectiles, 2),
            dtype=jnp.int32,
        ),
        
        # potions
        potion_mapping=jax.random.permutation(rng_potions, 6),
        
        # farming
        growing_plants_positions=jnp.zeros(
            (static_params.max_growing_plants, 2),
            dtype=jnp.int32,
        ),
        growing_plants_age=jnp.zeros(
            static_params.max_growing_plants,
            dtype=jnp.int32,
        ),
        growing_plants_mask=jnp.zeros(
            static_params.max_growing_plants,
            dtype=bool,
        ),

        # progress
        achievements=jnp.zeros((len(constants.Achievement),), dtype=bool),
        chests_opened=jnp.zeros(static_params.num_levels, dtype=bool),
        monsters_killed=jnp.zeros(
            static_params.num_levels,
            dtype=jnp.int32,
        # (start the overworld killcount at 10 to open the first ladder)
        ).at[0].set(10),
        boss_progress=jnp.asarray(0, dtype=jnp.int32),
        boss_timesteps_to_spawn_this_round=jnp.asarray(
            constants.BOSS_FIGHT_SPAWN_TURNS,
            dtype=jnp.int32,
        ),
        light_level=jnp.asarray(
            calculate_light_level(0, params),
            dtype=jnp.float32,
        ),

        # misc
        state_rng=rng_state,
        timestep=jnp.asarray(0, dtype=jnp.int32),
    )

    return state


if __name__ == "__main__":
    print("importing libraries...")
    from craftax.craftax.renderer import render_craftax_pixels
    from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
    import matplotlib.pyplot as plt
    import numpy as np


    print("initialising...")
    env = CraftaxSymbolicEnv()
    
    print("resetting... (generating world)")
    rng = jax.random.PRNGKey(seed=0)
    obs, state = env.reset(rng)

    print("generating custom world state...")
    custom_state = generate_test_world(rng)
    
    print("rendering...")
    rgb = render_craftax_pixels(
        custom_state,
        block_pixel_size=64, # or 16 or 64
        do_night_noise=True,
    )
    plt.imshow(rgb/255)
    plt.savefig("/workspace/CraftaxDevinterp/design_state.png")
