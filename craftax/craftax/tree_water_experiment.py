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
from craftax.craftax.renderer import render_craftax_symbolic
from craftax.models.actor_critic import ActorCritic
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv


def generate_test_world(
    rng,
    num_water, 
    num_wood, 
    pickaxe=False,
    crafting_table=False,
    sword=False,
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

    def generate_tree_water_map_layer(map_rng):
        # hardcode world structure
        block_map = jnp.zeros(static_params.map_size)
        block_map = block_map.at[:,:].set(
            constants.BlockType.STONE.value
        )
        block_map = block_map.at[21:28,21:28].set(
            constants.BlockType.GRASS.value
        )
        block_map = block_map.at[26:28, 21:28].set(
            constants.BlockType.WATER.value
        )
        if crafting_table:
            block_map = block_map.at[24, 25].set(
                constants.BlockType.CRAFTING_TABLE.value
            )

        for i in range(21, 28):
            if i % 2 == 1:
                block_map = block_map.at[21, i].set(
                    constants.BlockType.TREE.value
                )
            else:
                block_map = block_map.at[22, i].set(
                    constants.BlockType.TREE.value
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
        generate_tree_water_map_layer,
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

    num_pickaxe = 0
    num_sword = 0
    if pickaxe: num_pickaxe = 1
    if sword: num_sword = 1
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
        player_drink=jnp.asarray(num_water, dtype=jnp.int32),
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
            wood=jnp.asarray(num_wood, dtype=jnp.int32),
            stone=jnp.asarray(0, dtype=jnp.int32),
            coal=jnp.asarray(0, dtype=jnp.int32),
            iron=jnp.asarray(0, dtype=jnp.int32),
            diamond=jnp.asarray(0, dtype=jnp.int32),
            sapling=jnp.asarray(0, dtype=jnp.int32),
            pickaxe=jnp.asarray(num_pickaxe, dtype=jnp.int32),
            sword=jnp.asarray(num_sword, dtype=jnp.int32),
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

def optimized_experiment(
        num_wood, 
        num_water, 
        network_params,
        crafting_table=False, 
        pickaxe=False, 
        sword=False, 
        torch = False # to implement
):
    network = ActorCritic(43, 512)
    rng = jax.random.PRNGKey(seed=0)
    custom_state = generate_test_world(rng, num_wood=num_wood, num_water=num_water, crafting_table=crafting_table, pickaxe=pickaxe, sword = sword)
    obs = render_craftax_symbolic(custom_state)
    pi, _ = network.apply(network_params, obs)
    probs = pi.probs
    return probs



def run_experiment(env, env_state, models=1525, count_by=1):
    import os
    import orbax.checkpoint as ocp
    from tqdm import tqdm
    env_params = env.default_params
    network = ActorCritic(env.action_space(env_params).n, 512)
    up = list()
    down = list()
    left = list()
    right = list()
    probses = list()
    for model in tqdm(range(0, models, count_by)):
        checkpointer = ocp.StandardCheckpointer()
        checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{model}"
        folder_list = os.listdir(checkpoint_directory)
        network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
        obs = render_craftax_symbolic(env_state)
        pi, _ = network.apply(network_params, obs)
        probs = pi.probs
        left_prob = probs[1]
        right_prob = probs[2]
        up_prob = probs[3]
        down_prob = probs[4]

        up.append(up_prob)
        down.append(down_prob)
        left.append(left_prob)
        right.append(right_prob)
        probses.append(probs)
    return up, down, left, right, probses

def experiment_with_varied_water_and_wood(
        wood_range: tuple, 
        water_range: tuple, 
        models=1525, 
        count_by=1, 
        save_dir = "/workspace/CraftaxDevinterp/ExperimentData/tree_water_experiment/data", 
        plotting = True, 
        crafting_table = False, 
        pickaxe = False, 
        sword = False
    ):
    from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv

    import os
    import pickle
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display
    import numpy as np
    env = CraftaxSymbolicEnv()
    experiment_data = dict()
    rng = jax.random.PRNGKey(seed=0)


    for num_wood in range(*wood_range):
        for num_water in range(*water_range):
            custom_state = generate_test_world(rng, num_wood=num_wood, num_water=num_water, crafting_table=crafting_table, pickaxe=pickaxe, sword = sword)
            up, down, left, right, probs = run_experiment(env, custom_state, models=models, count_by=count_by)

            experiment_data[(num_wood, num_water, "UP")] = up
            experiment_data[(num_wood, num_water, "DOWN")] = down
            experiment_data[(num_wood, num_water, "LEFT")] = left
            experiment_data[(num_wood, num_water, "RIGHT")] = right
            experiment_data[(num_wood, num_water, "PROBS")] = probs
    
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/experiment_data.pkl", "wb") as f:
        pickle.dump(experiment_data, f)
    
    if plotting:
        from ipywidgets import interactive, Output
        from ipywidgets.embed import embed_minimal_html
        wood_slider = widgets.IntSlider(value=wood_range[0], min=wood_range[0], max=wood_range[1]-1, step=1, description='Wood')
        water_slider = widgets.IntSlider(value=water_range[0], min=water_range[0], max=water_range[1]-1, step=1, description='Water')
        plot_output = Output()

        def update_plot(num_wood, num_water):
            with plot_output:
                plot_output.clear_output(wait=True)  # Clear the previous plot
                plt.figure(figsize=(10, 6))
                probs = np.array( experiment_data.get((num_wood, num_water, "PROBS") ))
                plt.plot(probs[:, 0], label="NOOP")
                plt.plot(probs[:, 1], label="LEFT")
                plt.plot(probs[:, 2], label="RIGHT")
                plt.plot(probs[:, 3], label="UP")
                plt.plot(probs[:, 4], label="DOWN")
                plt.plot(probs[:, 5], label="DO")
                plt.plot(probs[:, 6], label="SLEEP")
                plt.plot(probs[:, 7], label="PLACE_STONE")
                plt.plot(probs[:, 8], label="PLACE_TABLE")
                plt.plot(probs[:, 9], label="PLACE_FURNACE")
                plt.plot(probs[:, 10], label="PLACE_PLANT")
                plt.plot(probs[:, 11], label="MAKE_WOOD_PICKAXE")
                plt.plot(probs[:, 12], label="MAKE_STONE_PICKAXE")
                plt.plot(probs[:, 13], label="MAKE_IRON_PICKAXE")
                plt.plot(probs[:, 14], label="MAKE_WOOD_SWORD")
                plt.plot(probs[:, 15], label="MAKE_STONE_SWORD")
                plt.plot(probs[:, 16], label="MAKE_IRON_SWORD")
                plt.plot(probs[:, 17], label="REST")
                # plt.plot(probs[:, 18], label="DESCEND")
                # plt.plot(probs[:, 19], label="ASCEND")
                plt.plot(probs[:, 20], label="MAKE_DIAMOND_PICKAXE")
                plt.plot(probs[:, 21], label="MAKE_DIAMOND_SWORD")
                plt.plot(probs[:, 22], label="MAKE_IRON_ARMOUR")
                plt.plot(probs[:, 23], label="MAKE_DIAMOND_ARMOUR")
                # plt.plot(probs[:, 24], label="SHOOT_ARROW")
                plt.plot(probs[:, 25], label="MAKE_ARROW")
                # plt.plot(probs[:, 26], label="CAST_FIREBALL")
                # plt.plot(probs[:, 27], label="CAST_ICEBALL")
                # plt.plot(probs[:, 28], label="PLACE_TORCH")
                # plt.plot(probs[:, 29], label="DRINK_POTION_RED")
                # plt.plot(probs[:, 30], label="DRINK_POTION_GREEN")
                # plt.plot(probs[:, 31], label="DRINK_POTION_BLUE")
                # plt.plot(probs[:, 32], label="DRINK_POTION_PINK")
                # plt.plot(probs[:, 33], label="DRINK_POTION_CYAN")
                # plt.plot(probs[:, 34], label="DRINK_POTION_YELLOW")
                # plt.plot(probs[:, 35], label="READ_BOOK")
                # plt.plot(probs[:, 36], label="ENCHANT_SWORD")
                # plt.plot(probs[:, 37], label="ENCHANT_ARMOUR")
                plt.plot(probs[:, 38], label="MAKE_TORCH")
                # plt.plot(probs[:, 39], label="LEVEL_UP_DEXTERITY")
                # plt.plot(probs[:, 40], label="LEVEL_UP_STRENGTH")
                # plt.plot(probs[:, 41], label="LEVEL_UP_INTELLIGENCE")
                # plt.plot(probs[:, 42], label="ENCHANT_BOW")
                plt.title(f"Probability Curves for {num_wood} Wood and {num_water} Water")
                plt.xlabel('Steps')
                plt.ylabel('Probability')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.grid(True)
                plt.show()

        wood_water_interactive = interactive(update_plot, num_wood=wood_slider, num_water=water_slider)
        display(widgets.VBox([wood_water_interactive, plot_output]))

        # Save the interactive plot and controls as an HTML file
        filename = f"{save_dir}/interactive_plot.html"
        embed_minimal_html(filename, views=[wood_water_interactive, plot_output], title='Interactive Probability Curves')
        print(f"Interactive plot saved to {filename}")

if __name__ == "__main__":
    import orbax.checkpoint as ocp
    import os
    import time
    from tqdm import tqdm
    import pickle

    env = CraftaxSymbolicEnv()
    env_params = env.default_params
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{0}"
    folder_list = os.listdir(checkpoint_directory)
    network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
    # network_params = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), network_params)
    experiment_mapped = jax.jit(jax.vmap(jax.vmap(lambda wood, water, params: optimized_experiment(wood, water, params), (0, 0, None)), (0, 0, None)))
    wood_range = jnp.arange(0, 5, 1)
    water_range = jnp.arange(0, 5, 1)
    wood, water = jnp.meshgrid(wood_range, water_range)

    t0 = time.time()
    os.makedirs("/workspace/CraftaxDevinterp/ExperimentData/tree_water_table_pickaxe_sword/results", exist_ok=True)
    for modelno in tqdm(range(1525)):
        checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{modelno}"
        folder_list = os.listdir(checkpoint_directory)
        network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
        result = experiment_mapped(wood, water, network_params)
        with open(f"/workspace/CraftaxDevinterp/ExperimentData/tree_water_table_pickaxe_sword/results/{modelno}.pkl", "wb") as f:
            pickle.dump(result, f)
    print(f"Time for mapped: {time.time() - t0}")

    # t0 = time.time()
    # results = list()
    # for wood in wood_range:
    #     result = optimized_experiment(wood, 0, network, network_params)
    #     results.append(result)
    # print(f"Time for unmapped: {time.time() - t0}")


    # experiment_with_varied_water_and_wood(
    #     wood_range = (0, 5), 
    #     water_range = (0, 5), 
    #     count_by=1, 
    #     crafting_table=True, 
    #     pickaxe=True, 
    #     sword = True, 
    #     plotting=False, 
    #     save_dir = "/workspace/CraftaxDevinterp/ExperimentData/tree_water_table_pickaxe_sword/data"
    # )

    # print("importing libraries...")
    # from craftax.craftax.renderer import render_craftax_pixels
    # from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import pickle
    # import os
    # os.makedirs("/workspace/CraftaxDevinterp/ExperimentData/tree_water_experiment/images", exist_ok=True)
    # os.makedirs("/workspace/CraftaxDevinterp/ExperimentData/tree_water_experiment/data", exist_ok=True)

    # print("initialising...")
    # env = CraftaxSymbolicEnv()
    
    # print("resetting... (generating world)")
    # rng = jax.random.PRNGKey(seed=0)
    # obs, state = env.reset(rng)

    # print("generating custom world state...")
    # num_water = 0
    # num_wood = 3
    # custom_state = generate_test_world(rng, num_wood=num_wood, num_water=num_water, crafting_table=True, pickaxe=True)
    
    # # print("Running experiment")
    # # up, down, left, right, probs = run_experiment(env, custom_state, count_by=300)
    # # plt.plot(up, label="up")
    # # plt.plot(down, label="down")
    # # plt.plot(left, label="left")
    # # plt.plot(right, label="right")
    # # plt.legend()
    # # plt.savefig("/workspace/CraftaxDevinterp/ExperimentData/tree_water_experiment/images/movement_probs_water_9_wood_0.png")
    # # plt.close()

    # # with open("/workspace/CraftaxDevinterp/ExperimentData/tree_water_experiment/data/probs_water_9_wood_0.pkl", "wb") as f:
    # #     pickle.dump(probs, f)
    # # with open("/workspace/CraftaxDevinterp/ExperimentData/tree_water_experiment/data/state_water_9_wood_0.pkl", "wb") as f:
    # #     pickle.dump(custom_state, f)

    # print("rendering...")
    # rgb = render_craftax_pixels(
    #     custom_state,
    #     block_pixel_size=64, # or 16 or 64
    #     do_night_noise=True,
    # )
    # plt.imshow(rgb/255)
    # plt.savefig(f"/workspace/CraftaxDevinterp/ExperimentData/tree_water_experiment/images/map_water_{num_water}_wood_{num_wood}.png")
