#%%
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

#%%
def generate_test_world(
    rng,
    num_water, 
    num_wood, 
    pickaxe=False,
    crafting_table=False,
    sword=False,
    torch=0,
    placed_torch=False,
    num_stone = 0, 
    num_coal = 0, 
    num_iron = 0, 
    num_diamond = 0, 
    num_sapling = 0, 
    num_bow = 0, 
    num_arrows = 0, 
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
        block_map = jnp.zeros(static_params.map_size, dtype=jnp.int32)
        block_map = block_map.at[:,:].set(
            constants.BlockType.OUT_OF_BOUNDS.value
        )
        block_map = block_map.at[21:28,21:28].set(
            constants.BlockType.GRASS.value
        )
        block_map = block_map.at[21:23, 21:28].set(
            constants.BlockType.STONE.value
        )
        if crafting_table:
            block_map = block_map.at[24, 25].set(
                constants.BlockType.CRAFTING_TABLE.value
            )

        block_map = block_map.at[26:28, 21:28].set(
            constants.BlockType.STONE.value
        )
        block_map = block_map.at[27, 24].set(
            constants.BlockType.IRON.value
        )

        # for i in range(21, 28):
        #     if i % 2 == 1:
        #         block_map = block_map.at[21, i].set(
        #             constants.BlockType.TREE.value
        #         )
        #     else:
        #         block_map = block_map.at[22, i].set(
        #             constants.BlockType.TREE.value
        #         )

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
        if placed_torch:
            item_map = item_map.at[24, 23].set(
                constants.ItemType.TORCH.value
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

    def generate_melee_mob(max_mobs, x, y):
        position=jnp.zeros(
            (static_params.num_levels, max_mobs, 2),
            dtype=jnp.int32,
        )
        position = position.at[0, 0, 0].set(x)
        position = position.at[0, 0, 1].set(y)
        health=jnp.ones(
            (static_params.num_levels, max_mobs),
            dtype=jnp.float32,
        )
        mask=jnp.zeros(
            (static_params.num_levels, max_mobs),
            dtype=bool,
        )
        mask = mask.at[0, 0].set(False)
        attack_cooldown=jnp.zeros(
            (static_params.num_levels, max_mobs),
            dtype=jnp.int32,
        )
        type_id=jnp.zeros(
            (static_params.num_levels, max_mobs),
            dtype=jnp.int32,
        )
        # type_id = type_id.at[0, 0].set(1)
        return Mobs(
            position = position, 
            health = health,
            mask = mask, 
            attack_cooldown = attack_cooldown, 
            type_id = type_id
        )


    num_torch = torch
    num_pickaxe = pickaxe
    num_sword = sword
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
            constants.Action.DOWN.value,
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
            stone=jnp.asarray(num_stone, dtype=jnp.int32),
            coal=jnp.asarray(num_coal, dtype=jnp.int32),
            iron=jnp.asarray(num_iron, dtype=jnp.int32),
            diamond=jnp.asarray(num_diamond, dtype=jnp.int32),
            sapling=jnp.asarray(num_sapling, dtype=jnp.int32),
            pickaxe=jnp.asarray(num_pickaxe, dtype=jnp.int32),
            sword=jnp.asarray(num_sword, dtype=jnp.int32),
            bow=jnp.asarray(num_bow, dtype=jnp.int32),
            arrows=jnp.asarray(num_arrows, dtype=jnp.int32),
            torches=jnp.asarray(num_torch, dtype=jnp.int32),
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
        melee_mobs=generate_melee_mob(static_params.max_melee_mobs, 24, 23),
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

#%%
def optimized_experiment(
        num_wood, 
        num_water, 
        network_params,
        crafting_table, 
        pickaxe, 
        sword, 
        torch, 
        placed_torch, 
        num_stone, 
        num_coal, 
        num_iron, 
        num_diamond, 
        num_sapling, 
        num_bow, 
        num_arrows, 
        return_logits = False
):
    network = ActorCritic(43, 512)
    rng = jax.random.PRNGKey(seed=0)
    custom_state = generate_test_world(
        rng, 
        num_wood=num_wood, 
        num_water=num_water, 
        crafting_table=crafting_table, 
        pickaxe=pickaxe, 
        sword = sword, 
        torch = torch, 
        placed_torch = placed_torch, 
        num_stone = num_stone, 
        num_coal = num_coal, 
        num_iron = num_iron, 
        num_diamond = num_diamond, 
        num_sapling = num_sapling, 
        num_bow = num_bow, 
        num_arrows = num_arrows
    )
    obs = render_craftax_symbolic(custom_state)
    pi, _ = network.apply(network_params, obs)
    probs = pi.probs
    logits = pi.logits
    if return_logits: return logits
    return probs

#%%
def view_experiment(folder, wood_range, pickaxe_range, sword_range, stone_range, iron_range):
    import pickle
    import os
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display
    import numpy as np
    from ipywidgets import interactive, Output
    from ipywidgets.embed import embed_minimal_html

    results = list()
    for modelno in range(1525):
        with open(f"{folder}/{modelno}.pkl", "rb") as f:
            result = pickle.load(f)
        results.append(result)
    results = np.array(results)
    wood_slider = widgets.IntSlider(value=wood_range[0], min=wood_range[0], max=wood_range[1]-1, step=1, description='Wood')
    pickaxe_slider = widgets.IntSlider(value=pickaxe_range[0], min=pickaxe_range[0], max=pickaxe_range[1]-1, step=1, description='Pickaxe')
    sword_slider = widgets.IntSlider(value=sword_range[0], min=sword_range[0], max=sword_range[1]-1, step=1, description='Sword')
    stone_slider = widgets.IntSlider(value=stone_range[0], min=stone_range[0], max=stone_range[1]-1, step=1, description='Stone')
    iron_slider = widgets.IntSlider(value=iron_range[0], min=iron_range[0], max=iron_range[1]-1, step=1, description='Iron')

    plot_output = Output()

    def update_plot(num_wood, num_pickaxe, num_sword, num_stone, num_iron):
        with plot_output:
            plot_output.clear_output(wait=True)
            plt.figure(figsize=(10,6))
            probs = results[:, num_pickaxe, num_wood, num_sword, num_stone, num_iron, :]
            plt.plot(probs[:, 0], label="NOOP", color='b', linestyle='-')
            plt.plot(probs[:, 1], label="LEFT", color='g', linestyle='-')
            plt.plot(probs[:, 2], label="RIGHT", color='r', linestyle='-')
            plt.plot(probs[:, 3], label="UP", color='c', linestyle='-')
            plt.plot(probs[:, 4], label="DOWN", color='m', linestyle='-')
            plt.plot(probs[:, 5], label="DO", color='y', linestyle='-')
            plt.plot(probs[:, 6], label="SLEEP", color='k', linestyle='-')
            plt.plot(probs[:, 7], label="PLACE_STONE", color='orange', linestyle='-')
            plt.plot(probs[:, 8], label="PLACE_TABLE", color='purple', linestyle='-')
            plt.plot(probs[:, 9], label="PLACE_FURNACE", color='brown', linestyle='-')
            plt.plot(probs[:, 10], label="PLACE_PLANT", color='pink', linestyle='-')
            plt.plot(probs[:, 11], label="MAKE_WOOD_PICKAXE", color='gray', linestyle='-')
            plt.plot(probs[:, 12], label="MAKE_STONE_PICKAXE", color='olive', linestyle='-')
            plt.plot(probs[:, 13], label="MAKE_IRON_PICKAXE", color='cyan', linestyle='-')
            plt.plot(probs[:, 14], label="MAKE_WOOD_SWORD", color='navy', linestyle='-')
            plt.plot(probs[:, 15], label="MAKE_STONE_SWORD", color='teal', linestyle='-')
            plt.plot(probs[:, 16], label="MAKE_IRON_SWORD", color='lime', linestyle='-')
            plt.plot(probs[:, 17], label="REST", color='indigo', linestyle='-')
            plt.plot(probs[:, 18], label="DESCEND", color='violet', linestyle='-')
            plt.plot(probs[:, 19], label="ASCEND", color='gold', linestyle='-')
            plt.plot(probs[:, 20], label="MAKE_DIAMOND_PICKAXE", color='b', linestyle='--')
            plt.plot(probs[:, 21], label="MAKE_DIAMOND_SWORD", color='g', linestyle='--')
            plt.plot(probs[:, 22], label="MAKE_IRON_ARMOUR", color='r', linestyle='--')
            plt.plot(probs[:, 23], label="MAKE_DIAMOND_ARMOUR", color='c', linestyle='--')
            plt.plot(probs[:, 24], label="SHOOT_ARROW", color='m', linestyle='--')
            plt.plot(probs[:, 25], label="MAKE_ARROW", color='y', linestyle='--')
            plt.plot(probs[:, 26], label="CAST_FIREBALL", color='k', linestyle='--')
            plt.plot(probs[:, 27], label="CAST_ICEBALL", color='orange', linestyle='--')
            plt.plot(probs[:, 28], label="PLACE_TORCH", color='purple', linestyle='--')
            plt.plot(probs[:, 29], label="DRINK_POTION_RED", color='brown', linestyle='--')
            plt.plot(probs[:, 30], label="DRINK_POTION_GREEN", color='pink', linestyle='--')
            plt.plot(probs[:, 31], label="DRINK_POTION_BLUE", color='gray', linestyle='--')
            plt.plot(probs[:, 32], label="DRINK_POTION_PINK", color='olive', linestyle='--')
            plt.plot(probs[:, 33], label="DRINK_POTION_CYAN", color='cyan', linestyle='--')
            plt.plot(probs[:, 34], label="DRINK_POTION_YELLOW", color='navy', linestyle='--')
            plt.plot(probs[:, 35], label="READ_BOOK", color='teal', linestyle='--')
            plt.plot(probs[:, 36], label="ENCHANT_SWORD", color='lime', linestyle='--')
            plt.plot(probs[:, 37], label="ENCHANT_ARMOUR", color='indigo', linestyle='--')
            plt.plot(probs[:, 38], label="MAKE_TORCH", color='violet', linestyle='--')
            plt.plot(probs[:, 39], label="LEVEL_UP_DEXTERITY", color='gold', linestyle='--')
            plt.plot(probs[:, 40], label="LEVEL_UP_STRENGTH", color='b', linestyle='-.')
            plt.plot(probs[:, 41], label="LEVEL_UP_INTELLIGENCE", color='g', linestyle='-.')
            plt.plot(probs[:, 42], label="ENCHANT_BOW", color='r', linestyle='-.')
            plt.title(f"Probability Curves for {num_wood} Wood, {num_pickaxe} Pickaxe, {num_sword} Sword, {num_stone} Stone, {num_iron} Iron")
            plt.xlabel('Steps')
            plt.ylabel('Probability')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True)
            plt.show()
    wood_water_interact = interactive(update_plot, num_wood=wood_slider, num_pickaxe=pickaxe_slider, num_sword=sword_slider, num_stone=stone_slider, num_iron=iron_slider)
    display(widgets.VBox([wood_water_interact, plot_output]))  

def view_experiment_stacked(folder, wood_range, pickaxe_range, sword_range, stone_range, iron_range):
    import pickle
    import os
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display
    import numpy as np
    from ipywidgets import interactive, Output
    from ipywidgets.embed import embed_minimal_html
    results = list()
    for modelno in range(1525):
        with open(f"{folder}/{modelno}.pkl", "rb") as f:
            result = pickle.load(f)
        results.append(result)
    results = np.array(results)
    
    wood_slider = widgets.IntSlider(value=wood_range[0], min=wood_range[0], max=wood_range[1]-1, step=1, description='Wood')
    pickaxe_slider = widgets.IntSlider(value=pickaxe_range[0], min=pickaxe_range[0], max=pickaxe_range[1]-1, step=1, description='Pickaxe')
    sword_slider = widgets.IntSlider(value=sword_range[0], min=sword_range[0], max=sword_range[1]-1, step=1, description='Sword')
    stone_slider = widgets.IntSlider(value=stone_range[0], min=stone_range[0], max=stone_range[1]-1, step=1, description='Stone')
    iron_slider = widgets.IntSlider(value=iron_range[0], min=iron_range[0], max=iron_range[1]-1, step=1, description='Iron')

    plot_output = Output()

    def update_plot(num_wood, num_pickaxe, num_sword, num_stone, num_iron):
        with plot_output:
            plot_output.clear_output(wait=True)
            plt.figure(figsize=(10,6))
            probs = results[:, num_pickaxe, num_wood, num_sword, num_stone, num_iron, :]
            
            # Normalize the probabilities so they sum to 1 across all actions at each step
            probs /= probs.sum(axis=1, keepdims=True)

            # List of actions for better code readability and maintenance
            actions = [
                "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE_STONE",
                "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT", "MAKE_WOOD_PICKAXE",
                "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE", "MAKE_WOOD_SWORD",
                "MAKE_STONE_SWORD", "MAKE_IRON_SWORD", "REST", "DESCEND", "ASCEND",
                "MAKE_DIAMOND_PICKAXE", "MAKE_DIAMOND_SWORD", "MAKE_IRON_ARMOUR",
                "MAKE_DIAMOND_ARMOUR", "SHOOT_ARROW", "MAKE_ARROW", "CAST_FIREBALL",
                "CAST_ICEBALL", "PLACE_TORCH", "DRINK_POTION_RED", "DRINK_POTION_GREEN",
                "DRINK_POTION_BLUE", "DRINK_POTION_PINK", "DRINK_POTION_CYAN",
                "DRINK_POTION_YELLOW", "READ_BOOK", "ENCHANT_SWORD", "ENCHANT_ARMOUR",
                "MAKE_TORCH", "LEVEL_UP_DEXTERITY", "LEVEL_UP_STRENGTH",
                "LEVEL_UP_INTELLIGENCE", "ENCHANT_BOW"
            ]

            plt.stackplot(range(probs.shape[0]), *probs.T, labels=actions)
            plt.title(f"Fractional Probability Curves for {num_wood} Wood, {num_pickaxe} Pickaxe, {num_sword} Sword, {num_stone} Stone, {num_iron} Iron")
            plt.xlabel('Steps')
            plt.ylabel('Fraction of Probability')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True)
            plt.show()

    # Interactive widget setup
    interact_widget = interactive(update_plot, num_wood=wood_slider, num_pickaxe=pickaxe_slider, num_sword=sword_slider, num_stone=stone_slider, num_iron=iron_slider)
    display(widgets.VBox([interact_widget, plot_output]))

#%%
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

#%%
def restricted_ed(env_state, models=1525, num_pcs = 3):
    import os
    import orbax.checkpoint as ocp
    from tqdm import tqdm
    import numpy as np
    from sklearn.decomposition import PCA

    network = ActorCritic(43, 512)
    logits = np.zeros((models, 43))
    obs = render_craftax_symbolic(env_state)
    for model in tqdm(range(0, models)):
        checkpointer = ocp.StandardCheckpointer()
        checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{model}"
        network_params = checkpointer.restore(f"{checkpoint_directory}/model_{model}")
        pi, _ = network.apply(network_params, obs)
        logits[model, :] = pi.logits
    
    pca = PCA(n_components=num_pcs)
    pca.fit(logits)
    projected = pca.transform(logits)
    return projected, pca
    
def make_experiment_pca(
        data_folder, 
        range_wood, 
        range_pickaxe, 
        range_sword,
        range_stone, 
        range_iron, 
        num_pcs = 3, 
        save_folder = None
):
    import pickle
    import os
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display
    import numpy as np
    from ipywidgets import interactive, Output
    from ipywidgets.embed import embed_minimal_html
    from sklearn.decomposition import PCA
    import itertools
    assert "logits" in data_folder, "Error: Data folder must be folder of logits"
    if save_folder is None:
        save_folder = data_folder.replace("logits", "pca")
    os.makedirs(save_folder, exist_ok=True)
    results = list()
    for modelno in range(1525):
        with open(f"{data_folder}/{modelno}.pkl", "rb") as f:
            result = pickle.load(f)
        results.append(result)
    results = np.array(results)

    range_wood = np.arange(*range_wood, 1)
    range_pickaxe = np.arange(*range_pickaxe, 1)
    range_sword = np.arange(*range_sword, 1)
    range_stone = np.arange(*range_stone, 1)
    range_iron = np.arange(*range_iron, 1)
    num_iters = len(range_wood) * len(range_pickaxe) * len(range_sword) * len(range_stone) * len(range_iron)
    for num_wood, num_pickaxe, num_sword, num_stone, num_iron in tqdm(itertools.product(range_wood, range_pickaxe, range_sword, range_stone, range_iron), total=num_iters):
        result = results[:, num_pickaxe, num_wood, num_sword, num_stone, num_iron, :]
        pca = PCA(n_components=num_pcs)
        pca.fit(result)
        projected = pca.transform(result)
        with open(f"{save_folder}/{num_wood}_{num_pickaxe}_{num_sword}_{num_stone}_{num_iron}.pkl", "wb") as f:
            pickle.dump(projected, f)

def view_experiment_pca(
        data_folder, 
        range_wood, 
        range_pickaxe, 
        range_sword,
        range_stone, 
        range_iron, 
        num_pcs = 3
):
    import pickle
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display
    import numpy as np
    from ipywidgets import interactive, Output
    from ipywidgets.embed import embed_minimal_html
    from sklearn.decomposition import PCA
    import itertools
    assert "pca" in data_folder, "Error: Data folder must be folder of pca"
    results = list()
    product = list(itertools.product(np.arange(*range_wood, 1), np.arange(*range_pickaxe, 1), np.arange(*range_sword, 1), np.arange(*range_stone, 1), np.arange(*range_iron, 1)))
    for num_wood, num_pickaxe, num_sword, num_stone, num_iron in product:
        with open(f"{data_folder}/{num_wood}_{num_pickaxe}_{num_sword}_{num_stone}_{num_iron}.pkl", "rb") as f:
            result = pickle.load(f)
        results.append(result)
    results = np.array(results)
    wood_slider = widgets.IntSlider(value=range_wood[0], min=range_wood[0], max=range_wood[1]-1, step=1, description='Wood')
    pickaxe_slider = widgets.IntSlider(value=range_pickaxe[0], min=range_pickaxe[0], max=range_pickaxe[1]-1, step=1, description='Pickaxe')
    sword_slider = widgets.IntSlider(value=range_sword[0], min=range_sword[0], max=range_sword[1]-1, step=1, description='Sword')
    stone_slider = widgets.IntSlider(value=range_stone[0], min=range_stone[0], max=range_stone[1]-1, step=1, description='Stone')
    iron_slider = widgets.IntSlider(value=range_iron[0], min=range_iron[0], max=range_iron[1]-1, step=1, description='Iron')

    plot_output = Output()

    def update_plot(num_wood, num_pickaxe, num_sword, num_stone, num_iron):
        with plot_output:
            plot_output.clear_output(wait=True)
            fig, ax = plt.subplots(len(list(itertools.combinations(range(num_pcs), 2))), 1, figsize=(10, 6*len(list(itertools.combinations(range(num_pcs), 2)))))
            result_num = product.index((num_wood, num_pickaxe, num_sword, num_stone, num_iron))
            result = results[result_num, ...]
            for i, (pcx, pcy) in enumerate(itertools.combinations(range(num_pcs), 2)):
                sc = ax[i].scatter(result[:, pcx], result[:, pcy], label=f"PC{pcx} vs PC{pcy}", s=1, c=np.arange(1525), cmap="rainbow")
                ax[i].set_xlabel(f"PC{pcx}")
                ax[i].set_ylabel(f"PC{pcy}")
                # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax[i].grid(True)
            fig.suptitle(f"PCA for {num_wood} Wood, {num_pickaxe} Pickaxe, {num_sword} Sword, {num_stone} Stone, {num_iron} Iron")
            cbar = plt.colorbar(sc, ax=ax[-1])
            plt.show()
    wood_water_interact = interactive(update_plot, num_wood=wood_slider, num_pickaxe=pickaxe_slider, num_sword=sword_slider, num_stone=stone_slider, num_iron=iron_slider)
    display(widgets.VBox([wood_water_interact, plot_output]))



import plotly.graph_objects as go
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import pickle
import itertools
from plotly.subplots import make_subplots

def view_experiment_pca_plotly(
        data_folder, 
        range_wood, 
        range_pickaxe, 
        range_sword,
        range_stone, 
        range_iron, 
        num_pcs=3
):
    assert "pca" in data_folder, "Error: Data folder must be folder of pca"
    product = list(itertools.product(np.arange(*range_wood, 1), np.arange(*range_pickaxe, 1), np.arange(*range_sword, 1), np.arange(*range_stone, 1), np.arange(*range_iron, 1)))
    
    results = []
    for num_wood, num_pickaxe, num_sword, num_stone, num_iron in product:
        with open(f"{data_folder}/{num_wood}_{num_pickaxe}_{num_sword}_{num_stone}_{num_iron}.pkl", "rb") as f:
            result = pickle.load(f)
        results.append(result)
    results = np.array(results)

    # Widgets
    wood_slider = widgets.IntSlider(value=range_wood[0], min=range_wood[0], max=range_wood[1]-1, step=1, description='Wood')
    pickaxe_slider = widgets.IntSlider(value=range_pickaxe[0], min=range_pickaxe[0], max=range_pickaxe[1]-1, step=1, description='Pickaxe')
    sword_slider = widgets.IntSlider(value=range_sword[0], min=range_sword[0], max=range_sword[1]-1, step=1, description='Sword')
    stone_slider = widgets.IntSlider(value=range_stone[0], min=range_stone[0], max=range_stone[1]-1, step=1, description='Stone')
    iron_slider = widgets.IntSlider(value=range_iron[0], min=range_iron[0], max=range_iron[1]-1, step=1, description='Iron')
    
    pc_combinations = list(itertools.combinations(range(num_pcs), 2))
    num_combinations = len(pc_combinations)
    fig = make_subplots(rows=num_combinations, cols=1,
                        subplot_titles=[f"PC{pcx} vs PC{pcy}" for pcx, pcy in pc_combinations])

    # Initial empty plots setup for each combination of principal components
    for i, (pcx, pcy) in enumerate(pc_combinations, start=1):
        fig.add_trace(go.Scattergl(x=[], y=[], mode='markers',
                                   marker=dict(size=5, colorscale='Rainbow'),
                                   name=f"PC{pcx} vs PC{pcy}"),
                      row=i, col=1)

    def update_plot(num_wood, num_pickaxe, num_sword, num_stone, num_iron):
        # Clear all plots first
        fig.data = []

        # Re-add plots with new data
        result_num = product.index((num_wood, num_pickaxe, num_sword, num_stone, num_iron))
        result = results[result_num, ...]
        for i, (pcx, pcy) in enumerate(pc_combinations, start=1):
            fig.add_trace(go.Scattergl(x=result[:, pcx], y=result[:, pcy], mode='markers',
                                       marker=dict(size=5, color=np.arange(result.shape[0]), colorscale='Rainbow'),
                                       name=f"PC{pcx} vs PC{pcy}"),
                          row=i, col=1)
        fig.update_layout(height=300 * num_combinations, title_text=f"PCA for {num_wood} Wood, {num_pickaxe} Pickaxe, {num_sword} Sword, {num_stone} Stone, {num_iron} Iron")
        fig.show()

    interact_widget = widgets.interactive(update_plot, num_wood=wood_slider, num_pickaxe=pickaxe_slider, num_sword=sword_slider, num_stone=stone_slider, num_iron=iron_slider)
    display(interact_widget)


#%%
def experiment_with_varied_water_and_wood(
        wood_range: tuple, 
        water_range: tuple, 
        models=1525, 
        count_by=1, 
        save_dir = "/workspace/CraftaxDevinterp/ExperimentData/tree_water_experiment/data", 
        plotting = True, 
        crafting_table = False, 
        pickaxe = False, 
        sword = False, 
        num_stone = 0
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
            custom_state = generate_test_world(rng, num_wood=num_wood, num_water=num_water, crafting_table=crafting_table, pickaxe=pickaxe, sword = sword, num_stone=num_stone)
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
                plt.plot(probs[:, 0], label="NOOP", color='b', linestyle='-')
                plt.plot(probs[:, 1], label="LEFT", color='g', linestyle='-')
                plt.plot(probs[:, 2], label="RIGHT", color='r', linestyle='-')
                plt.plot(probs[:, 3], label="UP", color='c', linestyle='-')
                plt.plot(probs[:, 4], label="DOWN", color='m', linestyle='-')
                plt.plot(probs[:, 5], label="DO", color='y', linestyle='-')
                plt.plot(probs[:, 6], label="SLEEP", color='k', linestyle='-')
                plt.plot(probs[:, 7], label="PLACE_STONE", color='orange', linestyle='-')
                plt.plot(probs[:, 8], label="PLACE_TABLE", color='purple', linestyle='-')
                plt.plot(probs[:, 9], label="PLACE_FURNACE", color='brown', linestyle='-')
                plt.plot(probs[:, 10], label="PLACE_PLANT", color='pink', linestyle='-')
                plt.plot(probs[:, 11], label="MAKE_WOOD_PICKAXE", color='gray', linestyle='-')
                plt.plot(probs[:, 12], label="MAKE_STONE_PICKAXE", color='olive', linestyle='-')
                plt.plot(probs[:, 13], label="MAKE_IRON_PICKAXE", color='cyan', linestyle='-')
                plt.plot(probs[:, 14], label="MAKE_WOOD_SWORD", color='navy', linestyle='-')
                plt.plot(probs[:, 15], label="MAKE_STONE_SWORD", color='teal', linestyle='-')
                plt.plot(probs[:, 16], label="MAKE_IRON_SWORD", color='lime', linestyle='-')
                plt.plot(probs[:, 17], label="REST", color='indigo', linestyle='-')
                plt.plot(probs[:, 18], label="DESCEND", color='violet', linestyle='-')
                plt.plot(probs[:, 19], label="ASCEND", color='gold', linestyle='-')
                plt.plot(probs[:, 20], label="MAKE_DIAMOND_PICKAXE", color='b', linestyle='--')
                plt.plot(probs[:, 21], label="MAKE_DIAMOND_SWORD", color='g', linestyle='--')
                plt.plot(probs[:, 22], label="MAKE_IRON_ARMOUR", color='r', linestyle='--')
                plt.plot(probs[:, 23], label="MAKE_DIAMOND_ARMOUR", color='c', linestyle='--')
                plt.plot(probs[:, 24], label="SHOOT_ARROW", color='m', linestyle='--')
                plt.plot(probs[:, 25], label="MAKE_ARROW", color='y', linestyle='--')
                plt.plot(probs[:, 26], label="CAST_FIREBALL", color='k', linestyle='--')
                plt.plot(probs[:, 27], label="CAST_ICEBALL", color='orange', linestyle='--')
                plt.plot(probs[:, 28], label="PLACE_TORCH", color='purple', linestyle='--')
                plt.plot(probs[:, 29], label="DRINK_POTION_RED", color='brown', linestyle='--')
                plt.plot(probs[:, 30], label="DRINK_POTION_GREEN", color='pink', linestyle='--')
                plt.plot(probs[:, 31], label="DRINK_POTION_BLUE", color='gray', linestyle='--')
                plt.plot(probs[:, 32], label="DRINK_POTION_PINK", color='olive', linestyle='--')
                plt.plot(probs[:, 33], label="DRINK_POTION_CYAN", color='cyan', linestyle='--')
                plt.plot(probs[:, 34], label="DRINK_POTION_YELLOW", color='navy', linestyle='--')
                plt.plot(probs[:, 35], label="READ_BOOK", color='teal', linestyle='--')
                plt.plot(probs[:, 36], label="ENCHANT_SWORD", color='lime', linestyle='--')
                plt.plot(probs[:, 37], label="ENCHANT_ARMOUR", color='indigo', linestyle='--')
                plt.plot(probs[:, 38], label="MAKE_TORCH", color='violet', linestyle='--')
                plt.plot(probs[:, 39], label="LEVEL_UP_DEXTERITY", color='gold', linestyle='--')
                plt.plot(probs[:, 40], label="LEVEL_UP_STRENGTH", color='b', linestyle='-.')
                plt.plot(probs[:, 41], label="LEVEL_UP_INTELLIGENCE", color='g', linestyle='-.')
                plt.plot(probs[:, 42], label="ENCHANT_BOW", color='r', linestyle='-.')
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

#%%
# if __name__ == "__main__":
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

#%%
wood = 0
water = 9
crafting_table = False
placed_torch = True
pickaxe = 2
sword = 0
torch = 0
num_stone = 0
num_coal = 0
num_iron = 0
num_diamond = 0
num_sapling = 0
num_bow = 0
num_arrows = 0
experiment_mapped = jax.jit(
    jax.vmap(
        jax.vmap(
            jax.vmap(
                jax.vmap(
                    jax.vmap(
                        lambda wood, pickaxe, sword, num_stone, num_iron, params: optimized_experiment(
                            wood,
                            water, 
                            params,
                            crafting_table,
                            pickaxe,
                            sword,
                            torch,
                            placed_torch,
                            num_stone,
                            num_coal,
                            num_iron,
                            num_diamond,
                            num_sapling,
                            num_bow,
                            num_arrows
                        ), 
                        (0, 0, 0, 0, 0, None)
                    ), 
                    (0, 0, 0, 0, 0, None)
                ), 
                (0, 0, 0, 0, 0, None)
            ),
            (0, 0, 0, 0, 0, None)
        ),
        (0, 0, 0, 0, 0, None)
    )
)

wood_range = jnp.arange(0, 15, 1)
water_range = jnp.arange(0, 10, 1)
pickaxe_range = jnp.arange(0, 4, 1)
sword_range = jnp.arange(0, 4, 1)
torch_range = jnp.arange(0, 20, 1)
stone_range = jnp.arange(0, 15, 1)
coal_range = jnp.arange(0, 15, 1)
iron_range = jnp.arange(0, 15, 1)
diamond_range = jnp.arange(0, 20, 1)
sapling_range = jnp.arange(0, 20, 1)
bow_range = jnp.arange(0, 2, 1)
arrows_range = jnp.arange(0, 20, 1)
wood, pickaxe, sword, num_stone, num_iron = jnp.meshgrid(
    wood_range, 
    pickaxe_range, 
    sword_range, 
    stone_range, 
    iron_range, 
)

#%%
t0 = time.time()
os.makedirs("/workspace/CraftaxDevinterp/ExperimentData/all_inventory/results", exist_ok=True)
for modelno in tqdm(range(1525)):
    checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{modelno}"
    folder_list = os.listdir(checkpoint_directory)
    network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
    result = experiment_mapped(wood, pickaxe, sword, num_stone, num_iron, network_params)
    with open(f"/workspace/CraftaxDevinterp/ExperimentData/all_inventory/results/{modelno}.pkl", "wb") as f:
        pickle.dump(result, f)
print(f"Time for mapped: {time.time() - t0}")
#%%

wood = 0
water = 9
crafting_table = False
placed_torch = True
pickaxe = 2
sword = 0
torch = 0
num_stone = 0
num_coal = 0
num_iron = 0
num_diamond = 0
num_sapling = 0
num_bow = 0
num_arrows = 0
experiment_mapped = jax.jit(
    jax.vmap(
        jax.vmap(
            jax.vmap(
                jax.vmap(
                    jax.vmap(
                        lambda wood, pickaxe, sword, num_stone, num_iron, params: optimized_experiment(
                            wood,
                            water, 
                            params,
                            crafting_table,
                            pickaxe,
                            sword,
                            torch,
                            placed_torch,
                            num_stone,
                            num_coal,
                            num_iron,
                            num_diamond,
                            num_sapling,
                            num_bow,
                            num_arrows, 
                            return_logits = True
                        ), 
                        (0, 0, 0, 0, 0, None)
                    ), 
                    (0, 0, 0, 0, 0, None)
                ), 
                (0, 0, 0, 0, 0, None)
            ),
            (0, 0, 0, 0, 0, None)
        ),
        (0, 0, 0, 0, 0, None)
    )
)

wood_range = jnp.arange(0, 15, 1)
water_range = jnp.arange(0, 10, 1)
pickaxe_range = jnp.arange(0, 4, 1)
sword_range = jnp.arange(0, 4, 1)
torch_range = jnp.arange(0, 20, 1)
stone_range = jnp.arange(0, 15, 1)
coal_range = jnp.arange(0, 15, 1)
iron_range = jnp.arange(0, 15, 1)
diamond_range = jnp.arange(0, 20, 1)
sapling_range = jnp.arange(0, 20, 1)
bow_range = jnp.arange(0, 2, 1)
arrows_range = jnp.arange(0, 20, 1)
wood, pickaxe, sword, num_stone, num_iron = jnp.meshgrid(
    wood_range, 
    pickaxe_range, 
    sword_range, 
    stone_range, 
    iron_range, 
)

t0 = time.time()
os.makedirs("/workspace/CraftaxDevinterp/ExperimentData/all_inventory/logits", exist_ok=True)
for modelno in tqdm(range(1525)):
    checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{modelno}"
    folder_list = os.listdir(checkpoint_directory)
    network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
    result = experiment_mapped(wood, pickaxe, sword, num_stone, num_iron, network_params)
    with open(f"/workspace/CraftaxDevinterp/ExperimentData/all_inventory/logits/{modelno}.pkl", "wb") as f:
        pickle.dump(result, f)

#%%
view_experiment_stacked(
    "/workspace/CraftaxDevinterp/ExperimentData/all_inventory/results", 
    (0, 15), 
    (0, 4), 
    (0, 4), 
    (0, 15), 
    (0, 15)
)

#%%
make_experiment_pca(
    "/workspace/CraftaxDevinterp/ExperimentData/all_inventory/logits",
    (0, 15), 
    (0, 4), 
    (0, 4), 
    (0, 15), 
    (0, 15),
)
#%%
view_experiment_pca(
    "/workspace/CraftaxDevinterp/ExperimentData/all_inventory/pca",
    (0, 15), 
    (0, 4), 
    (0, 4), 
    (0, 15), 
    (0, 15),
)


#%%
# view_experiment_pca_plotly(
#     "/workspace/CraftaxDevinterp/ExperimentData/all_inventory/pca",
#     (0, 15), 
#     (0, 4), 
#     (0, 4), 
#     (0, 15), 
#     (0, 15),
# )
# #%%
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
from craftax.craftax.renderer import render_craftax_pixels
# from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
import matplotlib.pyplot as plt
# import numpy as np
# import pickle
# import os
# os.makedirs("/workspace/CraftaxDevinterp/ExperimentData/tree_water_experiment/images", exist_ok=True)
# os.makedirs("/workspace/CraftaxDevinterp/ExperimentData/tree_water_experiment/data", exist_ok=True)

# print("initialising...")
# env = CraftaxSymbolicEnv()

# print("resetting... (generating world)")
rng = jax.random.PRNGKey(seed=0)
# obs, state = env.reset(rng)

# print("generating custom world state...")
# num_water = 0
# num_wood = 3
custom_state = generate_test_world(
    rng, 
    num_wood=wood, 
    num_water=0, 
    crafting_table=crafting_table, 
    pickaxe=pickaxe, 
    sword=sword,
    torch=torch, 
    placed_torch=placed_torch
)

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

print("rendering...")
rgb = render_craftax_pixels(
    custom_state,
    block_pixel_size=64, # or 16 or 64
    do_night_noise=True
)
plt.imshow(rgb/255)
plt.savefig(f"/workspace/CraftaxDevinterp/ExperimentData/tree_water_experiment/images/map_water_{0}_wood_{19}.png")
