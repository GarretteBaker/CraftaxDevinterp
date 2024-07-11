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

import craftax.craftax_classic.constants as constants
from craftax.craftax_classic.envs.craftax_state import EnvState, Inventory, Mobs
from craftax.craftax_classic.envs.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax_classic.game_logic import calculate_light_level
from craftax.craftax_classic.renderer import render_craftax_symbolic, render_craftax_pixels
from craftax.models.actor_critic import ActorCritic
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv

#%%
def generate_test_world(
    rng,
    num_water, 
    num_wood, 
    pickaxe=False,
    crafting_table=False,
    sword=False,
    num_stone = 0, 
    num_coal = 0, 
    num_iron = 0, 
    num_diamond = 0, 
    num_sapling = 0, 
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
    rng, rng_map = jax.random.split(rng)
    rng, rng_state = jax.random.split(rng)

    def generate_tree_water_map_layer(map_rng):
        # hardcode world structure
        block_map = jnp.zeros(static_params.map_size, dtype=jnp.int32)
        block_map = block_map.at[:,:].set(
            constants.BlockType.OUT_OF_BOUNDS.value
        )
        block_map = block_map.at[29:36,29:36].set(
            constants.BlockType.GRASS.value
        )
        block_map = block_map.at[29:31, 29:36].set(
            constants.BlockType.STONE.value
        )
        if crafting_table:
            block_map = block_map.at[32, 33].set(
                constants.BlockType.CRAFTING_TABLE.value
            )

        block_map = block_map.at[34:36, 29:36].set(
            constants.BlockType.STONE.value
        )
        block_map = block_map.at[35, 32].set(
            constants.BlockType.IRON.value
        )
        return block_map


    block_map = generate_tree_water_map_layer(rng_map)


    def generate_empty_mobs(max_mobs):
        return Mobs(
            position=jnp.zeros(
                (max_mobs, 2),
                dtype=jnp.int32,
            ),
            health=jnp.ones(
                (max_mobs),
                dtype=jnp.float32,
            ),
            mask=jnp.zeros(
                (max_mobs),
                dtype=bool,
            ),
            attack_cooldown=jnp.zeros(
                (max_mobs),
                dtype=jnp.int32,
            )
        )

    def generate_melee_mob(max_mobs, x, y):
        position=jnp.zeros(
            (max_mobs, 2),
            dtype=jnp.int32,
        )
        position = position.at[0, 0].set(x)
        position = position.at[0, 1].set(y)
        health=jnp.ones(
            (max_mobs),
            dtype=jnp.float32,
        )
        mask=jnp.zeros(
            (max_mobs),
            dtype=bool,
        )
        mask = mask.at[0].set(False)
        attack_cooldown=jnp.zeros(
            (max_mobs),
            dtype=jnp.int32,
        )
        # type_id = type_id.at[0, 0].set(1)
        return Mobs(
            position = position, 
            health = health,
            mask = mask, 
            attack_cooldown = attack_cooldown, 
        )


    num_pickaxe = pickaxe
    num_sword = sword
    state = EnvState(
        # world
        map=block_map,

        # player
        player_position=jnp.array((
            static_params.map_size[0] // 2,
            static_params.map_size[1] // 2,
        )),
        player_direction=jnp.asarray(
            constants.Action.DOWN.value,
            dtype=jnp.int32,
        ),
        player_health=jnp.asarray(9.0, dtype=jnp.int32),
        player_food=jnp.asarray(9, dtype=jnp.int32),
        player_drink=jnp.asarray(num_water, dtype=jnp.int32),
        player_energy=jnp.asarray(9, dtype=jnp.int32),
        player_recover=jnp.asarray(0.0, dtype=jnp.int32),
        player_hunger=jnp.asarray(0.0, dtype=jnp.int32),
        player_thirst=jnp.asarray(0.0, dtype=jnp.int32),
        player_fatigue=jnp.asarray(0.0, dtype=jnp.int32),
        is_sleeping=False,

        # inventory
        inventory=Inventory(
            wood=jnp.asarray(num_wood, dtype=jnp.int32),
            stone=jnp.asarray(num_stone, dtype=jnp.int32),
            coal=jnp.asarray(num_coal, dtype=jnp.int32),
            iron=jnp.asarray(num_iron, dtype=jnp.int32),
            diamond=jnp.asarray(num_diamond, dtype=jnp.int32),
            sapling=jnp.asarray(num_sapling, dtype=jnp.int32),
            wood_pickaxe=jnp.asarray(num_pickaxe, dtype=jnp.int32),
            stone_pickaxe=jnp.asarray(0, dtype=jnp.int32),
            iron_pickaxe=jnp.asarray(0, dtype=jnp.int32),
            wood_sword=jnp.asarray(num_sword, dtype=jnp.int32),
            stone_sword=jnp.asarray(0, dtype=jnp.int32),
            iron_sword=jnp.asarray(0, dtype=jnp.int32)
        ), 
    
        # mobs
        mob_map = jnp.zeros(
            static_params.map_size,
            dtype=bool,
        ),
        zombies=generate_melee_mob(static_params.max_zombies, 24, 23),
        cows=generate_empty_mobs(static_params.max_cows),
        skeletons=generate_empty_mobs(static_params.max_skeletons),
        arrows=generate_empty_mobs(static_params.max_arrows),
        arrow_directions=jnp.ones(
            (static_params.max_arrows, 2),
            dtype=jnp.int32,
        ),
        
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
        num_stone, 
        num_coal, 
        num_iron, 
        num_diamond, 
        num_sapling, 
        return_logits = False
):
    network = ActorCritic(17, 512)
    rng = jax.random.PRNGKey(seed=0)
    custom_state = generate_test_world(
        rng, 
        num_wood=num_wood, 
        num_water=num_water, 
        crafting_table=crafting_table, 
        num_stone = num_stone, 
        num_coal = num_coal, 
        num_iron = num_iron, 
        num_diamond = num_diamond, 
        num_sapling = num_sapling, 
    )
    obs = render_craftax_symbolic(custom_state)
    pi, _ = network.apply(network_params, obs)
    probs = pi.probs
    logits = pi.logits
    if return_logits: return logits
    return probs

#%%
def view_experiment(folder, wood_range, stone_range, coal_range, iron_range):
    import pickle
    import os
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display
    import numpy as np
    from ipywidgets import interactive, Output
    from ipywidgets.embed import embed_minimal_html

    results = np.zeros((1525, 16, 15, 17, 18, 17))
    for modelno in tqdm(range(1525)):
        with open(f"{folder}/{modelno}.pkl", "rb") as f:
            result = pickle.load(f)
        results[modelno, :] = result
    print("making sliders")
    wood_slider = widgets.IntSlider(value=wood_range[0], min=wood_range[0], max=wood_range[1]-1, step=1, description='Wood')
    stone_slider = widgets.IntSlider(value=stone_range[0], min=stone_range[0], max=stone_range[1]-1, step=1, description='Stone')
    coal_slider = widgets.IntSlider(value=coal_range[0], min=coal_range[0], max=coal_range[1]-1, step=1, description='Coal')
    iron_slider = widgets.IntSlider(value=iron_range[0], min=iron_range[0], max=iron_range[1]-1, step=1, description='Iron')

    plot_output = Output()

    def update_plot(num_wood, num_stone, num_coal, num_iron):
        with plot_output:
            plot_output.clear_output(wait=True)
            plt.figure(figsize=(10,6))
            probs = results[:, num_stone, num_wood, num_coal, num_iron, :]
            print(probs.shape)
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
            plt.title(f"Probability Curves for {num_wood} wood, {num_stone} stone, {num_coal} coal, {num_iron} iron")
            plt.xlabel('Steps')
            plt.ylabel('Probability')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True)
            plt.show()
    print("making plot")
    wood_water_interact = interactive(update_plot, num_wood=wood_slider, num_stone = stone_slider, num_coal = coal_slider, num_iron = iron_slider)
    display(widgets.VBox([wood_water_interact, plot_output]))  

def view_experiment_fillbetwween(folder, wood_range, stone_range, coal_range, iron_range):
    import pickle
    import os
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display
    import numpy as np
    from ipywidgets import interactive, Output
    from ipywidgets.embed import embed_minimal_html

    results = np.zeros((1525, 16, 15, 17, 18, 17))
    for modelno in tqdm(range(1525)):
        with open(f"{folder}/{modelno}.pkl", "rb") as f:
            result = pickle.load(f)
        results[modelno, :] = result
    print("making sliders")
    wood_slider = widgets.IntSlider(value=wood_range[0], min=wood_range[0], max=wood_range[1]-1, step=1, description='Wood')
    stone_slider = widgets.IntSlider(value=stone_range[0], min=stone_range[0], max=stone_range[1]-1, step=1, description='Stone')
    coal_slider = widgets.IntSlider(value=coal_range[0], min=coal_range[0], max=coal_range[1]-1, step=1, description='Coal')
    iron_slider = widgets.IntSlider(value=iron_range[0], min=iron_range[0], max=iron_range[1]-1, step=1, description='Iron')

    plot_output = Output()

    def update_plot(num_wood, num_stone, num_coal, num_iron):
        with plot_output:
            plot_output.clear_output(wait=True)
            plt.figure(figsize=(10,6))
            probs = results[:, num_stone, num_wood, num_coal, num_iron, :]
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'navy', 'teal', 'lime']

            # List of actions for readability
            actions = ["NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE_STONE", "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT", "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE", "MAKE_WOOD_SWORD", "MAKE_STONE_SWORD", "MAKE_IRON_SWORD"]
            plt.stackplot(range(probs.shape[0]), *probs.T, labels=actions, colors=colors, alpha=0.8)

            plt.title(f"Fractional Probability Curves for {num_wood} wood, {num_stone} stone, {num_coal} coal, {num_iron} iron")
            plt.xlabel('Steps')
            plt.ylabel('Fraction of Probability')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True)
            plt.show()

    print("making plot")
    wood_water_interact = interactive(update_plot, num_wood=wood_slider, num_stone = stone_slider, num_coal = coal_slider, num_iron = iron_slider)
    display(widgets.VBox([wood_water_interact, plot_output]))
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
        range_stone, 
        range_coal,
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
    results = np.zeros((1599, 16, 15, 17, 18, 17))
    for modelno in tqdm(range(1599)):
        with open(f"{data_folder}/{modelno}.pkl", "rb") as f:
            result = pickle.load(f)
        results[modelno, :] = result

    range_wood = np.arange(*range_wood, 1)
    range_stone = np.arange(*range_stone, 1)
    range_coal = np.arange(*range_coal, 1)
    range_iron = np.arange(*range_iron, 1)
    num_iters = len(range_wood) * len(range_stone) * len(range_coal) * len(range_iron)

    woods = results.shape[1]
    stones = results.shape[2]
    coals = results.shape[3]
    irons = results.shape[4]
    action_size = results.shape[5]

    # results = results[:, :, :, :, :, :]
    results = results[:, :, 0, 0, 0, :]
    num_samples = results.shape[0]
    results = results.reshape(num_samples * woods * 1 * 1 * 1, action_size)
    pca = PCA(n_components=num_pcs)
    pca.fit(results)
    projected = pca.transform(results)
    projected = projected.reshape(num_samples, woods, num_pcs)
    
    # for num_wood, num_stone, num_coal, num_iron in tqdm(itertools.product(range_wood, range_stone, range_coal, range_iron), total=num_iters):
    #     projected_instance = projected[:, num_wood, num_stone, num_coal, num_iron, :]
    #     with open(f"{save_folder}/{num_wood}_{num_stone}_{num_coal}_{num_iron}.pkl", "wb") as f:
    #         pickle.dump(projected_instance, f)

    for num_wood in tqdm(range_wood):
        projected_instance = projected[:, num_wood, :]
        with open(f"{save_folder}/{num_wood}_0_0_0.pkl", "wb") as f:
            pickle.dump(projected_instance, f)

def view_experiment_pca(
        data_folder, 
        range_wood, 
        range_stone, 
        range_coal,
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
    product = list(itertools.product(np.arange(*range_wood, 1), np.arange(*range_stone, 1), np.arange(*range_coal, 1), np.arange(*range_iron, 1)))
    for num_wood, num_stone, num_coal, num_iron in product:
        with open(f"{data_folder}/{num_wood}_{num_stone}_{num_coal}_{num_iron}.pkl", "rb") as f:
            result = pickle.load(f)
        results.append(result)
    results = np.array(results)
    wood_slider = widgets.IntSlider(value=range_wood[0], min=range_wood[0], max=range_wood[1]-1, step=1, description='Wood')
    stone_slider = widgets.IntSlider(value=range_stone[0], min=range_stone[0], max=range_stone[1]-1, step=1, description='Stone')
    coal_slider = widgets.IntSlider(value=range_coal[0], min=range_coal[0], max=range_coal[1]-1, step=1, description='Coal')
    iron_slider = widgets.IntSlider(value=range_iron[0], min=range_iron[0], max=range_iron[1]-1, step=1, description='Iron')

    plot_output = Output()

    def update_plot(num_wood, num_stone, num_coal, num_iron):
        with plot_output:
            plot_output.clear_output(wait=True)
            fig, ax = plt.subplots(len(list(itertools.combinations(range(num_pcs), 2))), 1, figsize=(10, 6*len(list(itertools.combinations(range(num_pcs), 2)))))
            result_num = product.index((num_wood, num_stone, num_coal, num_iron))
            result = results[result_num, ...]
            for i, (pcx, pcy) in enumerate(itertools.combinations(range(num_pcs), 2)):
                sc = ax[i].scatter(result[:, pcx], result[:, pcy], label=f"PC{pcx} vs PC{pcy}", s=1, c=np.arange(1599), cmap="rainbow")
                ax[i].set_xlabel(f"PC{pcx}")
                ax[i].set_ylabel(f"PC{pcy}")
                # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax[i].grid(True)
            fig.suptitle(f"PCA for {num_wood} Wood, {num_stone} Stone, {num_coal} coal, {num_iron} Iron")
            cbar = plt.colorbar(sc, ax=ax[-1])
            plt.show()
    wood_water_interact = interactive(update_plot, num_wood=wood_slider, num_stone=stone_slider, num_coal = coal_slider, num_iron=iron_slider)
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
        range_stone, 
        range_coal,
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
    coal_slider = widgets.IntSlider(value=range_coal[0], min=range_coal[0], max=range_coal[1]-1, step=1, description='Coal')
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

    def update_plot(num_wood, num_stone, num_coal, num_iron):
        # Clear all plots first
        fig.data = []

        # Re-add plots with new data
        result_num = product.index((num_wood, num_stone, num_coal, num_iron))
        result = results[result_num, ...]
        for i, (pcx, pcy) in enumerate(pc_combinations, start=1):
            fig.add_trace(go.Scattergl(x=result[:, pcx], y=result[:, pcy], mode='markers',
                                       marker=dict(size=5, color=np.arange(result.shape[0]), colorscale='Rainbow'),
                                       name=f"PC{pcx} vs PC{pcy}"),
                          row=i, col=1)
        fig.update_layout(height=300 * num_combinations, title_text=f"PCA for {num_wood} Wood, {num_pickaxe} Pickaxe, {num_sword} Sword, {num_stone} Stone, {num_iron} Iron")
        fig.show()

    interact_widget = widgets.interactive(update_plot, num_wood=wood_slider, num_stone=stone_slider, num_coal = coal_slider, num_iron=iron_slider)
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
    from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv

    import os
    import pickle
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display
    import numpy as np
    env = CraftaxClassicSymbolicEnv()
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

DIRECTIONS = jnp.concatenate(
    (
        jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32),
        jnp.zeros((11, 2), dtype=jnp.int32),
    ),
    axis=0,
)

def add_wood(state):
    map = state.map
    player_direction = state.player_direction
    player_position = state.player_position
    block_position = player_position + DIRECTIONS[player_direction]

    map = map.at[block_position[0], block_position[1]].set(constants.BlockType.TREE.value)
    return EnvState(
        # world
        map=map,

        # player
        player_position=state.player_position,
        player_direction=state.player_direction,
        player_health=state.player_health, 
        player_food=state.player_food,
        player_drink=state.player_drink,
        player_energy=state.player_energy,
        player_recover=state.player_recover,
        player_hunger=state.player_hunger,
        player_thirst=state.player_thirst,
        player_fatigue=state.player_fatigue,
        is_sleeping=state.is_sleeping,

        # inventory
        inventory=state.inventory,
    
        # mobs
        mob_map = state.mob_map,
        zombies=state.zombies,
        cows=state.cows,
        skeletons=state.skeletons,
        arrows=state.arrows,
        arrow_directions=state.arrow_directions,
        
        # farming
        growing_plants_positions=state.growing_plants_positions,
        growing_plants_age=state.growing_plants_age,
        growing_plants_mask=state.growing_plants_mask,

        # progress
        achievements=state.achievements,
        light_level=state.light_level,

        # misc
        state_rng=state.state_rng,
        timestep=state.timestep
    )

@jax.jit
def add_stone(state):
    map = state.map
    player_direction = state.player_direction
    player_position = state.player_position
    block_position = player_position + DIRECTIONS[player_direction]

    map = map.at[block_position[0], block_position[1]].set(constants.BlockType.STONE.value)
    return EnvState(
        # world
        map=map,

        # player
        player_position=state.player_position,
        player_direction=state.player_direction,
        player_health=state.player_health, 
        player_food=state.player_food,
        player_drink=state.player_drink,
        player_energy=state.player_energy,
        player_recover=state.player_recover,
        player_hunger=state.player_hunger,
        player_thirst=state.player_thirst,
        player_fatigue=state.player_fatigue,
        is_sleeping=state.is_sleeping,

        # inventory
        inventory=state.inventory,
    
        # mobs
        mob_map = state.mob_map,
        zombies=state.zombies,
        cows=state.cows,
        skeletons=state.skeletons,
        arrows=state.arrows,
        arrow_directions=state.arrow_directions,
        
        # farming
        growing_plants_positions=state.growing_plants_positions,
        growing_plants_age=state.growing_plants_age,
        growing_plants_mask=state.growing_plants_mask,

        # progress
        achievements=state.achievements,
        light_level=state.light_level,

        # misc
        state_rng=state.state_rng,
        timestep=state.timestep
    )

@jax.jit
def randomize_inventory(state, rng):
    inventory_rngs = jax.random.split(rng, 13)
    rng = inventory_rngs[-1]
    inventory = Inventory(
        wood=jax.random.randint(inventory_rngs[0], minval=0, maxval=10, shape=()),
        stone=jax.random.randint(inventory_rngs[1], minval=0, maxval=10, shape=()),
        coal=jax.random.randint(inventory_rngs[2], minval=0, maxval=10, shape=()),
        iron=jax.random.randint(inventory_rngs[3], minval=0, maxval=10, shape=()),
        diamond=jax.random.randint(inventory_rngs[4], minval=0, maxval=3, shape=()),
        sapling=jax.random.randint(inventory_rngs[5], minval=0, maxval=10, shape=()),
        wood_pickaxe=jax.random.randint(inventory_rngs[6], minval=0, maxval=2, shape=()),
        stone_pickaxe=jax.random.randint(inventory_rngs[7], minval=0, maxval=2, shape=()),
        iron_pickaxe=jax.random.randint(inventory_rngs[8], minval=0, maxval=2, shape=()),
        wood_sword=jax.random.randint(inventory_rngs[9], minval=0, maxval=2, shape=()),
        stone_sword=jax.random.randint(inventory_rngs[10], minval=0, maxval=2, shape=()),
        iron_sword=jax.random.randint(inventory_rngs[11], minval=0, maxval=2, shape=()),
    )

    return EnvState(
        # world
        map=state.map,

        # player
        player_position=state.player_position,
        player_direction=state.player_direction,
        player_health=state.player_health, 
        player_food=state.player_food,
        player_drink=state.player_drink,
        player_energy=state.player_energy,
        player_recover=state.player_recover,
        player_hunger=state.player_hunger,
        player_thirst=state.player_thirst,
        player_fatigue=state.player_fatigue,
        is_sleeping=state.is_sleeping,

        # inventory
        inventory=inventory,
    
        # mobs
        mob_map = state.mob_map,
        zombies=state.zombies,
        cows=state.cows,
        skeletons=state.skeletons,
        arrows=state.arrows,
        arrow_directions=state.arrow_directions,
        
        # farming
        growing_plants_positions=state.growing_plants_positions,
        growing_plants_age=state.growing_plants_age,
        growing_plants_mask=state.growing_plants_mask,

        # progress
        achievements=state.achievements,
        light_level=state.light_level,

        # misc
        state_rng=state.state_rng,
        timestep=state.timestep
    )

from typing import Sequence
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax

class ActorCritic_with_hook(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        activations = list()
        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_mean = activation(actor_mean)
        activations.append(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)
        activations.append(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)
        activations.append(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        activations.append(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), activations


def get_activations(obs, params):
    network = ActorCritic_with_hook(17, 512)
    _, _, activations = network.apply(params, obs)
    return activations

#%%
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
import os

env = CraftaxClassicSymbolicEnv()
rng = jax.random.PRNGKey(0)
def generate_states(carry, unused):
    rng, i = carry
    rng, env_rng = jax.random.split(rng)
    obs, state = env.reset(env_rng)
    state = randomize_inventory(state, rng)
    stone_state = add_stone(state)
    wood_state = add_wood(state)

    stone_obs = render_craftax_symbolic(stone_state)
    wood_obs = render_craftax_symbolic(wood_state)
    return (rng, i+1), (stone_obs, wood_obs)
_, (stone_states, wood_states) = jax.lax.scan(generate_states, (rng, 0), None, length=525)

vectorized_acts = jax.jit(jax.vmap(get_activations, in_axes=(0, None), out_axes=0))

checkpointer = ocp.StandardCheckpointer()
checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{1524}"
folder_list = os.listdir(checkpoint_directory)
params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
stone_activations = vectorized_acts(stone_states, params)
wood_activations = vectorized_acts(wood_states, params)
os.makedirs("/workspace/CraftaxDevinterp/intermediate_data/modelno_1524/stone", exist_ok=True)
os.makedirs("/workspace/CraftaxDevinterp/intermediate_data/modelno_1524/wood", exist_ok=True)
os.makedirs("/workspace/CraftaxDevinterp/intermediate_data/modelno_1524/stone-wood", exist_ok=True)
for i, stone_activation in enumerate(stone_activations):
    wood_activation = wood_activations[i]
    u, s, v = jnp.linalg.svd(stone_activation - wood_activation, full_matrices=False)
    plt.plot(s)
    plt.title(f"Singular Values of Activations for layer {i}")
    plt.savefig(f"/workspace/CraftaxDevinterp/intermediate_data/modelno_1524/stone-wood/singular_values_{i}.png")
    plt.close()


for i, stone_activation in enumerate(stone_activations):
    wood_activation = wood_activations[i]
    plt.imshow(stone_activation - wood_activation, cmap="viridis")
    plt.title(f"Activations for layer {i}")
    plt.colorbar()
    plt.savefig(f"/workspace/CraftaxDevinterp/intermediate_data/modelno_1524/stone-wood/activations_{i}.png")
    plt.close()
    print(f"Norm of difference for layer {i}: {jnp.linalg.norm(stone_activation - wood_activation)}")
    print(f"NOrm of wood activations for layer {i}: {jnp.linalg.norm(wood_activation)}")
    print(f"Norm of stone activations for layer {i}: {jnp.linalg.norm(stone_activation)}")

# %%
# conclusion: it is indifferent between whether there's a wood or stone in front of it.
# It immediately recognizes that both imply it should execute the DO action, and so it has
# the same representation for both situations.