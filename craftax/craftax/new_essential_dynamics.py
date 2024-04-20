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
num_steps = 2e3
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

        tracker = obsv

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


checkpointer = ocp.StandardCheckpointer()
checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{1524}"
folder_list = os.listdir(checkpoint_directory)
network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
seed = 0
rng = jax.random.PRNGKey(seed)
rng, _rng = jax.random.split(rng)
trajectory, done = jit_gen_traj(network_params, _rng)
print("Success!")
#%%


#%%
save_dir = "/workspace/CraftaxDevinterp/essential_dynamics"
os.makedirs(save_dir, exist_ok=True)
network = ActorCritic(43, 512)
for modelno in tqdm(range(0, 1525)):
    os.makedirs(f"{save_dir}/{modelno}", exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{modelno}"
    folder_list = os.listdir(checkpoint_directory)
    network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")

    p, _ = network.apply(network_params, trajectory)
    logits = p.logits

    with open(f"{save_dir}/{modelno}/logits.pkl", "wb") as f:
        pickle.dump(logits, f)

#%%
from tqdm import tqdm
import numpy as np
import pickle
save_dir = "/workspace/CraftaxDevinterp/essential_dynamics"
logits_matrix = np.zeros((1525//10 + 1, 2000, 64, 43))
for modelno in tqdm(range(0, 1525, 10)):
    with open(f"{save_dir}/{modelno}/logits.pkl", "rb") as f:
        logits = pickle.load(f)
    logits_matrix[modelno//10, :, :, :] = logits

logits_matrix = logits_matrix.reshape(1525//10+1, 2000*64*43)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(logits_matrix)
projected = pca.transform(logits_matrix)

import itertools
import matplotlib.pyplot as plt
pc_combos = itertools.combinations(range(3), 2)
for pc1, pc2 in pc_combos:
    plt.scatter(projected[:, pc1], projected[:, pc2])
    plt.xlabel(f"PC{pc1}")
    plt.ylabel(f"PC{pc2}")
    plt.title(f"PC{pc1} vs PC{pc2}")
    plt.show()
# %%
from tqdm import tqdm
import numpy as np
import pickle
import plotly.express as px
from sklearn.decomposition import PCA
import itertools
skip = 6
# Load and process data
save_dir = "/workspace/CraftaxDevinterp/essential_dynamics"
logits_matrix = np.zeros((1525 // skip + 1, 2000, 64, 43))
for modelno in tqdm(range(0, 1525, skip)):
    with open(f"{save_dir}/{modelno}/logits.pkl", "rb") as f:
        logits = pickle.load(f)
    logits_matrix[modelno // skip, :, :, :] = logits

# Reshape for PCA
logits_matrix = logits_matrix.reshape(1525 // skip + 1, 2000 * 64 * 43)

# Perform PCA
pca = PCA(n_components=3)
projected = pca.fit_transform(logits_matrix)

# Create interactive plots
timestamps = range(0, 1525, skip)
for pc1, pc2 in itertools.combinations(range(3), 2):
    fig = px.scatter(
        x=projected[:, pc1], 
        y=projected[:, pc2], 
        labels={
            'x': f'PC{pc1}',
            'y': f'PC{pc2}'
        },
        title=f'PC{pc1} vs PC{pc2}',
        hover_data=[timestamps]
    )
    fig.update_traces(marker=dict(size=5))
    fig.show()
#%%