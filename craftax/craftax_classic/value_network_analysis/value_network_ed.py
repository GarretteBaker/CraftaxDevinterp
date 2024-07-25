import jax
import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA
from craftax.craftax_classic.make_model_trajectory import jit_gen_traj
import orbax.checkpoint as ocp
import os
import einops
from craftax.models.actor_critic import ActorCritic
import time
from tqdm import tqdm
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from craftax.environment_base.wrappers import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
)
import matplotlib.pyplot as plt

rng = jax.random.PRNGKey(0)
# load end model
env = CraftaxClassicSymbolicEnv()
env = AutoResetEnvWrapper(env)
env = BatchEnvWrapper(env, 64)
env_params = env.default_params
rng, env_rng = jax.random.split(rng)
obsv, env_state = env.reset(env_rng, env_params)

network = ActorCritic(17, 512, activation="relu")
rng, init_rng = jax.random.split(rng)
network_structure = network.init(init_rng, obsv)
checkpointer = ocp.StandardCheckpointer()
def load_model(modelno):
    checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{modelno}"
    folder_list = os.listdir(checkpoint_directory)
    params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}", item=network_structure)
    return params
params = load_model(1524)

# generate observation dataset
num_envs = 64
num_steps = 1e2
rng, trajectory_rng = jax.random.split(rng)
(_, _, obses), dones = jit_gen_traj(params, trajectory_rng, num_envs, num_steps, log_obses=True)
obses = einops.rearrange(obses, 's e o -> (s e) o')

# loop through each model

    # find that model's value network estimation on each obs
    # collect such vectors
max_models = 1525
A = np.zeros((max_models, obses.shape[0]))
run_network = jax.jit(network.apply)
for modelno in tqdm(range(max_models)):
    params = load_model(modelno)
    p, a = run_network(params, obses)
    A[modelno, :] = jax.device_get(a)

# pca vector collection
pca = PCA(n_components=5)
pca.fit(A)
E = pca.transform(A)

def plot_all_pca_combinations(E, save_dir='pca_plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_components = E.shape[1]
    n_plots = n_components * (n_components - 1) // 2
    
    # Calculate the number of rows and columns for subplots
    n_rows = int(np.ceil(np.sqrt(n_plots)))
    n_cols = int(np.ceil(n_plots / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    fig.suptitle("All PCA Component Combinations", fontsize=16)
    
    plot_idx = 0
    for i in range(n_components):
        for j in range(i+1, n_components):
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            scatter = ax.scatter(E[:, i], E[:, j], c=range(E.shape[0]), cmap='viridis', s=50)
            ax.set_xlabel(f'PCA {i+1}')
            ax.set_ylabel(f'PCA {j+1}')
            ax.set_title(f'PCA {i+1} vs PCA {j+1}')
            
            # Add model number labels to each point
            for k, (x, y) in enumerate(zip(E[:, i], E[:, j])):
                ax.annotate(str(k), (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8)
            
            plot_idx += 1
    
    # Remove any unused subplots
    for i in range(plot_idx, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col] if n_rows > 1 else axes[col])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Add a colorbar
    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), label='Model Number', aspect=40)
    cbar.ax.tick_params(labelsize=10)
    
    plt.savefig(os.path.join(save_dir, 'all_pca_combinations.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("PCA visualization complete. Plot saved as 'all_pca_combinations.png' in the 'pca_plots' directory.")

# view pca
# create plot for all combinations
# loop over all combinations of pcas, and view them plotted against each other
# save
plot_all_pca_combinations(E, save_dir="/workspace/CraftaxDevinterp/craftax/craftax_classic/value_network_analysis")
