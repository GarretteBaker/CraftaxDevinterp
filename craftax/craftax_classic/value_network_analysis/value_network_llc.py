#%%
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

# TODO: Generate "true" (end-state-policy) a's

# loop through each model

    # find that model's value network estimation on each obs
    # collect such vectors
max_models = 1525
llcs = np.zeros((max_models))
run_network = jax.jit(network.apply)
for modelno in tqdm(range(max_models)):
    params = load_model(modelno)
    p, a = run_network(params, obses)
    # TODO: calculate llc
    # TODO: Add llc to llcs

# TODO: visualize llcs