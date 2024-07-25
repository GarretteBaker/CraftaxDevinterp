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
run_network = jax.jit(network.apply)

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

# Generate "true" (end-state-policy) a's
_, true_a = run_network(params, obses)

from craftax.craftax.sgld_utils import run_sgld, SGLDConfig

def mse_loss(param, inputs, targets):
    _, predictions = run_network(param, inputs)
    return jnp.mean((predictions - targets) ** 2)

loss_fn = jax.jit(
    lambda param, inputs, targets: mse_loss(param, inputs, targets)
)

sgld_config = SGLDConfig(
    epsilon = 1e-4, 
    gamma = 1e2, 
    num_steps = 1e2, 
    num_chains = 1,
    batch_size = 64
)
itemp = 0.01
num_training_data = obses.shape[0]
max_models = 1525
count_by = 100
rng, sgld_rng = jax.random.split(rng)
llcs = np.zeros((max_models//count_by + 1))
for i, modelno in tqdm(enumerate(range(0, max_models, count_by))):
    params = load_model(modelno)
    loss_trace, _, _ = run_sgld(
        sgld_rng, 
        loss_fn, 
        sgld_config, 
        params, 
        obses, 
        true_a, 
        itemp=itemp
    )
    init_loss = loss_fn(params, obses, true_a)
    lambdahat = float(np.mean(loss_trace) - init_loss) * num_training_data * itemp

    llcs[i] = lambdahat
plt.plot(llcs)
# %%
