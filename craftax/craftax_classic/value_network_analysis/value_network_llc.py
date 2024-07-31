#%%
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA
from craftax.craftax_classic.make_model_trajectory import jit_gen_traj
import orbax.checkpoint as ocp
import os
import einops
from craftax.models.actor_critic import ActorCritic, LinearActorCritic
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
# network = LinearActorCritic(17, 512)
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

# loop through each model

    # find that model's value network estimation on each obs
    # collect such vectors
from craftax.craftax.sgld_utils import run_sgld, SGLDConfig

def mse_loss(param, inputs, targets):
    _, predictions = run_network(param, inputs)
    return jnp.mean((predictions - targets) ** 2)

loss_fn = jax.jit(
    lambda param, inputs, targets: mse_loss(param, inputs, targets)
)
num_training_data = obses.shape[0]

rng, sgld_loop_rng = jax.random.split(rng)
itemp = 0.001
sgld_config = SGLDConfig(
    epsilon = 1e-5, 
    gamma = 10, 
    num_steps = 1e4, 
    num_chains = 1,
    batch_size = 1024
)

min_models = 300
max_models = 400
count_by = 1
llcs = np.zeros(((max_models-min_models)//count_by + 1))
pbar = tqdm(total=(max_models-min_models)//count_by + 1, desc="Model llc progress")

load_model(max_models-1)

def find_lambdahat(rng, params):
    loss_trace, _, _ = run_sgld(
        rng, 
        loss_fn, 
        sgld_config, 
        params, 
        obses, 
        true_a, 
        itemp=itemp
    )
    init_loss = loss_fn(params, obses, true_a)
    lambdahat = float(np.mean(loss_trace) - init_loss) * num_training_data * itemp
    return lambdahat

for i, modelno in tqdm(enumerate(range(min_models, max_models, count_by))):
    params = load_model(modelno)
    lambdahat = 0
    rng = sgld_loop_rng
    for j in range(sgld_config.num_chains):
        rng, sgld_rng = jax.random.split(rng)
        lambdahat += find_lambdahat(sgld_rng, params)
    lambdahat = lambdahat / sgld_config.num_chains
    llcs[i] = lambdahat
    pbar.update(1)

folder = f"/workspace/CraftaxDevinterp/craftax/craftax_classic/value_network_analysis/temp_{itemp}/eps_{sgld_config.epsilon}/gamma_{sgld_config.gamma}/num_steps_{sgld_config.num_steps}/num_chains_{sgld_config.num_chains}/batch_{sgld_config.batch_size}/min_models_{min_models}/max_models_{max_models}/countby_{count_by}"
os.makedirs(folder, exist_ok=True)

# visualize llcs
plt.plot(llcs[:-1])
plt.savefig(f"{folder}/llcs_over_time_countby.png")
plt.close()

np.save(f"{folder}/llcs_over_time_countby", llcs)

#%%
import numpy as np
import matplotlib.pyplot as plt

count_by = 10
llcs = np.load(
    "/workspace/CraftaxDevinterp/craftax/craftax_classic/value_network_analysis/temp_0.001/eps_1e-05/gamma_10/num_steps_10000.0/num_chains_10/batch_1024/min_models_300/max_models_400/countby_1/llcs_over_time_countby.npy"
)

plt.plot(llcs[:-1])
plt.show()
# plt.savefig(f"/workspace/CraftaxDevinterp/craftax/craftax_classic/value_network_analysis/llcs_over_time_countby_{count_by}.png")
# plt.close()

# %%
