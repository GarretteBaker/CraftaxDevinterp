# Much here taken from https://github.com/edmundlth/validating_lambdahat/blob/a923472086d326ff45ec678cacf7f2364e8257d5/expt_dln_saddle_dynamics.py

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
import os
from sgld_utils import (
    SGLDConfig, 
    run_sgld
)
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

layer_size = 512
seed = 0
rng = jax.random.PRNGKey(seed)
rng, _rng = jax.random.split(rng)

#%%
def generate_trajectory(network_params, rng, num_envs=1, num_steps=495):
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
        obs: jnp.ndarray
        action_logits: jnp.ndarray

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
            obs = obsv,
            action_logits = pi.logits
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
    return traj_batch.obs, traj_batch.action_logits

jit_gen_traj = jax.jit(generate_trajectory)
#%%
model_no = 1524
checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{model_no}"
checkpointer = ocp.StandardCheckpointer()
folder_list = os.listdir(checkpoint_directory)
network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
expert_obses, expert_logitses = jit_gen_traj(network_params, _rng)

#%%
def mse_loss(param, model, inputs, targets):
    predictions, _ = model.apply(param, inputs)
    predictions = predictions.logits
    return jnp.mean((predictions - targets) ** 2)

env = CraftaxSymbolicEnv()
env_params = env.default_params
env = LogWrapper(env)
env = OptimisticResetVecEnvWrapper(
    env,
    num_envs=1,
    reset_ratio=min(16, 1),
)
network = ActorCritic(env.action_space(env_params).n, layer_size)
loss_fn = jax.jit(
    lambda param, inputs, targets: mse_loss(param, network, inputs, targets)
)

model_no = 100
checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{model_no}"
checkpointer = ocp.StandardCheckpointer()
folder_list = os.listdir(checkpoint_directory)
network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")

eps_lower_pow = 4
eps_upper_pow = 7
num_eps = int(eps_upper_pow - eps_lower_pow) + 1

gam_lower_pow = 0
gam_upper_pow = 4
num_gam = int(gam_upper_pow - gam_lower_pow + 1)

fig, axs = plt.subplots(num_eps, num_gam, figsize=(16*num_eps, 16*num_gam))
for i, epsilon in tqdm(enumerate(np.logspace(eps_lower_pow, eps_upper_pow, num=num_eps, base=10)), desc="Epsilon"):
    for j, gamma in tqdm(enumerate(np.logspace(gam_lower_pow, gam_upper_pow, num=num_gam, base=10)), desc="Gamma"):
        sgld_config = SGLDConfig(
            epsilon = epsilon, 
            gamma = gamma, 
            num_steps = 1000, 
            num_chains = 1,
            batch_size = 64)

        num_models = 1525

        num_training_data = len(expert_obses)
        itemp = 1/np.log(num_training_data)

        loss_trace, distances, acceptance_probs = run_sgld(
            _rng, 
            loss_fn, 
            sgld_config, 
            network_params, 
            expert_obses, 
            expert_logitses, 
            itemp = itemp, 
            trace_batch_loss = True, 
            compute_distance = False, 
            verbose = False
        )

        init_loss = loss_fn(network_params, expert_obses, expert_logitses)
        lambdahat = float(np.mean(loss_trace) - init_loss) * num_training_data * itemp
        axs[i, j].plot(loss_trace)
        axs[i, j].set_title(f"epsilon {epsilon}, gamma {gamma}, lambda {lambdahat}")
plt.tight_layout("/workspace/CraftaxDevinterp/llc_estimation/debug/calibration.png")
plt.savefig()
#%%

# os.makedirs("/workspace/CraftaxDevinterp/llc_estimation/debug/trace_curves", exist_ok = True)
# os.makedirs("/workspace/CraftaxDevinterp/llc_estimation/debug/lambdahats", exist_ok=True)
# for model_no in tqdm(range(0, num_models, 300)):
#     checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{model_no}"
#     checkpointer = ocp.StandardCheckpointer()
#     folder_list = os.listdir(checkpoint_directory)
#     network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")

#     num_training_data = len(expert_obses)
#     itemp = 1/np.log(num_training_data)

#     loss_trace, distances, acceptance_probs = run_sgld(
#         _rng, 
#         loss_fn, 
#         sgld_config, 
#         network_params, 
#         expert_obses, 
#         expert_logitses, 
#         itemp = itemp, 
#         trace_batch_loss = True, 
#         compute_distance = False, 
#         verbose = False
#     )

#     init_loss = loss_fn(network_params, expert_obses, expert_logitses)
#     lambdahat = float(np.mean(loss_trace) - init_loss) * num_training_data * itemp
#     with open(f"/workspace/CraftaxDevinterp/llc_estimation/debug/lambdahats/{model_no}.pkl", "wb") as f:
#         pickle.dump(lambdahat, f)
#     # plt.plot(loss_trace)
#     # plt.savefig(f"/workspace/CraftaxDevinterp/llc_estimation/trace_curves/{model_no}.png")
#     # plt.close()

# # %%
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# from tqdm import tqdm
# num_models = 1525

# print("loading lambdahats")
# lambdahats = list()
# for modelno in tqdm(range(0, num_models, 300)):
#     with open(f"/workspace/CraftaxDevinterp/llc_estimation/debug/lambdahats/{modelno}.pkl", "rb") as f:
#         lambdahat = pickle.load(f)
#     lambdahats.append(lambdahat)
# plt.plot(lambdahats)
# plt.show()
# # %%
