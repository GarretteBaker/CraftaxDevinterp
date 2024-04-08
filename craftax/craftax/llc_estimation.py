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
    run_sgld, 
    run_sgld_with_scan
)
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import torch.utils.data as data
import optax
from flax.training import train_state

layer_size = 512
seed = 0
rng = jax.random.PRNGKey(seed)
rng, rng_train, rng_sgld = jax.random.split(rng, num=3)

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

jit_gen_traj_100 = jax.jit(lambda params, rng: generate_trajectory(params, rng, num_steps=128))
jit_gen_traj_495 = jax.jit(lambda params, rng: generate_trajectory(params, rng, num_steps=495))
#%%
model_no = 1524
checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{model_no}"
checkpointer = ocp.StandardCheckpointer()
folder_list = os.listdir(checkpoint_directory)
network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
# expert_obses_train, expert_logitses_train = jit_gen_traj_100(network_params, rng_train)
expert_obses, expert_logitses = jit_gen_traj_495(network_params, rng_sgld)

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

class ObsActDataset(data.Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        observation = self.x[idx, ...]
        action = self.y[idx, ...]
        return observation, action

teacher_dataset = ObsActDataset(expert_obses, expert_logitses)

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

# teacher_dataloader = data.DataLoader(teacher_dataset, batch_size = 128, shuffle=True, collate_fn=numpy_collate, drop_last=True)

# optimizer = optax.sgd(learning_rate=1e-4)

# model_state = train_state.TrainState.create(apply_fn=network.apply,
#                                             params=network_params,
#                                             tx=optimizer)

@jax.jit
def train_step(state, batch):
    obs, target = batch
    grad_fn = jax.value_and_grad(
        loss_fn, 
        argnums=0, 
        has_aux=False
    )
    loss, grads = grad_fn(state.params, obs, target)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_model(state, batch):
    loss = loss_fn(state.params, batch)
    return loss

def train_model(state, dataloader, num_epochs=1):
    losses = list()
    for epochs in tqdm(range(num_epochs)):
        epoch_loss = list()
        for batch in dataloader:
            state, loss = train_step(state, batch)
            epoch_loss.append(loss)
        losses.append(jnp.mean(jnp.array(epoch_loss)))
    return state, losses

# teacher_array_obs = list()
# teacher_array_acts = list()
# for batch in teacher_dataloader:
#     obs, act = batch
#     teacher_array_obs.append(obs)
#     teacher_array_acts.append(act)
# teacher_array_obs = np.array(teacher_array_obs)
# teacher_array_acts = np.array(teacher_array_acts)


# #%%
# batch_size = 128
# num_epochs=1e3
# @jax.jit
# def jit_train_model(state):
#     def train_epoch(state, unused):
#         i = 0
#         def train_batch(carry, unused):
#             i, state = carry
#             obs = jax.lax.dynamic_slice(expert_obses, (i * batch_size, 0, 0), (batch_size, expert_obses.shape[1], expert_obses.shape[2]))
#             act = jax.lax.dynamic_slice(expert_logitses, (i * batch_size, 0, 0), (batch_size, expert_logitses.shape[1], expert_logitses.shape[2]))
#             batch = obs, act
#             state, loss = train_step(state, batch)
#             return (i+1, state), loss
#         (i, state), loss = jax.lax.scan(train_batch, (i, state), jnp.arange(expert_obses.shape[0]//batch_size))
#         return state, jnp.mean(loss)
#     state, losses = jax.lax.scan(train_epoch, state, None, length=num_epochs)
#     return state, losses

# trained_state, losses = jit_train_model(model_state)

# plt.plot(losses.flatten())
# plt.show()
# trained_params = trained_state.params
#%%
# beta_lower_pow = 0
# beta_upper_pow = -7
# num_beta = int(abs(beta_upper_pow - beta_lower_pow)) + 1

# eps_lower_pow = -4
# eps_upper_pow = -11
# num_eps = int(abs(eps_lower_pow - eps_upper_pow)) + 1

# gam_lower_pow = -3
# gam_upper_pow = 4
# num_gam = int(gam_upper_pow - gam_lower_pow) + 1

# os.makedirs("/workspace/CraftaxDevinterp/llc_estimation/debug", exist_ok=True)
# fig, axs = plt.subplots(num_beta, num_gam, figsize=(16*num_beta, 16*num_gam))
# for k, epsilon in tqdm(enumerate(np.logspace(eps_lower_pow, eps_upper_pow, num=num_eps, base=10)), desc="Epsilon", total=num_eps):
#     for i, beta in tqdm(enumerate(np.logspace(beta_lower_pow, beta_upper_pow, num=num_beta, base=10)), desc="Beta", total=num_beta):
#         for j, gamma in tqdm(enumerate(np.logspace(gam_lower_pow, gam_upper_pow, num=num_gam, base=10)), desc="Gamma", total=num_gam):
#             sgld_config = SGLDConfig(
#                 epsilon = epsilon, 
#                 gamma = gamma, 
#                 num_steps = 10000, 
#                 num_chains = 1,
#                 batch_size = 64)

#             num_training_data = len(expert_obses)
#             itemp = beta

#             loss_trace, _, acceptance_probs = run_sgld(
#                 rng_sgld, 
#                 loss_fn, 
#                 sgld_config, 
#                 network_params, 
#                 expert_obses, 
#                 expert_logitses, 
#                 itemp = itemp, 
#                 trace_batch_loss = True, 
#                 compute_distance = False, 
#                 verbose = False
#             )

#             init_loss = loss_fn(network_params, expert_obses, expert_logitses)
#             lambdahat = float(np.mean(loss_trace) - init_loss) * num_training_data * itemp
#             axs[i, j].plot(loss_trace)
#             axs[i, j].axhline(y=init_loss, linestyle=':')
#             axs[i, j].set_title(f"epsilon {epsilon}, beta {beta}, gamma {gamma}, lambda {lambdahat}, mala {np.mean(np.array(acceptance_probs)[:, 1])}")
#     plt.tight_layout()
#     plt.savefig(f"/workspace/CraftaxDevinterp/llc_estimation/debug/calibration_eps_{epsilon}.png")
#%%
# import time
# t0 = time.time()
# sgld_config = SGLDConfig(
#     epsilon = 1e-5, 
#     gamma = 100, 
#     num_steps = 10000, 
#     num_chains = 1,
#     batch_size = 64)

# num_training_data = len(expert_obses)
# itemp = 1/np.log(num_training_data)

# loss_trace = run_sgld_with_scan(
#     rng_sgld, 
#     loss_fn, 
#     sgld_config, 
#     trained_params, 
#     expert_obses, 
#     expert_logitses, 
#     itemp = itemp, 
#     trace_batch_loss = True, 
#     compute_distance = False, 
#     verbose = False
# )
# init_loss = loss_fn(trained_params, expert_obses, expert_logitses)
# lambdahat = float(np.mean(loss_trace) - init_loss) * num_training_data * itemp
# print(f"Time to run scanned sgld: {time.time() - t0}")
# plt.plot(loss_trace)
# plt.show()

#%%
# import time
# for i in tqdm(range(5)):
#     sgld_config = SGLDConfig(
#         epsilon = 1e-4, 
#         gamma = 10.0, 
#         num_steps = 3000, 
#         num_chains = 1,
#         batch_size = 64)

#     num_training_data = len(expert_obses)
#     itemp = 0.01

#     loss_trace, distances, acceptance_probs = run_sgld(
#         rng_sgld, 
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
#     plt.plot(loss_trace)
# plt.show()
# #%%

sgld_config = SGLDConfig(
    epsilon = 1e-4, 
    gamma = 10, 
    num_steps = 3000, 
    num_chains = 1,
    batch_size = 64)

num_models = 1525
os.makedirs("/workspace/CraftaxDevinterp/llc_estimation/debug/trace_curves", exist_ok = True)
os.makedirs("/workspace/CraftaxDevinterp/llc_estimation/debug/lambdahats", exist_ok=True)
for model_no in tqdm(range(0, num_models, 300)):
    checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{model_no}"
    checkpointer = ocp.StandardCheckpointer()
    folder_list = os.listdir(checkpoint_directory)
    network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")

    num_training_data = len(expert_obses)
    itemp = 0.01

    loss_trace, distances, acceptance_probs = run_sgld(
        rng_sgld, 
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
    with open(f"/workspace/CraftaxDevinterp/llc_estimation/debug/lambdahats/{model_no}.pkl", "wb") as f:
        pickle.dump(lambdahat, f)
    plt.plot(loss_trace)
    plt.axhline(y=init_loss, linestyle=':')
    plt.title(f"lambda {lambdahat}, mala {np.mean(np.array(acceptance_probs)[:, 1])}")
    plt.savefig(f"/workspace/CraftaxDevinterp/llc_estimation/debug/trace_curves/{model_no}.png")
    plt.close()

# # %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
num_models = 1525

print("loading lambdahats")
lambdahats = list()
for modelno in tqdm(range(0, num_models, 300)):
    with open(f"/workspace/CraftaxDevinterp/llc_estimation/debug/lambdahats/{modelno}.pkl", "rb") as f:
        lambdahat = pickle.load(f)
    lambdahats.append(lambdahat)

plt.plot(lambdahats)
plt.show()
# # %%
