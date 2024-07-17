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
from craftax.craftax_classic.game_logic import get_player_attack_damage
import orbax.checkpoint as ocp
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from craftax.craftax_classic.envs.craftax_state import StaticEnvParams
import pickle
import os
from tqdm import tqdm
from craftax.craftax_classic.constants import *
import matplotlib.pyplot as plt
import pickle
import scipy as sp

#%%
def generate_trajectory(network_params, rng, num_envs, num_steps, log_obses = False):
    env = CraftaxClassicSymbolicEnv()
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs)
    env_params = env.default_params
    network = ActorCritic(env.action_space(env_params).n, 512, activation="relu")

    class Transition(NamedTuple):
        logits: jnp.ndarray
        probs: jnp.ndarray
        done: jnp.ndarray
        obs: jnp.ndarray = None

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
        logits = pi.logits
        probs = pi.probs
        action = pi.sample(seed=_rng)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state, _, done, _ = env.step(
            _rng, past_state, action, env_params
        )

        transition = Transition(
            logits = logits,
            probs = probs,
            done = done,
            obs = last_obs if log_obses else None
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
    return (traj_batch.logits, traj_batch.probs, traj_batch.obs), traj_batch.done
jit_gen_traj = jax.jit(generate_trajectory, static_argnames=("num_envs", "num_steps", "log_obses"))
# jit_gen_traj = jax.jit(generate_trajectory)
#%%
# # step 0: getting a sense of time
# rng = jax.random.PRNGKey(0)
# num_envs = 64
# num_steps = 1e2
# checkpointer = ocp.StandardCheckpointer()
# checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{1524}"
# folder_list = os.listdir(checkpoint_directory)
# params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")

# import time
# t0 = time.time()
# (logits, probs, _), _ = jit_gen_traj(params, rng, num_envs, num_steps)
# t1 = time.time()
# print(f"Time taken with {num_steps} and {num_envs} is {t1-t0}")

# num_steps = 1e3
# t0 = time.time()
# (logits, probs), _ = jit_gen_traj(params, rng, num_envs, num_steps)
# t1 = time.time()
# print(f"Time taken with {num_steps} and {num_envs} is {t1-t0}")

# num_steps = 1e4
# t0 = time.time()
# (logits, probs), _ = jit_gen_traj(params, rng, num_envs, num_steps)
# t1 = time.time()
# print(f"Time taken with {num_steps} and {num_envs} is {t1-t0}")

#%%
# # first we generate a small trajectory to view the distribution of logits
# rng = jax.random.PRNGKey(0)
# num_envs = 8
# num_steps = 1e6

# checkpointer = ocp.StandardCheckpointer()
# checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{1524}"
# folder_list = os.listdir(checkpoint_directory)
# params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
# (logits, probs, _), _ = jit_gen_traj(params, rng, num_envs, num_steps)
# print(logits.shape)
# # Then we view that distribution of logits for each action
# logits = jnp.reshape(logits, shape=(-1, 17))
# savedir = "/workspace/CraftaxDevinterp/intermediate_data/modelno_1524/action_distributions"
# os.makedirs(savedir, exist_ok=True)
# for action_no in range(17):
#     plt.hist(logits[:, action_no], bins=100)
#     plt.title(f"Action {action_no} logits")
#     plt.savefig(f"{savedir}/action_{action_no}_logits.png")
#     plt.close()


# probs = jnp.reshape(probs, shape=(-1, 17))

# for action_no in range(17):
#     plt.hist(probs[:, action_no], bins=100)
#     plt.title(f"Action {action_no} probs")
#     plt.savefig(f"{savedir}/action_{action_no}_probs.png")
#     plt.close()

#%%
# Seems like a reasonable cutoff is the dumb & obvious 50% prob. We can now condition on this, and get the relevant observations for each action
# TODO: scale this up

# checkpointer = ocp.StandardCheckpointer()
# rng = jax.random.PRNGKey(0)
# num_envs = 8
# num_steps = 1e4
# checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{1524}"
# folder_list = os.listdir(checkpoint_directory)
# params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")

# (logits, probs, obs), _ = jit_gen_traj(params, rng, num_envs, num_steps, log_obses=True)
# probs = jnp.reshape(probs, shape=(-1, 17))
# obs_shape = (obs.shape[0] * obs.shape[1],) + obs.shape[2:]
# obs = jnp.reshape(obs, shape=obs_shape)

# indices = [jnp.where(probs[:, i] > 0.5) for i in range(17)]

# conditioned_obs = [obs[idx] for idx in indices]

#%%
# Lets now verify that we get the correct actions for each obs set
from typing import Sequence
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax

class ActorCritic_with_hook(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "relu"

    def setup(self):
        # Actor layers
        self.Dense_0 = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name='Dense_0'
        )
        self.Dense_1 = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name='Dense_1'
        )
        self.Dense_2 = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name='Dense_2'
        )
        self.Dense_3 = nn.Dense(
            self.action_dim, 
            kernel_init=orthogonal(0.01), 
            bias_init=constant(0.0),
            name='Dense_3'
        )

        # Critic layers
        self.Dense_4 = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name='Dense_4'
        )
        self.Dense_5 = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name='Dense_5'
        )
        self.Dense_6 = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name='Dense_6'
        )
        self.Dense_7 = nn.Dense(
            1, 
            kernel_init=orthogonal(1.0), 
            bias_init=constant(0.0),
            name='Dense_7'
        )

    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        activations = list()
        actor_mean = self.Dense_0(x)
        actor_mean = activation(actor_mean)
        activations.append(actor_mean)

        actor_mean = self.Dense_1(actor_mean)
        actor_mean = activation(actor_mean)
        activations.append(actor_mean)

        actor_mean = self.Dense_2(actor_mean)
        actor_mean = activation(actor_mean)
        activations.append(actor_mean)

        actor_mean = self.Dense_3(actor_mean)
        activations.append(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # critic = self.Dense_4(x)
        # critic = activation(critic)

        # critic = self.Dense_5(critic)
        # critic = activation(critic)

        # critic = self.Dense_6(critic)
        # critic = activation(critic)

        # critic = self.Dense_7(critic)

        return pi, 0, activations
    def apply_dense(self, params, x):
        return jnp.dot(x, params['kernel']) + params['bias']
    
    def add_act_to_layer(
            self,
            params, 
            x, 
            activation_addition,
            layer: int, 
            activation_type: str = "relu"
        ):        
        activation = nn.relu if activation_type == "relu" else nn.tanh

        actor_mean = self.apply_dense(params['params']['Dense_0'], x)
        actor_mean = activation(actor_mean)
        actor_mean = jax.lax.select(
            layer == 0,
            actor_mean + activation_addition,
            actor_mean
        )
        
        actor_mean = self.apply_dense(params['params']['Dense_1'], actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = jax.lax.select(
            layer == 1,
            actor_mean + activation_addition,
            actor_mean
        )
        
        actor_mean = self.apply_dense(params['params']['Dense_2'], actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = jax.lax.select(
            layer == 2,
            actor_mean + activation_addition,
            actor_mean
        )
        
        actor_mean = self.apply_dense(params['params']['Dense_3'], actor_mean)
        pi = distrax.Categorical(logits=actor_mean)
        return pi


def get_activations(obs, params):
    network = ActorCritic_with_hook(17, 512)
    _, _, activations = network.apply(params, obs)
    return activations

def get_action_activations(
        states: jnp.ndarray, 
        params: dict, 
        debug=False
    ):
    vectorized_acts = jax.jit(jax.vmap(get_activations, in_axes=(0, None), out_axes=0))

    activations = vectorized_acts(states, params)
    if debug:
        ACTION_MAP = {
            0: "NOOP",
            1: "LEFT",
            2: "RIGHT",
            3: "UP",
            4: "DOWN",
            5: "DO",
            6: "SLEEP",
            7: "PLACE_STONE",
            8: "PLACE_TABLE",
            9: "PLACE_FURNACE",
            10: "PLACE_PLANT",
            11: "MAKE_WOOD_PICKAXE",
            12: "MAKE_STONE_PICKAXE",
            13: "MAKE_IRON_PICKAXE",
            14: "MAKE_WOOD_SWORD",
            15: "MAKE_STONE_SWORD",
            16: "MAKE_IRON_SWORD"
        }

        action = activations[-1]
        maximum_action = jnp.argmax(action, axis=1)

        for action in maximum_action:
            human_readable = ACTION_MAP[action.item()]
            print(human_readable)
    return activations

# for i in range(17):
#     get_action_activations(conditioned_obs[i], params, debug=True)
# %%
# And finally we can make the activation addition function
@jax.jit
def get_vec_addition_result(
        params: dict,
        obs: jnp.ndarray,
        addition: jnp.ndarray, 
        layer: int
):
    network = ActorCritic_with_hook(17, 512)
    pi = network.add_act_to_layer(params, obs, addition, layer)
    return pi.logits

@jax.jit
def logits_to_probs(logits):
    return jax.nn.softmax(logits, axis=-1)

def test_vector_addition(
        params: dict, 
        obs1: jnp.ndarray,
        obs2: jnp.ndarray,
        add_act_no: int, 
        sub_act_no: int, 
        layer: int, 
        scale: float = 1.0,
        debug: bool = False,
        verbose: bool = False
):
    network = ActorCritic(17, 512)
    pi, _ = network.apply(params, obs1)
    probs = pi.probs

    indices = [jnp.where(probs[:, i] > 0.5) for i in range(17)]
    conditioned_obs = [obs1[idx] for idx in indices]

    add_act = get_action_activations(conditioned_obs[add_act_no], params)[layer]
    if verbose:
        print(f"Norm of add act: {jnp.linalg.norm(add_act)}")
        print(f"Size of conditioned obs (act add): {conditioned_obs[add_act_no].shape}")
    add_act = add_act.mean(axis=0)
    sub_act = get_action_activations(conditioned_obs[sub_act_no], params)[layer]
    if verbose:
        print(f"Norm of sub act: {jnp.linalg.norm(sub_act)}")
        print(f"Size of conditioned obs (act sub): {conditioned_obs[sub_act_no].shape}")
    sub_act = sub_act.mean(axis=0)
    act_add = (add_act - sub_act) * scale
    if verbose:
        print(f"Norm of act add: {jnp.linalg.norm(act_add)}")

    pi, _ = network.apply(params, obs2)
    probs = pi.probs

    indices = [jnp.where(probs[:, i] > 0.5) for i in range(17)]
    conditioned_obs = [obs2[idx] for idx in indices]

    vectorized_act_addition = jax.vmap(
            get_vec_addition_result, 
            in_axes=(None, 0, None, None)
        )
    test_logits = vectorized_act_addition(params, conditioned_obs[sub_act_no], act_add, layer)
    if verbose:
        print(f"Norm of test logits: {jnp.linalg.norm(test_logits)}")

    control_act_nos = [i for i in range(17) if i != sub_act_no]
    control_obs_indices = jnp.where(probs[:, sub_act_no] <= 0.5)
    control_obs = obs2[control_obs_indices]
    control_logits_add = vectorized_act_addition(params, control_obs, act_add, layer)
    control_logits_null = vectorized_act_addition(params, control_obs, jnp.zeros_like(act_add), layer)
    control_diffs = control_logits_add - control_logits_null
    control_diff = jnp.linalg.norm(control_diffs)

    # for control_act_no in control_act_nos:
    #     control_logits_add = vectorized_act_addition(params, conditioned_obs[control_act_no], act_add, layer)
    #     control_logits_null = vectorized_act_addition(params, conditioned_obs[control_act_no], jnp.zeros_like(act_add), layer)
    #     control_diff = jnp.linalg.norm(control_logits_add - control_logits_null)

    # conditioned_obs_vec = jnp.concatenate( [conditioned_obs[i] for i in control_act_nos], axis=0)
    # if verbose:
    #     print(f"Conditioned obs vec shape: {conditioned_obs_vec.shape}")
    #     print(f"Conditioned obs vec norm: {jnp.linalg.norm(conditioned_obs_vec)}")
    # control_logits_add = vectorized_act_addition(params, conditioned_obs_vec, act_add, layer)
    # control_logits_null = vectorized_act_addition(params, conditioned_obs_vec, jnp.zeros_like(act_add), layer)

    # if verbose:
    #     print(f"Control logits add: {control_logits_add.mean()}")
    #     print(f"Control logits null: {control_logits_null.mean()}")
    # control_probs_add = logits_to_probs(control_logits_add)
    # control_probs_null = logits_to_probs(control_logits_null)
    # if verbose:
    #     print(f"Control probs add: {control_probs_add.mean()}")
    #     print(f"Control probs null: {control_probs_null.mean()}")

    # control_diffs = list()
    # for control_act_no in control_act_nos:
    #     if control_act_no == 0:
    #         start = 0
    #         end = conditioned_obs[control_act_no].shape[0]
    #     else:
    #         start = end
    #         end = start + conditioned_obs[control_act_no].shape[0]
    #     print(f"Start: {start}, End: {end}")
    #     if start != end:
    #         control_diffs.append(jnp.mean(control_probs_add[start:end] - control_probs_null[start:end]))
    #         print(f"Control diff for {control_act_no} is {control_diffs[-1]}")
    #     else:
    #         print(f"Control diff for {control_act_no} is nan so skipping")
    # if len(control_diffs) == 0:
    #     control_diff = 0.0
    # else:
    #     control_diff = jnp.mean(jnp.array(control_diffs))

    if verbose: print(f"Control diff is {control_diff}")


    # lets actually just look at the increase in the target action, and the decrease in the
    # nontarget action, relative to no act addition
    null_logits = vectorized_act_addition(
        params, 
        conditioned_obs[sub_act_no], 
        jnp.zeros_like(act_add), 
        layer
    )

    test_probs = logits_to_probs(test_logits)
    null_probs = logits_to_probs(null_logits)
    if verbose:
        print(f"Norm of test logits: {jnp.linalg.norm(test_logits)}")
        print(f"Norm of null logits: {jnp.linalg.norm(null_logits)}")
        print(f"Norm of test probs: {jnp.linalg.norm(test_probs)}")
        print(f"Norm of null probs: {jnp.linalg.norm(null_probs)}")

    target_action_add_logits = test_logits[:, add_act_no]
    target_action_null_logits = null_logits[:, add_act_no]
    if verbose:
        print(f"target action add logits mean: {target_action_add_logits.mean()}")
        print(f"target action null logits mean: {target_action_null_logits.mean()}")
    target_action_logit_diff = jnp.mean(target_action_add_logits - target_action_null_logits)

    nontarget_action_add_logits = test_logits[:, sub_act_no]
    nontarget_action_null_logits = null_logits[:, sub_act_no]
    if verbose:
        print(f"nontarget action add logits mean: {nontarget_action_add_logits.mean()}")
        print(f"nontarget action null logits mean: {nontarget_action_null_logits.mean()}")
    nontarget_action_logit_diff = jnp.mean(nontarget_action_add_logits - nontarget_action_null_logits)

    target_action_add_probs = test_probs[:, add_act_no]
    target_action_null_probs = null_probs[:, add_act_no]
    target_action_probs_diff = jnp.mean(target_action_add_probs - target_action_null_probs)
    if verbose:
        print(f"target action add probs mean: {target_action_add_probs.mean()}")
        print(f"target action null probs mean: {target_action_null_probs.mean()}")

    nontarget_action_add_probs = test_probs[:, sub_act_no]
    nontarget_action_null_probs = null_probs[:, sub_act_no]
    nontarget_action_probs_diff = jnp.mean(nontarget_action_add_probs - nontarget_action_null_probs)
    if verbose:
        print(f"nontarget action add probs mean: {nontarget_action_add_probs.mean()}")
        print(f"nontarget action null probs mean: {nontarget_action_null_probs.mean()}")

    if debug:
        ACTION_MAP = {
            0: "NOOP",
            1: "LEFT",
            2: "RIGHT",
            3: "UP",
            4: "DOWN",
            5: "DO",
            6: "SLEEP",
            7: "PLACE_STONE",
            8: "PLACE_TABLE",
            9: "PLACE_FURNACE",
            10: "PLACE_PLANT",
            11: "MAKE_WOOD_PICKAXE",
            12: "MAKE_STONE_PICKAXE",
            13: "MAKE_IRON_PICKAXE",
            14: "MAKE_WOOD_SWORD",
            15: "MAKE_STONE_SWORD",
            16: "MAKE_IRON_SWORD"
        }

        action = test_logits
        maximum_action = jnp.argmax(action, axis=1)

        for action in maximum_action:
            human_readable = ACTION_MAP[action.item()]
            print(human_readable)
    
    return test_logits, control_diff, target_action_logit_diff, nontarget_action_logit_diff, target_action_probs_diff, nontarget_action_probs_diff, control_diffs

import time
checkpointer = ocp.StandardCheckpointer()
rng = jax.random.PRNGKey(0)
num_envs = 8
num_steps = 1e4
checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{1524}"
folder_list = os.listdir(checkpoint_directory)
params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
seed = 0
rng = jax.random.PRNGKey(seed)
rng, traj_rng = jax.random.split(rng)
(_, probs, obs), _ = jit_gen_traj(params, traj_rng, num_envs, num_steps, log_obses=True)
obs_shape = (obs.shape[0] * obs.shape[1],) + obs.shape[2:]
obs1 = jnp.reshape(obs, shape=obs_shape)

rng, traj_rng = jax.random.split(rng)
(_, probs, obs), _ = jit_gen_traj(params, traj_rng, num_envs, num_steps, log_obses=True)
probs = jnp.reshape(probs, shape=(-1, 17))
obs_shape = (obs.shape[0] * obs.shape[1],) + obs.shape[2:]
obs2 = jnp.reshape(obs, shape=obs_shape)


t0 = time.time()
h = test_vector_addition(params, obs1, obs2, 8, 7, 1)
test_logits, control_diff, target_action_logit_diff, nontarget_action_logit_diff, target_action_probs_diff, nontarget_action_probs_diff, control_diffs = h
t1 = time.time()
print(f"Time taken is {t1-t0}")
print(f"Control diff is {control_diff}")
print(f"Target action logit delta is {target_action_logit_diff}")
print(f"Nontarget action logit delta is {nontarget_action_logit_diff}")
print(f"Target action probs delta is {target_action_probs_diff}")
print(f"Nontarget action probs delta is {nontarget_action_probs_diff}")

#%%
# Now there are three things we care to vary here:
# 1. Time
# 2. Add vec number
# 3. Sub vec number

# I'm skeptical of control. Its always zero. That can't be right!

# numbers = range(17)
# all_pairs = [(i, j) for i in numbers for j in numbers if i != j]

jitted_tva = jax.jit(test_vector_addition, static_argnames=("add_act_no", "sub_act_no", "layer"))
add_act_no = 8
sub_act_no = 7
batch_size = 10
save_dir = f"/workspace/CraftaxDevinterp/intermediate_data/add_act_{add_act_no}_sub_act_{sub_act_no}/time_series"
os.makedirs(save_dir, exist_ok=True)

def save_batch(results, modelnos, save_dir):
    for result, modelno in zip(results, modelnos):
        np.save(f"{save_dir}/{modelno}.npy", result)

results = list()
modelnos = list()
for modelno in tqdm(range(1525)):
    checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{modelno}"
    folder_list = os.listdir(checkpoint_directory)
    params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
    r = test_vector_addition(params, obs1, obs2, add_act_no, sub_act_no, 1)
    test_logits, control_diff, target_action_logit_diff, nontarget_action_logit_diff, target_action_probs_diff, nontarget_action_probs_diff, control_diffs = r
    r = np.array([control_diff, target_action_probs_diff, nontarget_action_probs_diff])
    if modelno % batch_size == 0:
        results.append(r)
        modelnos.append(modelno)
        save_batch(results, modelnos, save_dir)
        results = list()
        modelnos = list()
    else:
        results.append(r)
        modelnos.append(modelno)
save_batch(results, modelnos, save_dir)

results = []
for modelno in range(1525):
    file_path = f"{save_dir}/{modelno}.npy"
    if os.path.exists(file_path):
        results.append(np.load(file_path))
    else:
        print(f"Warning: File not found for model {modelno}")

results = np.array(results)

plt.plot(results[:, 0], label="control_delta")
plt.plot(results[:, 1], label="target_delta")
plt.plot(results[:, 2], label="nontarget_delta")
plt.legend()
plt.savefig(f"{save_dir}/results.png")
plt.close()

#%%
import numpy as np
dir = "/workspace/CraftaxDevinterp/intermediate_data/add_act_8_sub_act_7/time_series"
results = []
for modelno in range(1525):
    file_path = f"{dir}/{modelno}.npy"
    if os.path.exists(file_path):
        results.append(np.load(file_path))
    else:
        print(f"Warning: File not found for model {modelno}")

results = np.array(results)
# results = np.nan_to_num(results, nan=0.0)
print(results[:, 1])
# %%
from matplotlib import pyplot as plt
save_dir = "/workspace/CraftaxDevinterp/intermediate_data/add_act_8_sub_act_7/time_series"
plt.plot(results[:, 0], label="control_delta")
plt.plot(results[:, 1], label="target_delta")
plt.plot(results[:, 2], label="nontarget_delta")
plt.legend()
# plt.savefig(f"{save_dir}/results.png")
plt.show()
plt.close()

# %%
