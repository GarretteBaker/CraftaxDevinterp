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
# step 0: getting a sense of time
rng = jax.random.PRNGKey(0)
num_envs = 64
num_steps = 1e2
checkpointer = ocp.StandardCheckpointer()
checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{1524}"
folder_list = os.listdir(checkpoint_directory)
params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")

import time
t0 = time.time()
(logits, probs, _), _ = jit_gen_traj(params, rng, num_envs, num_steps)
t1 = time.time()
print(f"Time taken with {num_steps} and {num_envs} is {t1-t0}")

num_steps = 1e3
t0 = time.time()
(logits, probs), _ = jit_gen_traj(params, rng, num_envs, num_steps)
t1 = time.time()
print(f"Time taken with {num_steps} and {num_envs} is {t1-t0}")

num_steps = 1e4
t0 = time.time()
(logits, probs), _ = jit_gen_traj(params, rng, num_envs, num_steps)
t1 = time.time()
print(f"Time taken with {num_steps} and {num_envs} is {t1-t0}")

#%%
# first we generate a small trajectory to view the distribution of logits
rng = jax.random.PRNGKey(0)
num_envs = 8
num_steps = 1e6

checkpointer = ocp.StandardCheckpointer()
checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{1524}"
folder_list = os.listdir(checkpoint_directory)
params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
(logits, probs, _), _ = jit_gen_traj(params, rng, num_envs, num_steps)
print(logits.shape)
# Then we view that distribution of logits for each action
logits = jnp.reshape(logits, shape=(-1, 17))
savedir = "/workspace/CraftaxDevinterp/intermediate_data/modelno_1524/action_distributions"
os.makedirs(savedir, exist_ok=True)
for action_no in range(17):
    plt.hist(logits[:, action_no], bins=100)
    plt.title(f"Action {action_no} logits")
    plt.savefig(f"{savedir}/action_{action_no}_logits.png")
    plt.close()


probs = jnp.reshape(probs, shape=(-1, 17))

for action_no in range(17):
    plt.hist(probs[:, action_no], bins=100)
    plt.title(f"Action {action_no} probs")
    plt.savefig(f"{savedir}/action_{action_no}_probs.png")
    plt.close()

#%%
# Seems like a reasonable cutoff is the dumb & obvious 50% prob. We can now condition on this, and get the relevant observations for each action
# TODO: scale this up

checkpointer = ocp.StandardCheckpointer()
rng = jax.random.PRNGKey(0)
num_envs = 8
num_steps = 1e4
checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{1524}"
folder_list = os.listdir(checkpoint_directory)
params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")

(logits, probs, obs), _ = jit_gen_traj(params, rng, num_envs, num_steps, log_obses=True)
probs = jnp.reshape(probs, shape=(-1, 17))
obs_shape = (obs.shape[0] * obs.shape[1],) + obs.shape[2:]
obs = jnp.reshape(obs, shape=obs_shape)

indices = [jnp.where(probs[:, i] > 0.5) for i in range(17)]

conditioned_obs = [obs[idx] for idx in indices]

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

get_action_activations(conditioned_obs[0], params, debug=True)
# %%
