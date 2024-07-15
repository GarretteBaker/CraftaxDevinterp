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
def generate_trajectory(network_params, rng, num_envs, num_steps):
    env = CraftaxClassicSymbolicEnv()
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs)
    env_params = env.default_params
    network = ActorCritic(env.action_space(env_params).n, 512, activation="relu")

    class Transition(NamedTuple):
        logits: jnp.ndarray
        probs: jnp.ndarray
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
    return (traj_batch.logits, traj_batch.probs), traj_batch.done
jit_gen_traj = jax.jit(generate_trajectory, static_argnames=("num_envs", "num_steps"))
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

#%%
import time
t0 = time.time()
(logits, probs), _ = jit_gen_traj(params, rng, num_envs, num_steps)
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
num_envs = 64
num_steps = 1e6

checkpointer = ocp.StandardCheckpointer()
checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{1524}"
folder_list = os.listdir(checkpoint_directory)
params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
(logits, probs), _ = jit_gen_traj(params, rng, num_envs, num_steps)
print(logits.shape)
# %%
# Then we view that distribution of logits for each action
logits = jnp.reshape(logits, shape=(-1, 17))
savedir = "/workspace/CraftaxDevinterp/intermediate_data/modelno_1524/action_distributions"

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