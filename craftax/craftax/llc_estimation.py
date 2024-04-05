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

layer_size = 512
seed = 0
rng = jax.random.PRNGKey(seed)
rng, _rng = jax.random.split(rng)

#%%
def generate_trajectory(network_params, rng, num_envs=1, num_steps=100):
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
    predictions = model.apply(param, inputs)
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

sgld_config = SGLDConfig(
    {
        'epsilon': 1e-6, 
        'gamma': 1.0, 
        'num_steps': 100, 
        'num_chains': 1, # NOTE: Code doesn't have >1 chains available yet
        'batch_size': 128
    }
)

model_no = 1000
checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{model_no}"
checkpointer = ocp.StandardCheckpointer()
folder_list = os.listdir(checkpoint_directory)
network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")

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
    train_batch_loss = True, 
    compute_distance = False, 
    verbose = True
)

init_loss = loss_fn(network_params, expert_obses, expert_logitses)
lambdahat = float(np.mean(loss_trace) - init_loss) * num_training_data * itemp
print(lambdahat)
#%%