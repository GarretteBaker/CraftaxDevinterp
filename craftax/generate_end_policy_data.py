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
import pickle
import os

#%%
layer_size = 512
seed = 0

def generate_trajectory(rng, num_envs=1, num_steps=496):
    env = CraftaxSymbolicEnv()
    env_params = env.default_params
    env = LogWrapper(env)
    env = OptimisticResetVecEnvWrapper(
        env,
        num_envs=num_envs,
        reset_ratio=min(16, num_envs),
    )
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_directory = "/workspace/Craftax/end_model"
    folder_list = os.listdir(checkpoint_directory)
    network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
    network = ActorCritic(env.action_space(env_params).n, layer_size)

    class Transition(NamedTuple):
        obs: jnp.ndarray

    # COLLECT TRAJECTORIES
    def _env_step(runner_state, unused):
        (
            env_state,
            last_obs,
            rng,
        ) = runner_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi, _ = network.apply(network_params["params"], last_obs)
        action = pi.sample(seed=_rng)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state, _, _, _ = env.step(
            _rng, env_state, action, env_params
        )

        transition = Transition(
            obs=last_obs,
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
    return traj_batch.obs

rng = jax.random.PRNGKey(seed)
rng, _rng = jax.random.split(rng)

jit_gen_traj = jax.jit(generate_trajectory)
#%%
obs = generate_trajectory(_rng)
obs = jnp.squeeze(obs, axis=1)
print(obs.shape)
#%%
obs = jax.device_get(obs)
end_data_dir = "/workspace/Craftax/craftax/end_data"
os.makedirs(end_data_dir, exist_ok=True)
with open(f"{end_data_dir}/end_data.pkl", "wb") as f:
    pickle.dump(obs, f)

# %%

obs = jax.device_put(obs)
env = CraftaxSymbolicEnv()
env_params = env.default_params
env = LogWrapper(env)
env = OptimisticResetVecEnvWrapper(
    env,
    num_envs=1,
    reset_ratio=min(16, 1),
)
checkpointer = ocp.StandardCheckpointer()
checkpoint_directory = "/workspace/Craftax/end_model"
folder_list = os.listdir(checkpoint_directory)
network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")
network = ActorCritic(env.action_space(env_params).n, layer_size)

pi, _ = network.apply(network_params["params"], obs)
logits = pi.logits
# %%
