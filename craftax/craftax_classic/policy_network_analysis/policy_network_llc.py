#%%
import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax.training.train_state import TrainState
import orbax.checkpoint as ocp

from craftax.logz.batch_logging import batch_log, create_log_dict
from craftax.models.actor_critic import (
    ActorCritic
)
from craftax.environment_base.wrappers import (
    LogWrapper,
    AutoResetEnvWrapper,
    BatchEnvWrapper,
)
from craftax.craftax_classic.envs.craftax_symbolic_env import (
    CraftaxClassicSymbolicEnv,
)
from typing import NamedTuple
import os

num_envs = 64
gamma = 0.99
gae_lambda = 0.8
clip_eps = 0.2
entropy_coeff = 0.01
max_grad_norm = 1.0
lr = 2e-4
num_steps = 64

env = CraftaxClassicSymbolicEnv()
env_params = env.default_params
env = LogWrapper(env)
env = AutoResetEnvWrapper(env)
env = BatchEnvWrapper(env, num_envs=num_envs)
network = ActorCritic(env.action_space(env_params).n, 512, activation="relu")

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

def _env_step(runner_state, unused):
    (
        train_state,
        env_state,
        last_obs,
        ex_state,
        rng,
        update_step,
    ) = runner_state

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    pi, value = network.apply(train_state.params, last_obs)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    obsv, env_state, reward_e, done, info = env.step(
        _rng, env_state, action, env_params
    )

    reward_i = jnp.zeros(num_envs)

    reward = reward_e + reward_i

    transition = Transition(
        done=done,
        action=action,
        value=value,
        reward=reward,
        reward_i=reward_i,
        reward_e=reward_e,
        log_prob=log_prob,
        obs=last_obs,
        next_obs=obsv,
        info=info,
    )
    runner_state = (
        train_state,
        env_state,
        obsv,
        ex_state,
        rng,
        update_step,
    )
    return runner_state, transition

def _calculate_gae(traj_batch, last_val):
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = (
            transition.done,
            transition.value,
            transition.reward,
        )
        delta = reward + gamma * next_value * (1 - done) - value
        gae = (
            delta
            + gamma * gae_lambda * (1 - done) * gae
        )
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value

def policy_loss(params, traj_batch, gae):
    pi, value = network.apply(params, traj_batch.obs)
    log_prob = pi.log_prob(traj_batch.action)

    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio, 
            1.0 - clip_eps, 
            1.0 + clip_eps
        )
        * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()
    return loss_actor - entropy_coeff * entropy

def data_from_params(params):
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    obsv, env_state = env.reset(_rng, env_params)

    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )

    ex_state = {
        "icm_encoder": None,
        "icm_forward": None,
        "icm_inverse": None,
        "e3b_matrix": None,
    }


    runner_state = (
        train_state,
        env_state,
        obsv,
        ex_state,
        _rng,
        0,
    )


    runner_state, traj_batch = jax.lax.scan(
        _env_step, runner_state, None, num_steps
    )
    (
        train_state,
        env_state,
        last_obs,
        ex_state,
        rng,
        update_step,
    ) = runner_state

    last_val = network.apply(params, last_obs)
    advantages, targets = _calculate_gae(traj_batch, last_val)
    return traj_batch, advantages, targets

rng = jax.random.PRNGKey(0)
rng, _rng = jax.random.split(rng)
obsv, env_state = env.reset(_rng, env_params)

rng, init_rng = jax.random.split(rng)
network_structure = network.init(init_rng, obsv)
checkpointer = ocp.StandardCheckpointer()
def load_model(modelno):
    checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{modelno}"
    folder_list = os.listdir(checkpoint_directory)
    params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}", item=network_structure)
    return params

#%%
from craftax.craftax.sgld_utils import run_sgld, SGLDConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
get_data = jax.jit(
    lambda param: data_from_params(param)
)
loss_fn = jax.jit(
    lambda param, inputs, targets: policy_loss(param, inputs, targets)
)

def find_lambdahat(rng, params):
    traj_batch, advantages, _ = get_data(params)
    loss_trace, _, _ = run_sgld(
        rng, 
        loss_fn, 
        sgld_config, 
        params, 
        traj_batch, 
        advantages, 
        itemp=itemp
    )
    init_loss = loss_fn(params, traj_batch, advantages)
    lambdahat = float(np.mean(loss_trace) - init_loss) * traj_batch.obs.shape[0] * itemp
    return lambdahat

itemp = 0.001
sgld_config = SGLDConfig(
    epsilon = 1e-5, 
    gamma = 10, 
    num_steps = 1e4, 
    num_chains = 1,
    batch_size = 1024
)

min_models = 0
max_models = 1525
count_by = 100
llcs = np.zeros(((max_models-min_models)//count_by + 1))
pbar = tqdm(total=(max_models-min_models)//count_by + 1, desc="Model llc progress")

load_model(max_models-1) # for verification purposes

rng, sgld_loop_rng = jax.random.split(rng)
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

folder = f"/workspace/CraftaxDevinterp/craftax/craftax_classic/policy_network_analysis/temp_{itemp}/eps_{sgld_config.epsilon}/gamma_{sgld_config.gamma}/num_steps_{sgld_config.num_steps}/num_chains_{sgld_config.num_chains}/batch_{sgld_config.batch_size}/min_models_{min_models}/max_models_{max_models}/countby_{count_by}"
os.makedirs(folder, exist_ok=True)

# visualize llcs
plt.plot(llcs[:-1])
plt.savefig(f"{folder}/llcs_over_time_countby.png")
plt.close()

np.save(f"{folder}/llcs_over_time_countby", llcs)

# TODO: make behavioral
# TODO: change hyperparams to:
# itemp = 0.01
# eps = 1e-6
# gam = 1e3
# num_step = 1e4
# batch = 64