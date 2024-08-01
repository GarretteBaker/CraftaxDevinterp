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
from itertools import product
import einops

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
    obs = traj_batch[..., :-2]
    action = traj_batch[..., -2]
    traj_batch_log_prob = traj_batch[..., -1]
    pi, value = network.apply(params, obs)
    log_prob = pi.log_prob(action)

    ratio = jnp.exp(log_prob - traj_batch_log_prob)
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
    num_steps = 64
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

    _, last_val = network.apply(params, last_obs)
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
import textwrap

get_data = jax.jit(
    lambda param: data_from_params(param)
)
loss_fn = jax.jit(
    lambda param, inputs, targets: policy_loss(param, inputs, targets)
)

def find_lambdahat(rng, params, itemp, epsilon, gamma, num_steps, num_chains, batch_size):
    sgld_config = SGLDConfig(
        epsilon = epsilon, 
        gamma = gamma, 
        num_steps = num_steps, 
        num_chains = num_chains,
        batch_size = batch_size
    )
    traj_batch, advantages, _ = get_data(params)

    traj_batch_vect = jnp.concatenate(
        [
            traj_batch.obs, 
            jnp.expand_dims(
                traj_batch.action, 
                axis=-1
            ), 
            jnp.expand_dims(
                traj_batch.log_prob, 
                axis=-1
            )
        ], 
        axis=-1
    )
    traj_batch_vect = einops.rearrange(traj_batch_vect, "e s d -> (e s) d")
    advantages = einops.rearrange(advantages, "e s-> (e s)")
    loss_trace, _, mala = run_sgld(
        rng, 
        loss_fn, 
        sgld_config, 
        params, 
        traj_batch_vect, 
        advantages, 
        itemp=itemp
    )
    init_loss = loss_fn(params, traj_batch_vect, advantages)
    lambdahat = float(np.mean(loss_trace) - init_loss) * traj_batch.obs.shape[0] * itemp
    return loss_trace, init_loss, lambdahat, np.mean([e[1] for e in mala])

# # Debugging
# itemps = [1e-1]
# epsilons = [1e-3]
# gammas = [10]
# num_steps = [1e3]
# num_chains = [1]
# batch_sizes = [64]


itemps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
epsilons = [1e-3, 1e-4, 1e-5, 1e-6]
gammas = [10, 100, 1e3, 1e4]
num_steps = [1e3, 1e4]
num_chains = [1]
batch_sizes = [64]
def calibrate(modelno):
    params = load_model(modelno)
    results = list()
    combos = product(itemps, epsilons, gammas, num_steps, batch_sizes)
    combo_len = len(itemps) * len(epsilons) * len(gammas) * len(num_steps) * len(batch_sizes)
    pbar = tqdm(total=combo_len, desc=f"Parameter Search Progress on {modelno}")
    for itemp, epsilon, gamma, num_step, batch_size in combos:
        loss_trace, init_loss, lambdahat, mala = find_lambdahat(
            rng, 
            params, 
            itemp, 
            epsilon, 
            gamma, 
            num_step, 
            1,
            batch_size
        )
        results.append(
            {
                "ITEMP": itemp, 
                "EPSILON": epsilon, 
                "GAMMA": gamma, 
                "NUM_STEPS": num_step, 
                "BATCH_SIZE": batch_size, 
                "LOSS_TRACE": loss_trace, 
                "INIT_LOSS": init_loss, 
                "LLC": lambdahat, 
                "MALA": mala
            }
        )
        pbar.update(1)
    pbar.close()

    num_plots = len(results)
    print(f"Num plots: {num_plots}")

    num_cols = 5
    num_rows = int(np.ceil(num_plots / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))
    axes = axes.flatten()

    def create_title(itemp, eps, gamma, num_step, batch_size, lambdahat, mala):
        def format_number(num):
            if abs(num) >= 1000 or (abs(num) < 0.01 and num != 0):
                return f"{num:.2e}"
            else:
                return f"{num:.2f}"

        formatted_params = [
            f"itemp = {format_number(itemp)}",
            f"eps = {format_number(eps)}",
            f"gamma = {format_number(gamma)}",
            f"num_step = {num_step}",
            f"batch_size = {batch_size}",
            f"llc = {format_number(lambdahat)}",
            f"mala = {mala}"
        ]

        return ", ".join(formatted_params)

    pbar = tqdm(total = len(results), desc = f"Creating plot for modelno {modelno}")
    for i, result in enumerate(results):
        axes[i].plot(
            result["LOSS_TRACE"]
        )
        axes[i].axhline(
            y = result["INIT_LOSS"], 
            color="r", 
            linestyle="--"
        )
        title = create_title(
            result["ITEMP"], 
            result["EPSILON"], 
            result["GAMMA"], 
            result["NUM_STEPS"], 
            result["BATCH_SIZE"], 
            result["LLC"],
            result["MALA"]
        )
        max_title_length = 40
        wrapped_title = textwrap.fill(title, max_title_length)
        axes[i].set_title(wrapped_title, fontsize=10)
        pbar.update(1)

    folder = "/workspace/CraftaxDevinterp/craftax/craftax_classic/policy_network_analysis/llc_calibration"
    os.makedirs(folder, exist_ok=True)

    plt.tight_layout()
    plt.savefig(f"{folder}/{modelno}_policy_loss.png")
    plt.close()

print(f"Calibrating 100")
calibrate(100)
print("Calibrating 1524")
calibrate(1524)