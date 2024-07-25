import jax
import jax.numpy as jnp
from typing import NamedTuple
from craftax.environment_base.wrappers import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
)
from craftax.models.actor_critic import (
    ActorCritic,
)
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv

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
