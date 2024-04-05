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

#%%
