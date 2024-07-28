#%%
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA
from craftax.craftax_classic.make_model_trajectory import jit_gen_traj
import orbax.checkpoint as ocp
import os
import einops
from craftax.models.actor_critic import ActorCritic
import time
from tqdm import tqdm
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from craftax.environment_base.wrappers import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
)
import matplotlib.pyplot as plt

rng = jax.random.PRNGKey(0)
# load end model
env = CraftaxClassicSymbolicEnv()
env = AutoResetEnvWrapper(env)
env = BatchEnvWrapper(env, 64)
env_params = env.default_params
rng, env_rng = jax.random.split(rng)
obsv, env_state = env.reset(env_rng, env_params)

network = ActorCritic(17, 512, activation="relu")
run_network = jax.jit(network.apply)

rng, init_rng = jax.random.split(rng)
network_structure = network.init(init_rng, obsv)
checkpointer = ocp.StandardCheckpointer()
def load_model(modelno):
    checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{modelno}"
    folder_list = os.listdir(checkpoint_directory)
    params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}", item=network_structure)
    return params
params = load_model(1524)

# generate observation dataset
num_envs = 64
num_steps = 1e2
rng, trajectory_rng = jax.random.split(rng)
(_, _, obses), dones = jit_gen_traj(params, trajectory_rng, num_envs, num_steps, log_obses=True)
obses = einops.rearrange(obses, 's e o -> (s e) o')

# Generate "true" (end-state-policy) a's
_, true_a = run_network(params, obses)

from craftax.craftax.sgld_utils import run_sgld, SGLDConfig

def mse_loss(param, inputs, targets):
    _, predictions = run_network(param, inputs)
    return jnp.mean((predictions - targets) ** 2)

loss_fn = jax.jit(
    lambda param, inputs, targets: mse_loss(param, inputs, targets)
)
num_training_data = obses.shape[0]
max_models = 1525
count_by = 100
rng, sgld_rng = jax.random.split(rng)
itemp = 0.02
sgld_config = SGLDConfig(
    epsilon = 1e-5, 
    gamma = 1e3, 
    num_steps = 1e3, 
    num_chains = 1,
    batch_size = 64
)
# loss_trace, _, _ = run_sgld(
#     sgld_rng, 
#     loss_fn, 
#     sgld_config, 
#     params, 
#     obses, 
#     true_a, 
#     itemp=itemp
# )
# init_loss = loss_fn(params, obses, true_a)
# lambdahat = float(np.mean(loss_trace) - init_loss) * num_training_data * itemp

# plt.plot(loss_trace)
#%%
from itertools import product


def sgld_parameter_search(sgld_rng, params, obses, true_a, debug=False, filename="llc_traces.png"):
    def run_search():
        if debug:
            epsilon_range = [1e-4]
            gamma_range = [1e2]
            num_steps_range = [1e2]
            itemp_range = [0.02]
        else:
            epsilon_range = [1e-5, 1e-4, 1e-3]
            gamma_range = [1e1, 1e2, 1e3]
            num_steps_range = [1e3, 1e4]
            itemp_range = [1.0, 0.1, 0.01, 0.001, 1e-3]

        num_training_data = obses.shape[0]
        results = []

        total_iterations = len(epsilon_range) * len(gamma_range) * len(num_steps_range) * len(itemp_range)
        pbar = tqdm(total=total_iterations, desc="Parameter Search Progress")

        for epsilon, gamma, num_steps, itemp in product(epsilon_range, gamma_range, num_steps_range, itemp_range):
            sgld_config = SGLDConfig(
                epsilon=epsilon,
                gamma=gamma,
                num_steps=int(num_steps),
                num_chains=1,
                batch_size=64
            )

            loss_trace, _, mala = run_sgld(
                sgld_rng,
                loss_fn,
                sgld_config,
                params,
                obses,
                true_a,
                itemp=itemp
            )
            mala_acceptance = np.mean([e[1] for e in mala])

            init_loss = loss_fn(params, obses, true_a)
            lambdahat = float(np.mean(loss_trace) - init_loss) * num_training_data * itemp

            results.append({
                'epsilon': epsilon,
                'gamma': gamma,
                'num_steps': num_steps,
                'itemp': itemp,
                'lambdahat': lambdahat,
                'final_loss': loss_trace[-1],
                'loss_trace': loss_trace,
                'init_loss': init_loss,
                'mala_acceptance': mala_acceptance
            })

            pbar.update(1)

        pbar.close()
        return results

    def save_results(results, filename):
        num_param_plots = 4
        num_loss_plots = len(results)
        total_plots = num_param_plots + num_loss_plots

        # Create a figure with a 2x2 grid for parameter plots and a larger grid for loss traces
        fig = plt.figure(figsize=(40, 20 + 5 * ((num_loss_plots + 3) // 4)))
        gs = fig.add_gridspec(2 + (num_loss_plots + 3) // 4, 4)
        
        plot_pbar = tqdm(total=total_plots, desc="Plotting Progress")

        # Parameter plots
        for idx, param in enumerate(['epsilon', 'gamma', 'num_steps', 'itemp']):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            x = [r[param] for r in results]
            y = [r['final_loss'] for r in results]
            ax.scatter(x, y)
            ax.set_xlabel(param)
            ax.set_ylabel('Final Loss')
            ax.set_title(f'Final Loss vs {param}')
            plot_pbar.update(1)

        # Loss trace plots
        for idx, result in enumerate(results):
            ax = fig.add_subplot(gs[2 + idx // 4, idx % 4])
            ax.plot(result['loss_trace'], label='Loss Trace')
            ax.axhline(y=result['init_loss'], color='r', linestyle='--', label='Initial Loss')
            ax.set_title(f"ε={result['epsilon']}, γ={result['gamma']},\nn_steps={result['num_steps']}, itemp={result['itemp']}\nMALA acc: {result['mala_acceptance']:.4f}, llc={result['lambdahat']}")
            ax.set_xlabel('Steps')
            ax.set_ylabel('Loss')
            ax.legend()
            plot_pbar.update(1)

        plot_pbar.close()

        plt.tight_layout()
        
        save_dir = '/workspace/CraftaxDevinterp/craftax/craftax_classic/value_network_analysis/'
        os.makedirs(save_dir, exist_ok=True)
        
        filename = 'debugging_llc_traces.png' if debug else filename
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Plots saved to {filepath}")

    results = run_search()
    save_results(results, filename)

    return results

print(f"Calibrating model number 1524")
results = sgld_parameter_search(sgld_rng, params, obses, true_a, debug=False, filename="llc_traces_1524.png")

print("calibrating model number 100")
params = load_model(100)
results = sgld_parameter_search(sgld_rng, params, obses, true_a, debug=False, filename="llc_traces_0100.png")

#%%

itemp = 0.01
num_training_data = obses.shape[0]
max_models = 1525
count_by = 100
rng, sgld_rng = jax.random.split(rng)
llcs = np.zeros((max_models//count_by + 1))
for i, modelno in tqdm(enumerate(range(0, max_models, count_by))):
    params = load_model(modelno)
    loss_trace, _, _ = run_sgld(
        sgld_rng, 
        loss_fn, 
        sgld_config, 
        params, 
        obses, 
        true_a, 
        itemp=itemp
    )
    init_loss = loss_fn(params, obses, true_a)
    lambdahat = float(np.mean(loss_trace) - init_loss) * num_training_data * itemp

    llcs[i] = lambdahat
plt.plot(llcs)
# %%
