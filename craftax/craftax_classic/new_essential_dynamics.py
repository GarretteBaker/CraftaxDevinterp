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
import orbax.checkpoint as ocp
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import scipy as sp

class Tracker(NamedTuple):
    block_placements: jnp.ndarray
    block_mining: jnp.ndarray
    player_location: jnp.ndarray
    player_movement: jnp.ndarray
    # revealed_blocks: jnp.ndarray
    doings: jnp.ndarray
    mob_kills: jnp.ndarray
    mob_attacks: jnp.ndarray
    time: jnp.ndarray


#%%
num_steps = 2e3
def generate_trajectory(network_params, rng, num_envs=64, num_steps=num_steps):
    env = CraftaxClassicSymbolicEnv()
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs)
    env_params = env.default_params
    network = ActorCritic(env.action_space(env_params).n, 512)

    class Transition(NamedTuple):
        tracking: jnp.ndarray
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
        action = pi.sample(seed=_rng)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state, _, done, _ = env.step(
            _rng, past_state, action, env_params
        )

        tracker = obsv

        transition = Transition(
            tracking = tracker,
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
    return traj_batch.tracking, traj_batch.done
jit_gen_traj = jax.jit(generate_trajectory)


checkpointer = ocp.StandardCheckpointer()
checkpoint_directory = "/workspace/CraftaxDevinterp/intermediate"
checkpoint_list = os.listdir(checkpoint_directory)
checkpoint_folder = f"{checkpoint_directory}/{checkpoint_list[-1]}"
folder_list = os.listdir(checkpoint_folder)
network_params = checkpointer.restore(f"{checkpoint_folder}/{folder_list[0]}")
seed = 0
rng = jax.random.PRNGKey(seed)
rng, _rng = jax.random.split(rng)
trajectory, done = jit_gen_traj(network_params, _rng)
print("Success!")
#%%


#%%
save_dir = "/workspace/CraftaxDevinterp/essential_dynamics"
os.makedirs(save_dir, exist_ok=True)
network = ActorCritic(17, 512)
num_models = len(checkpoint_list)

for modelno, foldername in tqdm(enumerate(checkpoint_list)):
    os.makedirs(f"{save_dir}/{modelno}", exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_directory = f"/workspace/CraftaxDevinterp/intermediate/{foldername}"
    folder_list = os.listdir(checkpoint_directory)
    network_params = checkpointer.restore(f"{checkpoint_directory}/{folder_list[0]}")

    p, _ = network.apply(network_params, trajectory)
    logits = p.logits

    with open(f"{save_dir}/{modelno}/logits.pkl", "wb") as f:
        pickle.dump(logits, f)

#%%
from tqdm import tqdm
import numpy as np
import pickle

checkpoint_directory = "/workspace/CraftaxDevinterp/intermediate"
checkpoint_list = os.listdir(checkpoint_directory)
num_models = len(checkpoint_list)
save_dir = "/workspace/CraftaxDevinterp/essential_dynamics"
logits_matrix = np.zeros((num_models//10 + 1, 2000, 64, 17))
for modelno in tqdm(range(0, num_models, 10)):
    with open(f"{save_dir}/{modelno}/logits.pkl", "rb") as f:
        logits = pickle.load(f)
    logits_matrix[modelno//10, :, :, :] = logits

logits_matrix = logits_matrix.reshape(num_models//10+1, 2000*64*17)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(logits_matrix)
projected = pca.transform(logits_matrix)

import itertools
import matplotlib.pyplot as plt
pc_combos = itertools.combinations(range(3), 2)
for pc1, pc2 in pc_combos:
    plt.scatter(projected[:, pc1], projected[:, pc2])
    plt.xlabel(f"PC{pc1}")
    plt.ylabel(f"PC{pc2}")
    plt.title(f"PC{pc1} vs PC{pc2}")
    plt.show()
# %%
from tqdm import tqdm
import numpy as np
import pickle
import plotly.express as px
from sklearn.decomposition import PCA
import itertools
skip = 3
add = 1
checkpoint_directory = "/workspace/CraftaxDevinterp/intermediate"
checkpoint_list = os.listdir(checkpoint_directory)
num_models = len(checkpoint_list)
if num_models % skip == 0: add = 0
# Load and process data
save_dir = "/workspace/CraftaxDevinterp/essential_dynamics"
logits_matrix = np.zeros((num_models // skip + add, 2000, 64, 17))
for modelno in tqdm(range(0, num_models, skip)):
    with open(f"{save_dir}/{modelno}/logits.pkl", "rb") as f:
        logits = pickle.load(f)
    logits_matrix[modelno // skip, :, :, :] = logits

# Reshape for PCA
logits_matrix = logits_matrix.reshape(num_models // skip + add, 2000 * 64 * 17)

# Perform PCA
pca = PCA(n_components=3)
projected = pca.fit_transform(logits_matrix)

# Create interactive plots
timestamps = range(0, num_models, skip)
for pc1, pc2 in itertools.combinations(range(3), 2):
    fig = px.scatter(
        x=projected[:, pc1], 
        y=projected[:, pc2], 
        labels={
            'x': f'PC{pc1}',
            'y': f'PC{pc2}'
        },
        title=f'PC{pc1} vs PC{pc2}',
        hover_data=[timestamps]
    )
    fig.update_traces(marker=dict(size=5))
    fig.show()
#%%
import tqdm
import numpy as np
import pickle
import plotly.express as px
from sklearn.decomposition import PCA
import itertools
from scipy.ndimage import gaussian_filter1d
from plotly.subplots import make_subplots
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as pyo

def get_osculating_circle(curve, t_index):
    # Handle edge cases
    if t_index == 0:
        t_index = 1
    if t_index == len(curve) - 1:
        t_index = len(curve) - 2

    # Central differences for first and second derivatives
    r_prime = (curve[t_index + 1] - curve[t_index - 1]) / 2
    r_double_prime = (curve[t_index + 1] - 2 * curve[t_index] + curve[t_index - 1])

    # Append a zero for 3D cross product
    r_prime_3d = np.append(r_prime, [0])
    r_double_prime_3d = np.append(r_double_prime, [0])
    
    # Curvature calculation and normal vector direction
    cross_product = np.cross(r_prime_3d, r_double_prime_3d)
    curvature = np.linalg.norm(cross_product) / np.linalg.norm(r_prime)**3
    signed_curvature = np.sign(cross_product[2])  # Sign of z-component of cross product
    radius_of_curvature = 1 / (curvature + 1e-12)
    
    # Unit tangent vector
    tangent = r_prime / np.linalg.norm(r_prime)

    # Unit normal vector, direction depends on the sign of the curvature
    if signed_curvature >= 0:
        norm_perp = np.array([-tangent[1], tangent[0]])  # Rotate tangent by 90 degrees counter-clockwise
    else:
        norm_perp = np.array([tangent[1], -tangent[0]])  # Rotate tangent by 90 degrees clockwise
    
    # Center of the osculating circle
    center = curve[t_index] + radius_of_curvature * norm_perp

    return center, radius_of_curvature

def gaussian_filter1d_variable_sigma(input, sigma, axis=-1, order=0, output=None,
                                     mode="reflect", cval=0.0, truncate=4.0):
    """1-D Gaussian filter with variable sigma.

    Parameters
    ----------
    input : array_like
        Input array to filter.
    sigma : scalar or sequence of scalar
        Standard deviation(s) for Gaussian kernel. If a sequence is provided,
        it must have the same length as the input array along the specified axis.
    axis : int, optional
        The axis of input along which to calculate. Default is -1.
    order : int, optional
        An order of 0 corresponds to convolution with a Gaussian kernel.
        A positive order corresponds to convolution with that derivative of a Gaussian.
    output : ndarray, optional
        Output array. Has the same shape as `input`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the input array is extended beyond its boundaries.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'.
    truncate : float, optional
        Truncate the filter at this many standard deviations.

    Returns
    -------
    output : ndarray
        The result of the 1D Gaussian filter with variable sigma.
    """

    if input.ndim == 0:
        raise ValueError("Input array should have at least one dimension")

    if np.isscalar(sigma):
        sigma = np.ones(input.shape[axis]) * sigma
    elif len(sigma) != input.shape[axis]:
        raise ValueError("Length of sigma must match the dimension of the input array along the specified axis.")

    # Move the specified axis to the front
    input = np.moveaxis(input, axis, 0)
    
    # Define the output array if not provided
    if output is None:
        output = np.zeros_like(input)

    # Iterate over each position along the specified axis
    for i in range(input.shape[0]):
        # Extract the local sigma value
        local_sigma = sigma[i]
        lw = int(truncate * local_sigma + 0.5)
        min_i = max(0, i - lw)
        max_i = min(input.shape[0], i + lw + 1)
        # Generate the local weights for the Gaussian kernel
        output[i] = gaussian_filter1d(input[min_i:max_i], local_sigma, axis=axis, order=order, mode=mode, cval=cval, truncate=truncate)[i - min_i]
        # Apply the local filter

    # Move the axis back to its original position
    output = np.moveaxis(output, 0, axis)

    return output

def to_color_string(color):
    # return (256 * color[0], 256 * color[1], 256 * color[2], color[3])
    return f"rgb({int(256 * color[0])}, {int(256 * color[1])}, {int(256 * color[2])}, {color[3]})"

skip = 3
add = 1
checkpoint_directory = "/workspace/CraftaxDevinterp/intermediate"
checkpoint_list = os.listdir(checkpoint_directory)
num_models = len(checkpoint_list)
if num_models % skip == 0: add = 0
# Load and process data
save_dir = "/workspace/CraftaxDevinterp/essential_dynamics"
logits_matrix = np.zeros((num_models // skip + add, 2000, 64, 17))
for modelno in tqdm.tqdm(range(0, num_models, skip)):
    with open(f"{save_dir}/{modelno}/logits.pkl", "rb") as f:
        logits = pickle.load(f)
    logits_matrix[modelno // skip, :, :, :] = logits

# Reshape for PCA
logits_matrix = logits_matrix.reshape(num_models // skip + add, 2000 * 64 * 17)
outputs = logits_matrix

pca = PCA(n_components=10)
pca.fit(outputs)
pca_outputs = pca.transform(outputs)
pca_outputs.shape

start, end = 0.1, 100
pca_outputs_smoothed = gaussian_filter1d_variable_sigma(pca_outputs, np.linspace(start, end, len(pca_outputs)), axis=0)

labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

cmap1 = sns.color_palette("Spectral", as_cmap=True)
cmap2 = sns.color_palette("viridis", as_cmap=True)

num_components = 7
subplot_titles = []
fig = make_subplots(rows=num_components, cols=num_components, subplot_titles=subplot_titles)
colors = np.array([to_color_string(cmap1(c)) for c in np.linspace(0, 1, len(pca_outputs_smoothed)-4)])
colors2 = np.array([to_color_string(cmap2(c)) for c in np.linspace(0, 1, len(pca_outputs_smoothed)-4)])

for i, j in tqdm.tqdm(itertools.product(range(num_components), range(num_components)), total=num_components ** 2): 
    row, col = i + 1, j + 1
        
    ymin, ymax = (
        pca_outputs[:, i].min(),
        pca_outputs[:, i].max(),
    )
    xmin, xmax = (
        pca_outputs[:, j].min(),
        pca_outputs[:, j].max(),
    )

    # # Forms
    # for f, form in enumerate(forms):
    #     if form[j] is not None:
    #         # Vertical line
    #         fig.add_shape(
    #             type="line",
    #             x0=form[j],
    #             y0=ymin * 1.25,
    #             x1=form[j],
    #             y1=ymax * 1.25,
    #             line=dict(color=form_colors[f], width=1),
    #             row=row,
    #             col=col,
    #         )
    #     if form[i] is not None:
    #         # Horizontal line
    #         fig.add_shape(
    #             type="line",
    #             x0=xmin * 1.25,
    #             y0=form[i],
    #             x1=xmax * 1.25,
    #             y1=form[i],
    #             line=dict(color=form_colors[f], width=1),
    #             row=row,
    #             col=col,
    #         )

    ts = np.array(range(2, len(pca_outputs_smoothed) - 2))
    centers = np.zeros((len(ts), 2))

    # Circles
    for ti, t in enumerate(ts):
        center, radius = get_osculating_circle(
            pca_outputs_smoothed[:, (j, i)], t
        )
        # if ti % 16 == 0:
        #     # This seems to be cheaper than directly plotting a circle
        #     circle = go.Scatter(
        #         x=center[0] + radius * np.cos(np.linspace(0, 2 * np.pi, 100)),
        #         y=center[1] + radius * np.sin(np.linspace(0, 2 * np.pi, 100)),
        #         mode="lines",
        #         line=dict(color="rgba(0.1, 0.1, 1, 0.05)", width=1),
        #         showlegend=False,
        #     )
        #     fig.add_trace(circle, row=row, col=col)

        centers[ti] = center

    # Centers
    timestamps = range(0, num_models, skip)
    timestamp_labels = [f"Timestamp: {t}" for t in timestamps]
    fig.add_trace(
        go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode="markers",
            marker=dict(size=2, symbol="x", color=colors2),
            name="Centers",
            text = timestamp_labels, 
            hoverinfo = "text+x+y"
        ),
        row=row,
        col=col,
    )

    # Original samples
    # fig.add_trace(
    #     go.Scatter(
    #         x=pca_outputs[:, j],
    #         y=pca_outputs[:, i],
    #         mode="markers",
    #         marker=dict(color=colors, size=3),
    #         showlegend=False,
    #     ),
    #     row=row,
    #     col=col,
    # )

    # Smoothed trajectory
    fig.add_trace(
        go.Scatter(
            x=pca_outputs_smoothed[:, j],
            y=pca_outputs_smoothed[:, i],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    if j == 0:
        fig.update_yaxes(title_text=labels[str(i)], row=row, col=col)

    fig.update_xaxes(title_text=labels[str(j)], row=row, col=col)

    fig.update_xaxes(
        range=(xmin * 1.25, xmax * 1.25),
        row=row,
        col=col,
    )
    fig.update_yaxes(
        range=(ymin * 1.25, ymax * 1.25),
        row=row,
        col=col,
    )

fig.update_layout(width=2500, height=2500)  # Adjust the size as needed
fig.update_layout(title_text=f"ED (Unnormalized samples, linear post-smoothing {start} to {end})", showlegend=False)

# pyo.plot(fig, filename=str(FIGURES / MODEL_ID / "pca.html"))

fig.show()