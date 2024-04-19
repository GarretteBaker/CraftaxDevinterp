#%%
import jax
import jax.numpy as jnp
from typing import NamedTuple
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import scipy as sp
import numpy as np
from copy import copy
#%%

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
def get_average_quantity(quantity, dones, dim):
    assert type(quantity) is sp.sparse._coo.coo_array
    quantity = quantity.toarray()
    quantity = quantity.reshape(5000, 64, dim)
    quantity = quantity.transpose((1, 0, 2))
    dones = jax.device_get(dones).transpose((1,0))
    dones = copy(dones)
    dones[:, -1] = True

    quantity = quantity.reshape(64*5000, dim)
    dones = dones.reshape(64*5000)
    totals = list()
    s = np.zeros((dim))
    for done, step in zip(dones, quantity):
        if not done:
            s += step
        if done:
            totals.append(s)
            s = np.zeros((dim))
    totals = np.array(totals)
    average = np.mean(totals, axis=0)
    return average
        
#%%
load_dir = "/workspace/CraftaxDevinterp/ExperimentData/trackers"
block_placement_averages = np.zeros((1525, 4))
block_mining_averages = np.zeros((1525, 13))
player_location_averages = np.zeros((1525, 4))
player_do_averages = np.zeros((1525, 4))
mob_kill_averages = np.zeros((1525, 3))
mob_attack_averages = np.zeros((1525, 3))

for modelno in tqdm(range(0, 1525)):
    with open(f"{load_dir}/{modelno}/trajectory.pkl", "rb") as f:
        trajectory = pickle.load(f)
    with open(f"{load_dir}/{modelno}/done.pkl", "rb") as f:
        dones = pickle.load(f)
    
    block_placement_averages = get_average_quantity(trajectory.block_placements, dones, 4)
    block_placement_averages[modelno, :] = block_placement_averages
    block_mining_averages = get_average_quantity(trajectory.block_mining, dones, 13)
    block_mining_averages[modelno, :] = block_mining_averages
    player_location_averages = get_average_quantity(trajectory.player_location, dones, 4)
    player_location_averages[modelno, :] = player_location_averages
    player_do_averages = get_average_quantity(trajectory.doings, dones, 4)
    player_do_averages[modelno, :] = player_do_averages
    mob_kill_averages = get_average_quantity(trajectory.mob_kills, dones, 3)
    mob_kill_averages[modelno, :] = mob_kill_averages
    mob_attack_averages = get_average_quantity(trajectory.mob_attacks, dones, 3)
    mob_attack_averages[modelno, :] = mob_attack_averages

with open(f"{load_dir}/block_placement_averages.pkl", "wb") as f:
    pickle.dump(block_placement_averages, f)
with open(f"{load_dir}/block_mining_averages.pkl", "wb") as f:
    pickle.dump(block_mining_averages, f)
with open(f"{load_dir}/player_location_averages.pkl", "wb") as f:
    pickle.dump(player_location_averages, f)
with open(f"{load_dir}/player_do_averages.pkl", "wb") as f:
    pickle.dump(player_do_averages, f)
with open(f"{load_dir}/mob_kill_averages.pkl", "wb") as f:
    pickle.dump(mob_kill_averages, f)   
with open(f"{load_dir}/mob_attack_averages.pkl", "wb") as f:
    pickle.dump(mob_attack_averages, f)
