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

PLACEMENT = [
    "Sone", 
    "Table", 
    "Furnace", 
    "Plant"
]

MINE = [
    "Tree", 
    "Stone", 
    "Furnace", 
    "Table", 
    "Coal", 
    "Iron", 
    "Diamond", 
    "Sapphire", 
    "Ruby", 
    "Plant", 
    "Stalagmite", 
    "Chest", 
    "None"
]

MOVEMENT = [
    "Up", 
    "Down", 
    "Left", 
    "Right"
]

DO = [
    "NOOP", 
    "Drink", 
    "Sleep", 
    "Rest"
]

KILL = [
    "Melee", 
    "Ranged", 
    "Passive"
]

ATTACK = [
    "Melee", 
    "Ranged", 
    "Passive"
]

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
        
def get_average_time(quantity, dones, dim):
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
    for done, step in zip(dones, quantity):
        if done:
            totals.append(step)
    totals = np.array(totals)
    average = np.mean(totals, axis=0)
    return average

#%%
load_dir = "/workspace/CraftaxDevinterp/ExperimentData/trackers"

#%%
block_placement_averages = np.zeros((1525, 4))
block_mining_averages = np.zeros((1525, 13))
player_location_averages = np.zeros((1525, 2))
player_do_averages = np.zeros((1525, 4))
mob_kill_averages = np.zeros((1525, 3))
mob_attack_averages = np.zeros((1525, 3))
time_averages = np.zeros((1525, 1))

for modelno in tqdm(range(0, 1525)):
    with open(f"{load_dir}/{modelno}/trajectory.pkl", "rb") as f:
        trajectory = pickle.load(f)
    with open(f"{load_dir}/{modelno}/done.pkl", "rb") as f:
        dones = pickle.load(f)

    # print(trajectory.block_placements.shape)
    # print(trajectory.block_mining.shape)
    # print(trajectory.player_location.shape)
    # print(trajectory.doings.shape)
    # print(trajectory.mob_kills.shape)
    # print(trajectory.mob_attacks.shape)

    
    # block_placement_average = get_average_quantity(trajectory.block_placements, dones, 4)
    # block_placement_averages[modelno, :] = block_placement_average
    # block_mining_average = get_average_quantity(trajectory.block_mining, dones, 13)
    # block_mining_averages[modelno, :] = block_mining_average
    # player_location_average = get_average_quantity(trajectory.player_location, dones, 2)
    # player_location_averages[modelno, :] = player_location_average
    # player_do_average = get_average_quantity(trajectory.doings, dones, 4)
    # player_do_averages[modelno, :] = player_do_average
    # mob_kill_average = get_average_quantity(trajectory.mob_kills, dones, 3)
    # mob_kill_averages[modelno, :] = mob_kill_average
    # mob_attack_average = get_average_quantity(trajectory.mob_attacks, dones, 3)
    # mob_attack_averages[modelno, :] = mob_attack_average
    time_average = get_average_time(trajectory.time, dones, 1)
    time_averages[modelno, :] = time_average


# with open(f"{load_dir}/block_placement_averages.pkl", "wb") as f:
#     pickle.dump(block_placement_averages, f)
# with open(f"{load_dir}/block_mining_averages.pkl", "wb") as f:
#     pickle.dump(block_mining_averages, f)
# with open(f"{load_dir}/player_location_averages.pkl", "wb") as f:
#     pickle.dump(player_location_averages, f)
# with open(f"{load_dir}/player_do_averages.pkl", "wb") as f:
#     pickle.dump(player_do_averages, f)
# with open(f"{load_dir}/mob_kill_averages.pkl", "wb") as f:
#     pickle.dump(mob_kill_averages, f)   
# with open(f"{load_dir}/mob_attack_averages.pkl", "wb") as f:
#     pickle.dump(mob_attack_averages, f)
with open(f"{load_dir}/time_averages.pkl", "wb") as f:
    pickle.dump(time_averages, f)

#%%
load_dir = "/workspace/CraftaxDevinterp/ExperimentData/trackers"
t = np.arange(1525)
with open(f"{load_dir}/block_placement_averages.pkl", "rb") as f:
    data = pickle.load(f)
plt.stackplot(t, *[data[:, i] for i in range(4)], labels=PLACEMENT)
plt.title("Average block placements")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

with open(f"{load_dir}/block_mining_averages.pkl", "rb") as f:
    data = pickle.load(f)
plt.stackplot(t, *[data[:, i] for i in range(13)], labels=MINE)
plt.title("Average mining amounts")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  
plt.show()
with open(f"{load_dir}/player_do_averages.pkl", "rb") as f:
    data = pickle.load(f)
plt.stackplot(t, *[data[:, i] for i in range(4)], labels=DO)
plt.title("Average player actions")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

with open(f"{load_dir}/mob_kill_averages.pkl", "rb") as f:
    data = pickle.load(f)
plt.stackplot(t, *[data[:, i] for i in range(3)], labels=KILL)
plt.title("Average number of mob kills")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

with open(f"{load_dir}/mob_attack_averages.pkl", "rb") as f:
    data = pickle.load(f)
plt.stackplot(t, *[data[:, i] for i in range(3)], labels=ATTACK)
plt.title("Average number of mob attacks")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
#%%

with open(f"{load_dir}/block_placement_averages.pkl", "rb") as f:
    data = pickle.load(f)
plt.plot(data)
plt.title("Average block placements")
plt.legend(PLACEMENT, bbox_to_anchor=(1,1))
plt.show()

with open(f"{load_dir}/block_mining_averages.pkl", "rb") as f:
    data = pickle.load(f)
plt.plot(data)
plt.title("Average mining amounts")
plt.legend(MINE, bbox_to_anchor=(1,1))
plt.show()

with open(f"{load_dir}/player_do_averages.pkl", "rb") as f:
    data = pickle.load(f)
plt.plot(data)
plt.title("Average player actions")
plt.legend(DO, bbox_to_anchor=(1,1))
plt.show()

with open(f"{load_dir}/mob_kill_averages.pkl", "rb") as f:
    data = pickle.load(f)
plt.plot(data)
plt.title("Average number of mob kills")
plt.legend(KILL, bbox_to_anchor=(1,1))
plt.show()

with open(f"{load_dir}/mob_attack_averages.pkl", "rb") as f:
    data = pickle.load(f)
plt.plot(data)
plt.title("Average number of mob attacks")
plt.legend(ATTACK, bbox_to_anchor=(1,1))
plt.show()

#%%
with open(f"{load_dir}/time_averages.pkl", "rb") as f:
    data = pickle.load(f)
plt.plot(data)
plt.title("Average time")
plt.show()

#%%
with open(f"{load_dir}/block_placement_averages.pkl", "rb") as f:
    block_placement_averages = pickle.load(f)
with open(f"{load_dir}/block_mining_averages.pkl", "rb") as f:
    block_mining_averages = pickle.load(f)
# with open(f"{load_dir}/player_location_averages.pkl", "rb") as f:
#     player_location_averages = pickle.load(f)
with open(f"{load_dir}/player_do_averages.pkl", "rb") as f:
    player_do_averages = pickle.load(f)
with open(f"{load_dir}/mob_kill_averages.pkl", "rb") as f:
    mob_kill_averages = pickle.load(f)
with open(f"{load_dir}/mob_attack_averages.pkl", "rb") as f:
    mob_attack_averages = pickle.load(f)
with open(f"{load_dir}/time_averages.pkl", "rb") as f:
    time_averages = pickle.load(f)
#%%
all_data = np.concatenate((block_placement_averages, block_mining_averages, player_do_averages, mob_kill_averages, mob_attack_averages), axis=1)
DATA_LABELS = PLACEMENT + MINE + DO + KILL + ATTACK
start = 100
end=200
u, s, vt = np.linalg.svd(all_data[start:end], full_matrices=False)
plt.imshow(vt[:, :])
plt.colorbar()
plt.xticks(range(len(DATA_LABELS)), DATA_LABELS, rotation=90)  
plt.show()
# %%
projected = u @ np.diag(s)
plt.scatter(projected[:, 0], projected[:, 1], c=range(1525), cmap = "rainbow", s=0.5)
plt.colorbar()
plt.show()

plt.scatter(projected[:, 0], projected[:, 2], c=range(1525), cmap = "rainbow", s=0.5)
plt.show()

plt.scatter(projected[:, 1], projected[:, 2], c=range(1525), cmap = "rainbow", s=0.5)
plt.show()

# %%
