#%%
from craftax.craftax.tree_water_experiment import restricted_ed, generate_test_world
import jax
import matplotlib.pyplot as plt
from itertools import combinations

num_pcs = 3

num_wood = 0
num_water = 9
crafting_table = False
pickaxe = 2
sword = 0
torch = 0
placed_torch = True
num_coal = 0
num_iron = 0
num_diamond = 0
num_sapling = 0
num_bow = 0
num_arrows = 0
num_stone = 0

rng = jax.random.PRNGKey(seed=0)
env_state = generate_test_world(
    rng, 
    num_wood = num_wood, 
    num_water = num_water, 
    crafting_table = crafting_table, 
    pickaxe = pickaxe, 
    sword = sword, 
    torch = torch,
    placed_torch=placed_torch,
    num_stone=num_stone, 
    num_coal = num_coal, 
    num_iron = num_iron, 
    num_diamond= num_diamond, 
    num_sapling = num_sapling, 
    num_bow = num_bow, 
    num_arrows = num_arrows
)
projected, _ = restricted_ed(env_state, num_pcs = num_pcs)
#%%
import matplotlib.cm as cm
import numpy as np
combos = list(combinations(range(num_pcs), 2))
fig, ax = plt.subplots(1, len(combos), figsize=(3*5, 5))
fig.suptitle(f"{num_wood} wood, {num_water} water, {torch} torches, crafting table: {crafting_table}, pickaxe: {pickaxe}, sword: {sword}, placed torch: {placed_torch}")
colors = cm.rainbow(np.linspace(0, 1, 1525))
for i, (pcx, pcy) in enumerate(combos):
    pc1 = projected[:, pcx]
    pc2 = projected[:, pcy]
    sc = ax[i].scatter(pc1, pc2, s=0.5, c=np.arange(1525), cmap="rainbow")
    ax[i].set_xlabel(f"pc {pcx}")
    ax[i].set_ylabel(f"pc {pcy}")
cbar = plt.colorbar(sc, ax=ax[-1])
plt.show()
# %%
