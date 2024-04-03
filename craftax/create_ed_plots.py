#%%
import os
import pickle
import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

#%%
with open("/workspace/Craftax/projected_policies/projected_policies.pkl", "rb") as f:
    projected_policies = pickle.load(f)
shape = projected_policies.shape
reshaped = projected_policies.reshape(shape[0]*shape[1]*shape[2], shape[3]*shape[4])
#%%
num_pcs = 3
pca = PCA(n_components=num_pcs)
pca.fit(reshaped)
projected = pca.transform(reshaped)

# %%
p1 = projected[:, 0]
p2 = projected[:, 1]

plt.plot(p1, p2)
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.show()

#%%
p1 = projected[:, 0]
p2 = projected[:, 2]

plt.plot(p1, p2)
plt.xlabel("pc1")
plt.ylabel("pc3")
plt.show()

#%%
p1 = projected[:, 1]
p2 = projected[:, 2]

plt.plot(p1, p2)
plt.xlabel("pc2")
plt.ylabel("pc3")
plt.show()


# %%
plt.plot(projected[:, :10]/projected[:, :10].max(axis=0))

#%%
df = pd.read_csv("achievement_data.csv")
combos = list(combinations(range(num_pcs), 2))
for achievement_name in tqdm(df):
    if achievement_name == "Step" or "MAX" in achievement_name or "MIN" in achievement_name:
        continue
    achievement_datum = df[achievement_name]
    if np.array(achievement_datum)[-1] == 0:
        continue
    fig, ax = plt.subplots(1, len(combos), figsize=(3*50, 50))
    fig.suptitle(f"pc plot colored by progress in {achievement_name}", fontsize = 150)
    for i, (pcx, pcy) in enumerate(combos):
        pc1 = projected[::4, pcx]
        pc2 = projected[::4, pcy]

        ax[i].scatter(pc1, pc2, s=500, cmap="seismic", c=np.array(achievement_datum))
        # ax[i].xlabel(f"pc {pcx+1}")
        # ax[i].ylabel(f"pc {pcy+1}")
    directory = os.path.dirname(f"/workspace/Craftax/plots/100M/achievements/{achievement_name}.png")
    os.makedirs(directory, exist_ok=True)
    
    plt.savefig(f"/workspace/Craftax/plots/100M/achievements/{achievement_name}.png")
    plt.close()
    # break

# %%
