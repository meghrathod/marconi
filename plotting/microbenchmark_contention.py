# %%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# Resolve absolute paths based on this script's location
current_dir = os.path.dirname(os.path.abspath(__file__))
marconi_root = os.path.dirname(current_dir)
results_dir = os.path.join(marconi_root, "results")
figures_dir = os.path.join(marconi_root, "figures", "eval")
os.makedirs(figures_dir, exist_ok=True)

# Read hit rates from pickle files for sps=5, art=10 across all cache sizes
SPS = 5
ART = 10

capacity_bytes_list = [1e9, 5e9, 1e10, 2e10, 4e10, 6e10, 8e10, 1e11]

results = {}  # cache_gb -> {"SGLang+": float, "Marconi": float}

for cap in capacity_bytes_list:
    gb = int(cap / 1e9)
    subdir = os.path.join(results_dir, str(cap), f"swebench_sps={SPS}_art={ART}_nums=100")
    if not os.path.exists(subdir):
        continue
    files = sorted(f for f in os.listdir(subdir) if f.endswith(".pickle"))
    if not files:
        continue
    latest = os.path.join(subdir, files[-1])
    with open(latest, "rb") as f:
        d = pickle.load(f)
    results[gb] = {
        "SGLang+": d["v1_max_hit_rate"] * 100,
        "Marconi": d["v2_max_hit_rate"] * 100,
    }

cache_sizes_sorted = sorted(results)
cache_size_labels = [str(gb) for gb in cache_sizes_sorted]
hitrate_dict = {
    "SGLang+": tuple(results[gb]["SGLang+"] for gb in cache_sizes_sorted),
    "Marconi": tuple(results[gb]["Marconi"] for gb in cache_sizes_sorted),
}

print(f"Loaded {len(cache_sizes_sorted)} cache sizes: {cache_size_labels}")
for scheme, rates in hitrate_dict.items():
    print(f"  {scheme}: {[f'{r:.1f}%' for r in rates]}")

colors = {"Marconi": "#2D6A4F", "SGLang+": "#52B788", "vLLM+": "#95D5B2"}
fontsize = 14

x = np.arange(len(cache_size_labels))
width = 0.25
multiplier = 0

fig, ax = plt.subplots(figsize=(4, 2.7), layout="constrained")

for scheme, hitrate in hitrate_dict.items():
    offset = width * multiplier
    ax.bar(x + offset, hitrate, width, label=scheme, color=colors[scheme])
    multiplier += 1

ax.set_ylabel('Token Hit Rate (%)', fontsize=fontsize)
ax.set_xticks(x + width / 2, cache_size_labels)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.set_xlabel("Cache Size (GB)", fontsize=fontsize)
ax.legend(loc="upper center", ncols=3, fontsize=fontsize, bbox_to_anchor=(0.5, 1.2), columnspacing=0.8, handlelength=0.8, frameon=False, borderaxespad=0)
ax.set_axisbelow(True)
ax.grid(color='lightgrey', linestyle='dashed', axis="y", linewidth=0.8)

plt.show()
fig.savefig(os.path.join(figures_dir, "fig11_microbenchmark_contention.pdf"), dpi=500, bbox_inches='tight')
# %%
