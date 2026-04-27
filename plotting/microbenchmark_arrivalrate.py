# %%
import re
import matplotlib.pyplot as plt
import numpy as np
import os

# Resolve absolute paths based on this script's location
current_dir = os.path.dirname(os.path.abspath(__file__))
marconi_root = os.path.dirname(current_dir)
logs_dir = os.path.join(marconi_root, "logs")
figures_dir = os.path.join(marconi_root, "figures", "eval")
os.makedirs(figures_dir, exist_ok=True)

CAPACITY_GB = 5.0
log_path = os.path.join(logs_dir, "swebench.txt")

with open(log_path, "r") as f:
    log_data = f.read()

entries = [e for e in log_data.split("=" * 50) if e.strip()]

cache_size_pattern = re.compile(r"Cache size ([\d\.]+) GB")
trace_pattern = re.compile(r"swebench_sps=([\d\.]+)_art=([\d\.]+)_nums=\d+\.jsonl")
hit_rate_pattern = re.compile(r"^(V1|V2): hit rate ([\d\.]+)%", re.MULTILINE)

# keyed by (sps, art) -> {"SGLang+": float, "Marconi": float}
results = {}

for entry in entries:
    cache_match = cache_size_pattern.search(entry)
    if not cache_match or float(cache_match.group(1)) != CAPACITY_GB:
        continue
    trace_match = trace_pattern.search(entry)
    if not trace_match:
        continue
    sps = float(trace_match.group(1))
    art = float(trace_match.group(2))
    hit_rates = {m.group(1): float(m.group(2)) for m in hit_rate_pattern.finditer(entry)}
    if "V1" in hit_rates and "V2" in hit_rates:
        results[(sps, art)] = {"SGLang+": hit_rates["V1"], "Marconi": hit_rates["V2"]}


# --- Fig 13a: vary art, fixed sps=5 ---
SPS_13A = 5
arts = sorted({art for (sps, art) in results if sps == SPS_13A})
hitrate_13a = {
    "Marconi": tuple(results[(SPS_13A, art)]["Marconi"] for art in arts if (SPS_13A, art) in results),
    "SGLang+": tuple(results[(SPS_13A, art)]["SGLang+"] for art in arts if (SPS_13A, art) in results),
}
art_labels = [str(a) for a in arts if (SPS_13A, a) in results]

print(f"Fig 13a — sps={SPS_13A}, arts={art_labels}")
for scheme, rates in hitrate_13a.items():
    print(f"  {scheme}: {rates}")

colors = {"Marconi": "#2D6A4F", "SGLang+": "#52B788"}
fontsize = 14

x = np.arange(len(art_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(4, 2.7), layout="constrained")
for multiplier, (scheme, hitrate) in enumerate(hitrate_13a.items()):
    ax.bar(x + width * multiplier, hitrate, width, label=scheme, color=colors[scheme])
    if scheme == "SGLang+":
        for i, rate in enumerate(hitrate):
            diff = hitrate_13a["Marconi"][i] / rate
            ax.text(i + width * multiplier, rate + 2, f"{diff:.1f}×", rotation=90, fontsize=fontsize - 2)

ax.set_ylabel("Token Hit Rate (%)", fontsize=fontsize)
ax.set_xticks(x + width / 2, art_labels)
ax.tick_params(axis="both", which="major", labelsize=fontsize)
ax.set_xlabel("Avg Response Time (s)", fontsize=fontsize)
ax.legend(loc="upper center", ncols=3, fontsize=fontsize, bbox_to_anchor=(0.5, 1.2), columnspacing=0.8, handlelength=0.8, frameon=False, borderaxespad=0)
ax.set_axisbelow(True)
ax.grid(color="lightgrey", linestyle="dashed", axis="y", linewidth=0.8)

plt.show()
fig.savefig(os.path.join(figures_dir, "fig13a_microbenchmark_art.pdf"), dpi=500, bbox_inches="tight")


# --- Fig 13b: vary sps, fixed art=7.5 ---
ART_13B = 7.5
spss = sorted({sps for (sps, art) in results if art == ART_13B})
hitrate_13b = {
    "Marconi": tuple(results[(sps, ART_13B)]["Marconi"] for sps in spss if (sps, ART_13B) in results),
    "SGLang+": tuple(results[(sps, ART_13B)]["SGLang+"] for sps in spss if (sps, ART_13B) in results),
}
sps_labels = [str(int(s) if s == int(s) else s) for s in spss if (s, ART_13B) in results]

print(f"\nFig 13b — art={ART_13B}, sps={sps_labels}")
for scheme, rates in hitrate_13b.items():
    print(f"  {scheme}: {rates}")

x = np.arange(len(sps_labels))

fig, ax = plt.subplots(figsize=(4, 2.7), layout="constrained")
for multiplier, (scheme, hitrate) in enumerate(hitrate_13b.items()):
    ax.bar(x + width * multiplier, hitrate, width, label=scheme, color=colors[scheme])
    if scheme == "SGLang+":
        for i, rate in enumerate(hitrate):
            diff = hitrate_13b["Marconi"][i] / rate
            ax.text(i + width * multiplier, rate + 3, f"{diff:.1f}×", rotation=90, fontsize=fontsize - 2)

ax.set_ylabel("Token Hit Rate (%)", fontsize=fontsize)
ax.set_xticks(x + width / 2, sps_labels)
ax.tick_params(axis="both", which="major", labelsize=fontsize)
ax.set_xlabel("Num Sessions per Second", fontsize=fontsize)
ax.legend(loc="upper center", ncols=3, fontsize=fontsize, bbox_to_anchor=(0.5, 1.2), columnspacing=0.8, handlelength=0.8, frameon=False, borderaxespad=0)
ax.set_axisbelow(True)
ax.grid(color="lightgrey", linestyle="dashed", axis="y", linewidth=0.8)

plt.show()
fig.savefig(os.path.join(figures_dir, "fig13b_microbenchmark_sps.pdf"), dpi=500, bbox_inches="tight")
# %%
