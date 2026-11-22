import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})

D = json.load(open("spike_results.json"))["vram_scaling"]["by_batch"]
B = sorted(int(k) for k in D)
apia_pk = [D[str(b)]["apia"]["peak"] if "apia" in D[str(b)] else np.nan for b in B]
lora_pk = [D[str(b)]["lora"]["peak"] if "lora" in D[str(b)] else np.nan for b in B]
apia_ms = [D[str(b)]["apia"]["ms"] if "apia" in D[str(b)] else np.nan for b in B]
lora_ms = [D[str(b)]["lora"]["ms"] if "lora" in D[str(b)] else np.nan for b in B]
WALL = 6144

fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.3))
fig.subplots_adjust(left=.08, right=.985, top=.80, wspace=.26)

X = np.arange(len(B))
for ax in axs:
    ax.set_xticks(X); ax.set_xticklabels(B)

a = axs[0]
a.axhline(WALL, color="red", ls="--", lw=1.6)
a.text(X[1] - .1, WALL + 130, "6GB hardware wall", color="red", fontsize=8.5)
a.plot(X, lora_pk, "o-", color="tab:orange", lw=1.6, label="LoRA (fwd+bwd+AdamW)")
a.plot(X, apia_pk, "s-", color="tab:blue", lw=1.6, label="APIA (fwd-only + local RLS)")
a.annotate("spill to shared RAM", xy=(X[2], lora_pk[2]), xytext=(X[2] - .55, lora_pk[2] - 1400),
           fontsize=8, arrowprops=dict(arrowstyle="->", lw=.8))
a.set_xlabel("batch size"); a.set_ylabel("peak VRAM (MB)")
a.set_title("gpt2-large: peak memory vs batch", fontsize=9.5)
a.legend(frameon=False, fontsize=8, loc="upper left")
a.set_ylim(2800, 9800)

b = axs[1]
i4 = B.index(4)
b.plot(X[i4:i4+2], lora_ms[i4:i4+2], "o-", color="tab:orange", lw=1.6, label="LoRA")
b.plot(X[i4:], apia_ms[i4:], "s-", color="tab:blue", lw=1.6, label="APIA")
b.annotate("not compute:\nmemory-spill overhead", xy=(X[2], lora_ms[2]), xytext=(X[1] - .35, 1500),
           fontsize=8, arrowprops=dict(arrowstyle="->", lw=.8))
b.set_xlabel("batch size"); b.set_ylabel("ms per batch step")
b.set_title("step time once over the wall", fontsize=9.5)
b.legend(frameon=False, fontsize=8)
b.set_yscale("log")
b.set_ylim(12, 3500)

fig.suptitle("the scaling wall: backprop memory diverges, APIA stays flat\n"
             "frozen gpt2-large, seq-128, 6GB device - spike: vram_spike.py", fontsize=10)
fig.savefig("fig_vram_wall.png", dpi=170)
print("saved fig_vram_wall.png")
