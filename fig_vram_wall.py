import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})

D = json.load(open("spike_results.json"))["vram_scaling"]["by_batch"]
B = sorted(int(k) for k in D)
apia_pk = [D[str(b)]["apia"]["peak"] for b in B]
lora_pk = [D[str(b)]["lora"]["peak"] for b in B]
apia_ms = [D[str(b)]["apia"]["ms"] for b in B]
lora_ms = [D[str(b)]["lora"]["ms"] for b in B]
WALL = 6144

fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.3))
fig.subplots_adjust(left=.08, right=.985, top=.80, wspace=.26)

for ax in axs:
    ax.set_xticks(B)

a = axs[0]
a.axhline(WALL, color="red", ls="--", lw=1.6)
a.text(5.1, WALL + 130, "6GB hardware wall", color="red", fontsize=8.5)
a.plot(B, lora_pk, "o-", color="tab:orange", lw=1.6, label="LoRA (fwd+bwd+AdamW)")
a.plot(B, apia_pk, "s-", color="tab:blue", lw=1.6, label="APIA (fwd-only + local RLS)")
a.annotate("spill to shared RAM", xy=(8, lora_pk[-1]), xytext=(5.5, lora_pk[-1] - 1350),
           fontsize=8, arrowprops=dict(arrowstyle="->", lw=.8))
a.set_xlabel("batch size"); a.set_ylabel("peak VRAM (MB)")
a.set_title("gpt2-large: peak memory vs batch", fontsize=9.5)
a.legend(frameon=False, fontsize=8, loc="upper left")
a.set_ylim(2800, 9800)

b = axs[1]
i4 = B.index(4)
b.plot(B[i4:], lora_ms[i4:], "o-", color="tab:orange", lw=1.6, label="LoRA")
b.plot(B[i4:], apia_ms[i4:], "s-", color="tab:blue", lw=1.6, label="APIA")
b.annotate("not compute:\nmemory-spill overhead", xy=(8, lora_ms[-1]), xytext=(4.7, lora_ms[-1] - 120),
           fontsize=8, arrowprops=dict(arrowstyle="->", lw=.8))
b.set_xlabel("batch size"); b.set_ylabel("ms per batch step")
b.set_title("step time once over the wall", fontsize=9.5)
b.legend(frameon=False, fontsize=8)
b.set_yscale("log")

fig.suptitle("the scaling wall: backprop memory diverges, APIA stays flat\n"
             "frozen gpt2-large, seq-128, 6GB device - spike: vram_spike.py", fontsize=10)
fig.savefig("fig_vram_wall.png", dpi=170)
print("saved fig_vram_wall.png")
