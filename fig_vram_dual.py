import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})

R = json.load(open("spike_results.json"))
D6 = R["vram_scaling"]["by_batch"]
D16 = R["vram_scaling_opt13b"]["by_batch"]


def unpack(D):
    B = sorted(int(k) for k in D)
    apia = [D[str(b)].get("apia", {}).get("peak", np.nan) for b in B]
    lora = [D[str(b)].get("lora", {}).get("peak", np.nan) for b in B]
    return B, np.array(apia, float), np.array(lora, float)


B6, A6, L6 = unpack(D6)
B16, A16, L16 = unpack(D16)

fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4))
fig.subplots_adjust(left=.08, right=.985, top=.78, wspace=.24)


def panel(ax, B, apia, lora, wall, title, wall_label, wall_text_x, dy=320):
    X = np.arange(len(B))
    ax.set_xticks(X)
    ax.set_xticklabels(B)
    ax.axhline(wall, color="red", ls="--", lw=1.6)
    ax.text(wall_text_x, wall + dy, wall_label, color="red", fontsize=9.5,
            ha="right")
    ok = lora > 0
    ax.plot(X[ok], lora[ok], "o-", color="tab:orange", lw=1.6, label="LoRA (fwd+bwd+AdamW)")
    if (~ok).any():
        ax.scatter(X[~ok], [wall] * (~ok).sum(), marker="X", s=90, color="tab:orange",
                   edgecolor="red", zorder=5, label="LoRA OOM")
    ax.plot(X, apia, "s-", color="tab:blue", lw=1.6, label="APIA (fwd-only + local RLS)")
    ax.set_xlabel("batch size")
    ax.set_title(title, fontsize=11)


panel(axs[0], B6, A6, L6, 6144, "gpt2-large (774M) - 6GB device",
      "6GB hardware wall", wall_text_x=len(B6) - 1.2)
axs[0].set_ylabel("peak VRAM (MB)")
axs[0].set_ylim(2800, 10000)
axs[0].legend(frameon=False, fontsize=9, loc="upper left")

panel(axs[1], B16, A16, L16, 15636, "opt-1.3b fp16 - 16GB T4 (Kaggle)",
      "16GB hardware wall", wall_text_x=3.6, dy=-780)
axs[1].set_ylim(2000, 17500)
axs[1].annotate("wall moved, not removed", xy=(4.85, 15400), xytext=(1.0, 11200),
                fontsize=9, color="dimgray",
                arrowprops=dict(arrowstyle="->", lw=.8, color="dimgray"))
axs[1].legend(frameon=False, fontsize=9, loc="upper left")

fig.suptitle("the scaling wall at two hardware tiers\n"
             "frozen backbone, seq-128, peak VRAM vs batch - spike: vram_spike.py + Kaggle T4",
             fontsize=11.5)
fig.savefig("fig_vram_dual.png", dpi=170)
print("saved fig_vram_dual.png")
