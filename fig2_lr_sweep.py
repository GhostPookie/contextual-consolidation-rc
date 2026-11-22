"""Fig 2 (Phase-2): LoRA lr-fragility vs lr-free adapter band.

x: learning rate (log). LoRA U-curves for gpt2-medium / gpt2-large;
adapter has no lr -> flat shaded band = its onset range across scales.
Data: spike_results.json[results][model][lora][lr_*] / [apia].
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})

D = json.load(open("spike_results.json"))["results"]
MED, LRG = "openai-community/gpt2-medium", "openai-community/gpt2-large"
LRS = [1e-5, 1e-4, 1e-3]


def sweep(mid):
    l = D[mid]["lora"]
    keys = {1e-3: "lr_1e-3", 1e-4: "lr_1e-4", 1e-5: "lr_1e-5"}
    return [l.get(keys[lr], {}).get("p3_onset", np.nan) for lr in LRS]


med, lrg = sweep(MED), sweep(LRG)
band_lo = min(D[m]["apia"]["p3_onset"] for m in (MED, LRG, "openai-community/gpt2"))
band_hi = max(D[m]["apia"]["p3_onset"] for m in (MED, LRG, "openai-community/gpt2"))

fig, ax = plt.subplots(figsize=(5.6, 3.4))
fig.subplots_adjust(left=.11, right=.97, top=.82, bottom=.14)

ax.axhspan(band_lo, band_hi, color="tab:blue", alpha=.18)
ax.axhline((band_lo + band_hi) / 2, color="tab:blue", lw=1.6)
ax.text(1.15e-5, band_hi + .015, f"APIA: no lr — flat {band_lo:.2f}–{band_hi:.2f} across 124M–774M",
        color="tab:blue", fontsize=8)

ax.plot(LRS, med, "o--", color="tab:orange", lw=1.5, label="LoRA gpt2-medium (355M)")
ax.plot(LRS, lrg, "s-", color="tab:red", lw=1.5, label="LoRA gpt2-large (774M)")

ax.annotate("divergence", xy=(1e-3, lrg[2]), xytext=(3e-4, .58),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=.8))
ax.annotate("optimal — but the window\nmoves with scale", xy=(1e-4, lrg[1]),
            xytext=(2.2e-4, .30), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=.8))

ax.set_xscale("log")
ax.set_xticks(LRS); ax.set_xticklabels(["1e-5", "1e-4", "1e-3"])
ax.set_xlabel("learning rate (log)"); ax.set_ylabel("return-onset error")
ax.set_ylim(0, .62)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("the tuning window: LoRA needs lr found per scale —\n"
             "APIA has no lr to find  (domain-return stream, single pass)",
             fontsize=9.5)

fig.savefig("fig2_lr_sweep.png", dpi=170)
print("saved fig2_lr_sweep.png")
