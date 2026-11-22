"""fig_phase2_scale: 3-panel divergence plot from spike_results.json.

Left : P3-onset error vs model size (APIA flat band vs LoRA collapse;
       tuned-LoRA point shown honestly).
Mid  : peak VRAM, batch-1 streaming (APIA ~ weights+eps vs LoRA
       weights+backward+optimizer).
Right: update ms/step, log scale (scale-independent RLS vs growing
       LoRA step).
"""

import json
import numpy as np
import matplotlib.pyplot as plt

D = json.load(open("spike_results.json"))["results"]
MODELS = ["openai-community/gpt2", "openai-community/gpt2-medium",
          "openai-community/gpt2-large"]
XLAB = ["GPT-2\n124M", "GPT-2-M\n355M", "GPT-2-L\n774M"]
x = np.arange(3)

apia_on = [D[m]["apia"]["p3_onset"] for m in MODELS]
apia_vr = [D[m]["apia"]["vram_mb_batch1"] for m in MODELS]
apia_ms = [D[m]["apia"]["ms_per_step"] for m in MODELS]
def _lora(m):
    d = D[m]["lora"]
    return d if "p3_onset" in d else d["lr_1e-3"]  # headline = same lr

lora_on = [_lora(m)["p3_onset"] for m in MODELS]
lora_vr = [D[m]["lora"]["vram_mb"] for m in MODELS]
lora_ms = [D[m]["lora"]["ms_per_step"] for m in MODELS]
tuned_on = D[MODELS[1]]["lora"]["lr_1e-4"]["p3_onset"]

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))

ax[0].plot(x, apia_on, "o-", color="tab:blue", lw=2,
           label="APIA (no lr, zero retune)")
ax[0].plot(x, lora_on, "s--", color="tab:red", lw=2,
           label="LoRA r=8 (lr=1e-3)")
ax[0].plot(1, tuned_on, "*", ms=14, color="tab:green",
           label="LoRA tuned (lr=1e-4)")
ax[0].axhline(0.5, color="gray", ls=":", lw=1)
ax[0].annotate("chance", (2.05, 0.5), fontsize=8, color="gray",
               va="center")
ax[0].set_ylabel("phase-3 onset error")
ax[0].set_ylim(0, 0.6)
ax[0].legend(fontsize=8, loc="upper left")

ax[1].plot(x, apia_vr, "o-", color="tab:blue", lw=2, label="APIA")
ax[1].plot(x, lora_vr, "s--", color="tab:red", lw=2, label="LoRA")
ax[1].axhline(6144, color="k", ls=":", lw=1)
ax[1].annotate("6 GB", (0.05, 6300), fontsize=8, color="k")
ax[1].set_ylabel("peak VRAM (MB, batch-1)")
ax[1].legend(fontsize=8, loc="upper left")

ax[2].semilogy(x, apia_ms, "o-", color="tab:blue", lw=2, label="APIA")
ax[2].semilogy(x, lora_ms, "s--", color="tab:red", lw=2,
               label="LoRA")
ax[2].set_ylabel("update ms/step (log)")
for a in ax:
    a.set_xticks(x); a.set_xticklabels(XLAB)
fig.suptitle("Scale sweep on the domain-return protocol "
             "(20news A-B-A', single pass)", y=1.03)
fig.tight_layout()
fig.savefig("fig_phase2_scale.png", dpi=300, bbox_inches="tight")
print("wrote fig_phase2_scale.png")
