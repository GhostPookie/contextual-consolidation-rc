"""Multi-seed robustness for the Phase-2 GPT-2 family sweep.

Reviewer question killed upfront: "are the LLM numbers single-seed
point estimates?" For each model scale and seed we re-draw the domain
split/shuffle (seeded load_domains) and re-init LoRA (torch seed), then
run the identical A -> B -> A' protocol. APIA has no random init on raw
features, so its variance measures stream-composition sensitivity only
-- stated as such in the paper.

Writes incrementally to spike_results.json["multiseed_gpt2"].
"""

import json
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from cls_spike import (load_domains, get_features, SplineFeatureMap,
                       run_stream, phase_err, PA, PB, PA_RET, ONSET)
from adapter_spike import run_lora

MODELS = {
    "gpt2": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
}
LORA_LRS = {
    "gpt2": [1e-3],
    "gpt2-medium": [1e-3, 1e-4, 1e-5],
    "gpt2-large": [1e-3, 1e-4, 1e-5],
}
SEEDS = [0, 1, 2]
RESULTS = "spike_results.json"
KEY = "multiseed_gpt2"


def save(results):
    d = json.load(open(RESULTS))
    d[KEY] = results
    json.dump(d, open(RESULTS, "w"), indent=2)


def feats(model_tag, seed, seg, texts, tok, mdl, dev):
    cf = f"clsfeat_{model_tag}_{seg}_s{seed}"
    return get_features(cf, texts, tok, mdl, dev)


def summarize(vals):
    return {"seeds": vals, "mean": float(np.mean(vals)),
            "std": float(np.std(vals))}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = json.load(open(RESULTS))
    results = d.get(KEY, {})

    for tag, mid in MODELS.items():
        results.setdefault(tag, {})
        tok = AutoTokenizer.from_pretrained(mid)
        tok.pad_token = tok.eos_token

        # ---- APIA arms: one frozen feature pass per seed ----
        mdl_f = AutoModel.from_pretrained(mid).to(dev).eval()
        for p in mdl_f.parameters():
            p.requires_grad_(False)
        for seed in SEEDS:
            A1, B, A2 = load_domains(seed=seed)
            Y = np.array([l for _, l in A1 + B + A2],
                         dtype=float).reshape(-1, 1)
            bounds = [0, PA, PA + PB, PA + PB + PA_RET]
            H = np.concatenate([
                feats(tag, seed, n, [t for t, _ in s], tok, mdl_f, dev)
                for n, s in (("A1", A1), ("B", B), ("A2", A2))])
            H = (H - H.mean(0)) / (H.std(0) + 1e-6)
            X = np.clip(H, -3.0, 3.0) / 3.0
            fmap = SplineFeatureMap(X.shape[1], 10)
            Phi = np.stack([fmap.transform(x) for x in X])

            for arm, kw in (("apia_block", ("block", {})),
                            ("apia_ctxmem",
                             ("ctxmem", dict(e_gate=1.0, eta=5.0,
                                             alpha_ema=0.05)))):
                ex = results[tag].get(arm, [])
                if isinstance(ex, dict):
                    ex = ex["per_seed"]
                if len(ex) > SEEDS.index(seed):
                    continue
                kind, kws = kw
                preds, model = run_stream(Phi, Y, kind, **kws)
                e = lambda lo, hi: phase_err(preds, Y[:, 0], lo, hi)
                rec = {
                    "p1_end": e(bounds[1] - ONSET, bounds[1]),
                    "p2_end": e(bounds[2] - ONSET, bounds[2]),
                    "p3_onset": e(bounds[2], bounds[2] + ONSET),
                    "p3_end": e(bounds[3] - ONSET, bounds[3]),
                    "cons": getattr(model, "n_consolidations", 0),
                }
                slot = results[tag].setdefault(arm, [])
                if isinstance(slot, dict):
                    slot = slot["per_seed"]
                if len(slot) <= SEEDS.index(seed):
                    slot.append(rec)
                    save(results)
                print(f"{tag} s{seed} {arm}: onset={rec['p3_onset']:.3f} "
                      f"p3end={rec['p3_end']:.3f} cons={rec['cons']}",
                      flush=True)
        del mdl_f
        torch.cuda.empty_cache()

        # ---- LoRA arm: fresh init per seed per lr ----
        for seed in SEEDS:
            A1, B, A2 = load_domains(seed=seed)
            texts = [t for t, _ in A1 + B + A2]
            Y = np.array([l for _, l in A1 + B + A2],
                         dtype=float).reshape(-1, 1)
            bounds = [0, PA, PA + PB, PA + PB + PA_RET]
            for lr in LORA_LRS[tag]:
                arm = f"lora_lr{lr:g}"
                ex = results[tag].get(arm, [])
                if isinstance(ex, dict):
                    ex = ex["per_seed"]
                if len(ex) > SEEDS.index(seed):
                    continue
                preds, dt, peak = run_lora(texts, Y, tok, dev, bounds,
                                           lr=lr, seed=seed, model_id=mid)
                e = lambda lo, hi: phase_err(preds, Y[:, 0], lo, hi)
                rec = {
                    "p1_end": e(bounds[1] - ONSET, bounds[1]),
                    "p2_end": e(bounds[2] - ONSET, bounds[2]),
                    "p3_onset": e(bounds[2], bounds[2] + ONSET),
                    "p3_end": e(bounds[3] - ONSET, bounds[3]),
                    "ms_per_step": dt / len(texts) * 1000,
                    "peak_mb": peak,
                }
                slot = results[tag].setdefault(arm, [])
                if isinstance(slot, dict):
                    slot = slot["per_seed"]
                if len(slot) <= SEEDS.index(seed):
                    slot.append(rec)
                    save(results)
                print(f"{tag} s{seed} {arm}: onset={rec['p3_onset']:.3f} "
                      f"p3end={rec['p3_end']:.3f} ({dt:.0f}s)",
                      flush=True)
                del preds
                torch.cuda.empty_cache()

    # ---- summary pass ----
    for tag, arms in results.items():
        for arm, recs in arms.items():
            arms[arm] = {
                "per_seed": recs,
                "p3_onset": summarize([r["p3_onset"] for r in recs]),
                "p3_end": summarize([r["p3_end"] for r in recs]),
            }
    save(results)
    print("\ndone ->", KEY)


if __name__ == "__main__":
    main()
