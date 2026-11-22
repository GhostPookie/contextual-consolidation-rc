"""Phase-2 scaling profile: where does the backprop wall actually hit?

Sweep GPT-2 sizes on the same 20news A->B->A' single-pass stream:
  apia : frozen model -> last hidden -> spline -> RLS + ContextualMemory
         (learn state lives on CPU, ~tens of KB, size-independent)
  lora : same model + LoRA adapters + head, online AdamW
         (needs forward + backward + optimizer on GPU)

For each model size report: P3-onset error, P3-end error, peak VRAM,
update ms/step. LoRA OOM is data, not failure.
"""

import numpy as np
import time

import torch
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

from cls_spike import (load_domains, get_features, SplineFeatureMap,
                       run_stream, phase_err, PA, PB, PA_RET, ONSET,
                       MAX_TOK)

LORA_LR = 1e-3
MODELS = ["openai-community/gpt2",
          "openai-community/gpt2-medium",
          "openai-community/gpt2-large"]


def run_lora(mid, ids, mask, Y, dev):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        base = AutoModelForSequenceClassification.from_pretrained(
            mid, num_labels=2)
        base.config.pad_token_id = 50256
        cfg = LoraConfig(r=8, lora_alpha=16,
                         target_modules=["c_attn", "c_proj"],
                         lora_dropout=0.0, bias="none")
        mdl = get_peft_model(base, cfg).to(dev)
        opt = torch.optim.AdamW(mdl.parameters(), lr=LORA_LR)
        n_tr = sum(p.numel() for p in mdl.parameters() if p.requires_grad)

        preds = np.empty(len(Y))
        t0 = time.perf_counter()
        mdl.train()
        for i in range(len(Y)):
            out = mdl(input_ids=ids[i:i + 1].to(dev),
                      attention_mask=mask[i:i + 1].to(dev),
                      labels=torch.tensor([int(Y[i, 0])], device=dev))
            preds[i] = float(out.logits.argmax(-1).item())
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1e6
        del mdl, opt
        return preds, dt, peak, n_tr
    except torch.cuda.OutOfMemoryError:
        return None, 0.0, torch.cuda.max_memory_allocated() / 1e6, -1
    finally:
        torch.cuda.empty_cache()


def apia_arm(mid, tag, texts, Y, tok, dev):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mdl = AutoModel.from_pretrained(mid).to(dev).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    n = len(texts)
    nA, nB = PA, PB
    H = np.concatenate([
        get_features(f"{tag}_A1", texts[:nA], tok, mdl, dev),
        get_features(f"{tag}_B", texts[nA:nA + nB], tok, mdl, dev),
        get_features(f"{tag}_A2", texts[nA + nB:], tok, mdl, dev)])
    peak = torch.cuda.max_memory_allocated() / 1e6
    del mdl
    torch.cuda.empty_cache()

    H = (H - H.mean(0)) / (H.std(0) + 1e-6)
    X = np.clip(H, -3.0, 3.0) / 3.0
    fmap = SplineFeatureMap(X.shape[1], 10)
    Phi = np.stack([fmap.transform(x) for x in X])

    t0 = time.perf_counter()
    preds, mem = run_stream(Phi, Y, "ctxmem", e_gate=1.0, eta=5.0,
                            alpha_ema=0.05)
    dt = time.perf_counter() - t0
    n_state = mem.model.get_weights().size + mem.W_slow.size
    return preds, dt, peak, n_state


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODELS[0])
    tok.pad_token = tok.eos_token

    A1, B, A2 = load_domains()
    stream = A1 + B + A2
    texts = [t for t, _ in stream]
    Y = np.array([l for _, l in stream], dtype=float).reshape(-1, 1)
    bnd = [0, PA, PA + PB, PA + PB + PA_RET]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=MAX_TOK)

    err = lambda p, lo, hi: phase_err(p, Y[:, 0], lo, hi)
    print(f"{'model':<14}{'arm':<6}{'P3-onset':>9}{'P3-end':>8}"
          f"{'VRAM':>9}{'ms/step':>9}{'state':>9}")
    for mid in MODELS:
        tag = mid.split("/")[-1]
        p, dt, pk, n_st = apia_arm(mid, tag, texts, Y, tok, dev)
        print(f"{tag:<14}{'apia':<6}{err(p,bnd[2],bnd[2]+ONSET):>9.3f}"
              f"{err(p,bnd[3]-ONSET,bnd[3]):>8.3f}{pk:>8.0f}M"
              f"{dt/len(Y)*1000:>8.1f} {n_st*8/1e3:>7.0f}K", flush=True)
        p, dt, pk, n_tr = run_lora(mid, enc["input_ids"],
                                   enc["attention_mask"], Y, dev)
        if p is None:
            print(f"{tag:<14}{'lora':<6}{'OOM':>9}{'':>8}{pk:>8.0f}M",
                  flush=True)
        else:
            print(f"{tag:<14}{'lora':<6}{err(p,bnd[2],bnd[2]+ONSET):>9.3f}"
                  f"{err(p,bnd[3]-ONSET,bnd[3]):>8.3f}{pk:>8.0f}M"
                  f"{dt/len(Y)*1000:>8.1f} {n_tr*4/1e3:>7.0f}K",
                  flush=True)


if __name__ == "__main__":
    main()
