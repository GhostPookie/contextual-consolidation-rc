"""Phase-2 head-to-head: frozen-LLM + APIA contextual memory vs LoRA.

Same stream as cls_spike.py (20newsgroups A -> B -> A', single pass,
predict-then-update per sample). Three arms, identical protocol:

  block  : frozen GPT-2 features -> spline -> block-diag RLS (memoryless)
  ctxmem : same features -> RLS + ContextualMemory (fast-gate variant)
  lora   : GPT-2 + LoRA adapters (r=8, attn) + trainable head, online
           AdamW single-pass updates (the standard continual-FT baseline)

Metrics: per-phase end error, phase-3 onset error (first ONSET samples
after regime return), peak CUDA VRAM, wall-clock update time.
"""

import numpy as np
import time, os, sys

import torch
import torch.nn.functional as F
from transformers import (AutoModel, AutoTokenizer,
                          AutoModelForSequenceClassification)
from peft import LoraConfig, get_peft_model

from cls_spike import (load_domains, get_features, SplineFeatureMap,
                       BlockDiagRLS, ContextualMemory, run_stream,
                       phase_err, PA, PB, PA_RET, ONSET, MAX_TOK,
                       MODEL_ID)

LORA_LR = 1e-3


def run_lora(texts, Y, tok, dev, bounds, lr=LORA_LR, seed=None,
             model_id=None):
    """Online single-pass LoRA fine-tune; decisions recorded pre-update."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if seed is not None:
        torch.manual_seed(seed)
    base = AutoModelForSequenceClassification.from_pretrained(
        model_id or MODEL_ID, num_labels=2)
    base.config.pad_token_id = tok.pad_token_id
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn", "c_proj"],
                     lora_dropout=0.0, bias="none")
    mdl = get_peft_model(base, cfg).to(dev)
    mdl.print_trainable_parameters()
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr)

    # tokenize once (independent of update timing)
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=MAX_TOK)
    ids, mask = enc["input_ids"], enc["attention_mask"]

    preds = np.empty(len(texts))
    t0 = time.perf_counter()
    mdl.train()
    for i in range(len(texts)):
        inp = {k: v[i:i + 1].to(dev) for k, v in
               (("input_ids", ids), ("attention_mask", mask))}
        out = mdl(**inp, labels=torch.tensor(
            [int(Y[i, 0])], device=dev))
        preds[i] = float(out.logits.argmax(-1).item())  # pre-update
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e6
    return preds, dt, peak


def report(lab, preds, Y, bounds, extra=""):
    p = lambda lo, hi: phase_err(preds, Y[:, 0], lo, hi)
    print(f"{lab:>14}: P1-end={p(bounds[1]-ONSET, bounds[1]):.3f} "
          f"P2-end={p(bounds[2]-ONSET, bounds[2]):.3f} "
          f"P3-onset={p(bounds[2], bounds[2]+ONSET):.3f} "
          f"P3-end={p(bounds[3]-ONSET, bounds[3]):.3f} {extra}",
          flush=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token

    A1, B, A2 = load_domains()
    stream = A1 + B + A2
    texts = [t for t, _ in stream]
    Y = np.array([l for _, l in stream], dtype=float).reshape(-1, 1)
    bounds = [0, PA, PA + PB, PA + PB + PA_RET]

    # ---- APIA arms (features cached from cls_spike) ----
    torch.cuda.reset_peak_memory_stats()
    mdl_f = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
    for p in mdl_f.parameters():
        p.requires_grad_(False)
    H = np.concatenate([get_features(n, [t for t, _ in s], tok, mdl_f, dev)
                        for n, s in (("A1", A1), ("B", B), ("A2", A2))])
    feat_peak = torch.cuda.max_memory_allocated() / 1e6
    del mdl_f
    torch.cuda.empty_cache()

    H = (H - H.mean(0)) / (H.std(0) + 1e-6)
    X = np.clip(H, -3.0, 3.0) / 3.0
    fmap = SplineFeatureMap(X.shape[1], 10)
    Phi = np.stack([fmap.transform(x) for x in X])

    t0 = time.perf_counter()
    preds, _ = run_stream(Phi, Y, "block")
    report("apia-block", preds, Y, bounds,
           f"({time.perf_counter()-t0:.0f}s)")

    t0 = time.perf_counter()
    preds, mem = run_stream(Phi, Y, "ctxmem", e_gate=1.0, eta=5.0,
                            alpha_ema=0.05)
    report("apia-ctxmem", preds, Y, bounds,
           f"cons={mem.n_consolidations} "
           f"({time.perf_counter()-t0:.0f}s)")

    # ---- LoRA arm ----
    preds, dt, peak = run_lora(texts, Y, tok, dev, bounds)
    report("lora(r=8)", preds, Y, bounds, f"({dt:.0f}s)")

    wf = mem.model.get_weights() if hasattr(mem, "model") else None
    n_train = 0 if wf is None else wf.size + mem.W_slow.size
    print(f"\npeak VRAM:")
    print(f"  apia : feat-extract={feat_peak:.0f} MB, "
          f"learn-state=CPU-resident ({n_train} floats, "
          f"{n_train*8/1e6:.2f} MB f64)")
    print(f"  lora : {peak:.0f} MB (model+grads+AdamW state)")
    print(f"update time: lora={dt/len(texts)*1000:.1f} ms/step")


if __name__ == "__main__":
    main()
