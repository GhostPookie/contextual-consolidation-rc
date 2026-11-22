"""VRAM divergence spike: the backprop wall, measured on 6 GB.

gpt2-large fp32 (~3.1 GB weights). Batch-size sweep B = 1..64:
  apia : forward-only feature pass under no_grad (activations are
         transient, freed per batch; learn state is CPU-side)
  lora : forward + backward + AdamW step (activation graph retained,
         grads + optimizer moments resident)

Per batch size: peak VRAM after load / after fwd / after bwd / after
step. LoRA OOM is data, not failure.
"""

import time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

MID = "openai-community/gpt2-large"
BATCHES = [1, 4, 8]
SEQ = 128
REPS = 2                      # batches per setting (peak is stable)


def peak():
    return torch.cuda.max_memory_allocated() / 1e6


def make_batch(tok, n):
    t = ["the model must adapt to a new domain without forgetting "
         "what it already learned " * 6] * n
    return tok(t, return_tensors="pt", padding="max_length",
               truncation=True, max_length=SEQ)


def apia_scan(tok, dev):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = AutoModel.from_pretrained(MID).to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    w = peak()
    out = {}
    for b in BATCHES:
        enc = make_batch(tok, b)
        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(REPS):
                    m(input_ids=enc["input_ids"].to(dev),
                      attention_mask=enc["attention_mask"].to(dev))
            dt = (time.perf_counter() - t0) / REPS / b
            out[b] = dict(w=w, peak=peak(), ms=dt * 1e3)
            torch.cuda.reset_peak_memory_stats()
        except torch.cuda.OutOfMemoryError:
            out[b] = dict(w=w, peak=-1, ms=-1)
        print(f"apia B={b}: peak={out[b]['peak']:.0f} "
              f"ms={out[b]['ms']:.2f}", flush=True)
    del m
    return out


def lora_scan(tok, dev):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = AutoModelForSequenceClassification.from_pretrained(
        MID, num_labels=2)
    base.config.pad_token_id = 50256
    cfg = LoraConfig(r=8, lora_alpha=16,
                     target_modules=["c_attn", "c_proj"],
                     lora_dropout=0.0, bias="none")
    m = get_peft_model(base, cfg).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    w = peak()
    out = {}
    for b in BATCHES:
        enc = make_batch(tok, b)
        lab = torch.zeros(b, dtype=torch.long, device=dev)
        try:
            t0 = time.perf_counter()
            for _ in range(REPS):
                o = m(input_ids=enc["input_ids"].to(dev),
                      attention_mask=enc["attention_mask"].to(dev),
                      labels=lab)
                o.loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            dt = (time.perf_counter() - t0) / REPS / b
            out[b] = dict(w=w, peak=peak(), ms=dt * 1e3)
            torch.cuda.reset_peak_memory_stats()
        except torch.cuda.OutOfMemoryError:
            out[b] = dict(w=w, peak=-1, ms=-1)
            torch.cuda.empty_cache()
        print(f"lora B={b}: peak={out[b]['peak']:.0f} "
              f"ms={out[b]['ms']:.2f}", flush=True)
    del m, opt
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MID)
    tok.pad_token = tok.eos_token
    A = apia_scan(tok, dev)
    L = lora_scan(tok, dev)

    import json
    print(f"\n{'B':>4}{'apia_peak':>11}{'lora_peak':>11}"
          f"{'apia_ms':>9}{'lora_ms':>9}")
    res = {}
    for b in BATCHES:
        a, l = A[b], L[b]
        lp = "OOM" if l["peak"] < 0 else f"{l['peak']:.0f}"
        lm = "OOM" if l["ms"] < 0 else f"{l['ms']:.2f}"
        print(f"{b:>4}{a['peak']:>10.0f}M{lp:>10}M"
              f"{a['ms']:>9.2f}{lm:>9}")
        res[str(b)] = {"apia": a, "lora": l}
    D = json.load(open("spike_results.json"))
    D["vram_scaling"] = {"model": MID, "seq": SEQ,
                         "weights_mb": A[1]["w"], "by_batch": res}
    json.dump(D, open("spike_results.json", "w"), indent=2)
    print("saved -> spike_results.json[vram_scaling]")


if __name__ == "__main__":
    main()
