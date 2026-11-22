# kaggle_1p3b_vram.py — OPT-1.3B VRAM wall spike (self-contained for Kaggle)
# APIA: frozen fwd-only under no_grad | LoRA: fwd+bwd+AdamW, model built ONCE.
import torch, time, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

MODEL = "facebook/opt-1.3b"
SEQ = 128
REPS = 2
APIA_BS = [1, 4, 8, 16, 32, 64]
LORA_BS = [1, 4, 8, 16, 32]
OUT = "/kaggle/working/vram_opt13b.json"

dev = "cuda"
peak = lambda: torch.cuda.max_memory_allocated() / 1e6
tok = AutoTokenizer.from_pretrained(MODEL)
ids = torch.randint(0, 50256, (1, SEQ))

base = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16).to(dev)
base.eval()
for p in base.parameters():
    p.requires_grad_(False)
w = peak()
print(f"weights+load: {w:.0f} MB", flush=True)

def rep_batch(b):
    return ids.repeat(b, 1).to(dev)

results = {"model": MODEL, "seq": SEQ, "weights_mb": w,
           "device": torch.cuda.get_device_name(0),
           "device_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6),
           "by_batch": {}}

# ---- APIA: forward-only, no_grad, transient activations ----
for b in APIA_BS:
    torch.cuda.reset_peak_memory_stats()
    x = rep_batch(b)
    try:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(REPS):
                base(input_ids=x, labels=x)
        r = dict(w=w, peak=peak(), ms=(time.perf_counter() - t0) / REPS / b * 1e3)
    except torch.cuda.OutOfMemoryError:
        r = dict(w=w, peak=-1, ms=-1)
        torch.cuda.empty_cache()
    results["by_batch"].setdefault(str(b), {})["apia"] = r
    print(f"apia B={b}: peak={r['peak']:.0f} ms={r['ms']:.2f}", flush=True)

# ---- LoRA: built ONCE, reused for all batches ----
cfg = LoraConfig(r=8, lora_alpha=16,
                 target_modules=["q_proj", "v_proj"],
                 lora_dropout=0.0, bias="none",
                 task_type="CAUSAL_LM")
lora = get_peft_model(base, cfg)      # peft freezes base, unfreezes lora_*
lora.train()
opt = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad], lr=1e-4)

for b in LORA_BS:
    torch.cuda.reset_peak_memory_stats()
    x = rep_batch(b)
    try:
        t0 = time.perf_counter()
        for _ in range(REPS):
            o = lora(input_ids=x, labels=x)
            o.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        r = dict(w=peak(), peak=peak(), ms=(time.perf_counter() - t0) / REPS / b * 1e3)
        r["w"] = results["weights_mb"]
    except torch.cuda.OutOfMemoryError:
        r = dict(w=w, peak=-1, ms=-1)
        torch.cuda.empty_cache()
    results["by_batch"].setdefault(str(b), {})["lora"] = r
    print(f"lora B={b}: peak={r['peak']:.0f} ms={r['ms']:.2f}", flush=True)

json.dump({"vram_scaling": results}, open(OUT, "w"), indent=2)
print("saved ->", OUT)
