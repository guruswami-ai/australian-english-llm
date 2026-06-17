#!/usr/bin/env python3
"""
Qualitative generation test — the gate before any release. Every prior metric is a
logprob probe; this actually LOOKS at generations to check the model writes coherent
Australian English and isn't degenerate. Completion-style (these are CPT base models,
not chat). Prints base (adapter off) vs tuned for each prompt across registers.

  GEN_BASE=Qwen/Qwen3-8B-Base GEN_ADAPTER=models/aussie-qwen3-8b-base-lora-v1 GEN_LABEL=v1 \
    python gen_test.py
"""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = os.environ["GEN_BASE"]; ADAPTER = os.environ["GEN_ADAPTER"]; LABEL = os.environ.get("GEN_LABEL", "tuned")

PROMPTS = [
    ("informal",  "Yeah nah mate, the thing about the footy this weekend is"),
    ("informal",  "G'day, I was just down the shops and you wouldn't believe"),
    ("news",      "The Reserve Bank of Australia announced today that interest rates would"),
    ("formal",    "To whom it may concern, I am writing to formally apply for the position of"),
    ("everyday",  "Last weekend a few of us headed down to the beach and"),
    ("opinion",   "Honestly, the best thing about living in Australia is"),
    ("spelling",  "She picked up her favourite jumper, the colour of which"),
    ("local",     "The weather in Melbourne this time of year is famously"),
]

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
model = PeftModel.from_pretrained(base, ADAPTER).eval()


def gen(p):
    ids = tok(p, return_tensors="pt").to(0)
    torch.manual_seed(0)
    out = model.generate(**ids, max_new_tokens=70, do_sample=True, temperature=0.8, top_p=0.95,
                         repetition_penalty=1.2, pad_token_id=tok.eos_token_id or tok.pad_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


print(f"\n############ GENERATION TEST: base={BASE}  vs  {LABEL}={ADAPTER} ############")
for reg, p in PROMPTS:
    print("\n" + "=" * 78)
    print(f"[{reg}] PROMPT: {p}")
    with model.disable_adapter():
        b = gen(p)
    t = gen(p)
    print(f"\n  BASE → {b}")
    print(f"\n  {LABEL.upper()} → {t}")
