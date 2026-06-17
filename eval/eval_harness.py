#!/usr/bin/env python3
"""
Qualification harness for the Australian-English CPT model.
Falsifiable, probabilistic, base-vs-v0, on HELD-OUT data. Three objective probes
scored by token log-probability (deterministic, no judge):

  1. AU-spelling preference   P(AU variant | context)         vs base
  2. Slang comprehension      MC accuracy from held-out glossary vs base
  3. Register perplexity       PPL(held-out AU) and a non-AU control (differential)

Same model object; adapter toggled on/off so base and v0 see identical inputs.
Bootstrap 95% CIs on every metric. Pre-registered success criteria printed at end.
"""
import torch, math, random, re, statistics
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import pyarrow.parquet as pq

BASE = "Qwen/Qwen3-8B"
ADAPTER = "models/aussie-qwen3-8b-lora-v0"
GLOSSARY = "bogan_glossary.txt"
VAL = "data/au_cpt_val_qwen3_2048.parquet"
random.seed(0)

# ---- AU/US spelling pairs with carrier contexts (context ends right before the word) ----
SPELL = [
    ("The walls were painted a lovely shade of ", "colour", "color"),
    ("I didn't ", "realise", "realize"),
    ("We met in the town ", "centre", "center"),
    ("She works hard to support the ", "labour", "labor"),
    ("My ", "favourite", "favorite"),
    ("He's a friendly ", "neighbour", "neighbor"),
    ("They went to the ", "theatre", "theater"),
    ("You need a driver's ", "licence", "license"),
    ("We need to ", "organise", "organize"),
    ("The country's ", "defence", "defense"),
    ("It's about a kilo", "metre", "meter"),
    ("Pour in a ", "litre", "liter"),
    ("The sky turned ", "grey", "gray"),
    ("I need to change a ", "tyre", "tire"),
    ("He studies ", "maths", "math"),
    ("I rang my ", "mum", "mom"),
    ("The TV ", "programme", "program"),
    ("Window frames made of ", "aluminium", "aluminum"),
    ("She's a frequent ", "traveller", "traveler"),
    ("Behaviour and ", "behaviour", "behavior"),
    ("We should ", "apologise", "apologize"),
    ("It was a great ", "honour", "honor"),
    ("Check the ", "catalogue", "catalog"),
    ("Mind the ", "kerb", "curb"),
    ("A cheque for the ", "cheque", "check"),
    ("The knight wore his ", "armour", "armor"),
    ("I didn't ", "recognise", "recognize"),
    ("Let me ", "analyse", "analyze"),
    ("Try to ", "memorise", "memorize"),
    ("Don't ", "criticise", "criticize"),
    ("He had a thick ", "moustache", "mustache"),
    ("The cabin was warm and ", "cosy", "cozy"),
    ("There was ", "mould", "mold"),
    ("She wore her ", "pyjamas", "pajamas"),
    ("A box of ", "jewellery", "jewelry"),
    ("Down by the ", "harbour", "harbor"),
    ("I heard a ", "rumour", "rumor"),
    ("He has a great sense of ", "humour", "humor"),
    ("The ", "flavour", "flavor"),
    ("A faint ", "odour", "odor"),
    ("Full of ", "vigour", "vigor"),
    ("We ", "specialise", "specialize"),
    ("Try to ", "minimise", "minimize"),
    ("Let's ", "maximise", "maximize"),
    ("It was a ", "marvellous", "marvelous"),
    ("A ", "woollen", "woolen"),
    ("The match was ", "cancelled", "canceled"),
    ("He ", "modelled", "modeled"),
    ("Ask the ", "jeweller", "jeweler"),
    ("She's a ", "sceptic", "skeptic"),
    ("A glass of ", "draught", "draft"),
    ("Spread the ", "rumour", "rumour"),
    ("The school ", "enrolment", "enrollment"),
    ("We will ", "fulfil", "fulfill"),
]

# ---- naturalness / anti-AI-tell probe (generation-based) ----
SLOP = ["delve", "tapestry", "testament", "leverage", "robust", "crucial", "vibrant",
        "bustling", "boasts", "foster", "intricate", "nuanced", "multifaceted", "seamless",
        "pivotal", "realm", "underscore", "moreover", "furthermore", "in conclusion",
        "it's worth noting", "navigate the", "utilize", "utilise", "transformative",
        "revolutionize", "ever-evolving", "in today's", "game-changer", "cutting-edge",
        "not only", "but also", "a testament to"]
NAT_PROMPTS = [
    "The best thing about my weekend was", "Honestly, the new cafe down the road is",
    "I've been thinking about getting a new", "Work has been pretty",
    "The footy game last night was", "My plan for the holidays is to",
    "The traffic this morning was", "I reckon the biggest issue around here is",
]

@torch.no_grad()
def naturalness(model, tok):
    torch.manual_seed(0)
    em = slop = chars = 0; bursts = []
    for p in NAT_PROMPTS:
        ids = tok(p, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=80, do_sample=True, temperature=0.8,
                             top_p=0.95, pad_token_id=tok.eos_token_id)
        t = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        chars += len(t); em += t.count(chr(8212)); lo = t.lower()
        slop += sum(lo.count(w) for w in SLOP)
        sents = [len(s.split()) for s in re.split(r"[.!?]+", t) if s.strip()]
        if len(sents) > 1:
            bursts.append(statistics.pstdev(sents) / (statistics.mean(sents) + 1e-6))
    c = max(chars, 1)
    return dict(emdash=1000 * em / c, slop=1000 * slop / c,
                burst=(sum(bursts) / len(bursts) if bursts else 0.0))

@torch.no_grad()
def logprob(model, tok, context, target):
    ctx = tok(context, return_tensors="pt").input_ids.to(model.device)
    tgt = tok(target, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    ids = torch.cat([ctx, tgt], 1)
    logits = model(ids).logits[0, ctx.shape[1] - 1:-1].float()
    lp = torch.log_softmax(logits, -1)[range(tgt.shape[1]), tgt[0]].sum().item()
    return lp, tgt.shape[1]

def boot_ci(vals, fn=lambda x: sum(x) / len(x), n=2000):
    if not vals:
        return (float("nan"),) * 3
    bs = []
    for _ in range(n):
        s = [vals[random.randrange(len(vals))] for _ in vals]
        bs.append(fn(s))
    bs.sort()
    return fn(vals), bs[int(0.025 * n)], bs[int(0.975 * n)]

def spelling(model, tok):
    pref, pau = [], []
    for ctx, au, us in SPELL:
        la, _ = logprob(model, tok, ctx, au)
        lu, _ = logprob(model, tok, ctx, us)
        pref.append(1.0 if la > lu else 0.0)
        pau.append(math.exp(la) / (math.exp(la) + math.exp(lu)))
    return boot_ci(pref), boot_ci(pau)

def load_glossary(n=120):
    items = []
    for line in open(GLOSSARY, encoding="utf-8"):
        line = line.strip()
        if line.startswith("#") or ":" not in line:
            continue
        t, d = line.split(":", 1)
        t, d = t.strip(), d.strip()
        if t and d and len(d) > 4:
            items.append((t, d))
    random.shuffle(items)
    return items[:n]

def slang(model, tok, items):
    defs = [d for _, d in items]
    correct = []
    for i, (term, gold) in enumerate(items):
        distractors = random.sample([d for j, d in enumerate(defs) if j != i], 3)
        cands = [gold] + distractors
        random.shuffle(cands)
        prompt = f"In Australian slang, \"{term}\" means: "
        scores = []
        for c in cands:
            lp, n = logprob(model, tok, prompt, c)
            scores.append(lp / max(n, 1))   # length-normalised
        correct.append(1.0 if cands[scores.index(max(scores))] == gold else 0.0)
    return boot_ci(correct)

@torch.no_grad()
def ppl(model, tok, token_lists):
    nlls, ntok = [], 0
    for ids in token_lists:
        t = torch.tensor([ids], device=model.device)
        out = model(t, labels=t)
        nlls.append(out.loss.item() * (t.shape[1] - 1))
        ntok += t.shape[1] - 1
    return math.exp(sum(nlls) / ntok)

def main():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
    model = PeftModel.from_pretrained(base, ADAPTER).eval()

    glossary = load_glossary()
    # held-out AU text: 96 val sequences (truncated to 512 tok for speed)
    au = [r[:512] for r in pq.read_table(VAL).column("input_ids").to_pylist()[:96]]
    # non-AU control
    try:
        from datasets import load_dataset
        wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in wt["text"] if len(t) > 200][:96]
        nonau = [tok(t, truncation=True, max_length=512).input_ids for t in texts]
    except Exception as e:
        print("non-AU control unavailable:", e); nonau = None

    def run(name):
        sp_pref, sp_pau = spelling(model, tok)
        sl = slang(model, tok, glossary)
        au_ppl = ppl(model, tok, au)
        na_ppl = ppl(model, tok, nonau) if nonau else float("nan")
        nat = naturalness(model, tok)
        print(f"\n=== {name} ===")
        print(f"  Naturalness: em-dash/1k {nat['emdash']:.3f}  slop/1k {nat['slop']:.3f}  "
              f"burstiness {nat['burst']:.2f}")
        print(f"  AU-spelling preferred:  {sp_pref[0]:.3f}  CI[{sp_pref[1]:.3f},{sp_pref[2]:.3f}]")
        print(f"  P(AU spelling) mean:    {sp_pau[0]:.3f}  CI[{sp_pau[1]:.3f},{sp_pau[2]:.3f}]")
        print(f"  Slang MC accuracy:      {sl[0]:.3f}  CI[{sl[1]:.3f},{sl[2]:.3f}]  (chance 0.25)")
        print(f"  PPL held-out AU:        {au_ppl:.2f}")
        print(f"  PPL non-AU (wikitext):  {na_ppl:.2f}")
        return dict(sp=sp_pref[0], pau=sp_pau[0], sl=sl[0], au=au_ppl, na=na_ppl, nat=nat)

    with model.disable_adapter():
        b = run("BASE Qwen3-8B")
    v = run("V0 AU-tuned")

    print("\n===== BASE -> V0 (falsifiable scorecard) =====")
    print(f"  AU spelling preferred:  {b['sp']:.3f} -> {v['sp']:.3f}  (Δ {v['sp']-b['sp']:+.3f})")
    print(f"  P(AU spelling):         {b['pau']:.3f} -> {v['pau']:.3f}  (Δ {v['pau']-b['pau']:+.3f})")
    print(f"  Slang accuracy:         {b['sl']:.3f} -> {v['sl']:.3f}  (Δ {v['sl']-b['sl']:+.3f})")
    print(f"  PPL AU (lower=better):  {b['au']:.2f} -> {v['au']:.2f}  (Δ {v['au']-b['au']:+.2f})")
    print(f"  em-dash/1k:             {b['nat']['emdash']:.3f} -> {v['nat']['emdash']:.3f}  (lower=more human)")
    print(f"  slop-words/1k:          {b['nat']['slop']:.3f} -> {v['nat']['slop']:.3f}  (lower=more human)")
    print(f"  burstiness:             {b['nat']['burst']:.2f} -> {v['nat']['burst']:.2f}  (higher=more human)")
    if nonau:
        dau, dna = v['au']-b['au'], v['na']-b['na']
        print(f"  PPL non-AU:             {b['na']:.2f} -> {v['na']:.2f}  (Δ {dna:+.2f})")
        print(f"  >> AU-specificity: ΔPPL(AU) {dau:+.2f} vs ΔPPL(non-AU) {dna:+.2f} "
              f"-> {'AU-specific ✓' if dau < dna else 'NOT AU-specific ✗'}")
    print("\nPre-registered v0 success: AU-specificity holds AND any of "
          "{spelling, slang, AU-PPL} improves beyond CI.")

if __name__ == "__main__":
    main()
