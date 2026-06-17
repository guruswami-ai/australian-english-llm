# australian-english-llm

**A falsifiable, stage-gated methodology for resource-disciplined LLM specialisation — demonstrated by adapting a generic English LLM to produce believable Australian English.**

> ⚠️ **Status: incomplete and under active development.** This is an open research log, not a finished
> product. The current model is a **v0.1 research preview**; the flagship **v1.0** is a planned, not-yet-built
> target that must *quantifiably* beat v0.1 on the same falsifiable harness before it ships. We publish the
> journey — including a **negative result** — because the methodology is the point.

---

## Why this exists

Generic English LLMs are trained on huge, mostly-unknown, mostly-non-Australian data. Ask one to
"sound Australian" and you get **clichés** — *g'day, throw a shrimp on the barbie, crikey* — not
believable Australian English. Those clichés are baked into the base model's distribution and **cannot
be prompted out.** The distinction is subtle but it matters wherever language nuance carries the
experience: **chatbots, speech-to-speech (S2S) pipelines, fast localised inference.**

The only reliable fix is to **shift the model's distribution** with real, representative Australian
data. This repo is the method for doing that *with evidence at every step*.

**It worked — measurably.** On *"The weather in Melbourne this time of year is famously…"*:

| Model | Continuation |
|---|---|
| Generic Qwen3-8B | *"…hot and humid"* ❌ (wrong; another run spiralled into a Markov-chain maths tangent) |
| **Ours (v0)** | *"…unreliable and can change from one day to another"* ✅ |
| **Ours (v1)** | *"…changeable. … it also snowed on Mt Gambier"* ✅ (locally accurate) |

…with **zero ocker clichés**. The Australian-ness shows up as register, spelling, and *local knowledge* —
believable, not performed. (More samples in [`results/`](results/).)

## The actual contribution: a methodology, not a model

The transferable value here is **domain-agnostic** — it applies to *any* fine-tuning / specialisation
effort:

1. **Falsifiable stage-gates = resource-allocation discipline.** Every stage pre-registers *falsifiable*
   success criteria **before** spending compute, then measures and decides: **advance · refine · abandon.**
   "Abandon" is a first-class outcome. We spent 51 GPU-hours on a full-epoch run (`v1`) specifically to
   earn the evidence that *more training tokens don't help* — so we don't burn 500 hours discovering it
   the expensive way. A stage-gate turns "stop" from a failure into a cheap, deliberate decision.

2. **Data quality is the foundational gate (Stage 0).** Without good training data, the method, compute,
   and architecture are pointless — garbage in, garbage out. So the first and largest effort goes to the
   **corpus**: sourcing, cleaning, dedup, representativeness, AI-contamination control, and register
   coverage. A model is only ever as good as what it learned from. The **data pipeline is the most
   reusable part of this repo.**

3. **RLHF-free evaluation.** No human feedback loop, so qualification must be automated and *falsifiable*:
   held-out probes, an **AU-specificity differential** (does the adaptation help Australian text *specifically*,
   or just general fitting?), bootstrap confidence intervals, and pre-registered thresholds. See
   [`METHODOLOGY.md`](METHODOLOGY.md).

## The approach

Gather **contemporary** Australian English across the full register range — formal ↔ informal,
written ↔ dictated/transcribed speech — then clean it into a register-balanced corpus and use it to
shift a generic base model (QLoRA continued pre-training), qualifying every step on the falsifiable harness.

**Corpus (built):** ~25.5M documents / ~1.6B tokens across four registers:

| Register | Source | Scale |
|---|---|---|
| `casual` | Australian Reddit (2008–2020) | ~1.25B tok |
| `formal` | Commonwealth Hansard (2006–2020, House + Senate) | ~239M tok |
| `news` | Guardian Australia (≤2020) | ~115M tok |
| `vernacular` | ocker Reddit communities | ~6.5M tok |
| `dialogue` | TV scripts (transcribed speech) | thin (~0.5M tok) |

The cleaned-text stage is **tokenizer-agnostic** — switching base models is a *repack*, not a re-clean.

## Results so far (honest)

| | v0 (Qwen3-8B-instruct + LoRA) | v1 (Qwen3-8B-Base + LoRA, full epoch) |
|---|---|---|
| Pre-registered verdict | ✅ **qualified** | ❌ **missed bar (2/5)** |
| AU-specificity (ΔPPL AU : non-AU) | ~9:1 | ~450:1 |
| AU perplexity | 61.8 → 29.3 | 64.0 → 32.5 |
| Catastrophic forgetting | none | none |

**The headline finding is the negative result.** v1 trained on **6.5× more tokens** than v0 and did **not**
beat it — the *relative* AU adaptation was essentially identical. **Token scale was not the lever.** The
methodology told us to stop scaling and reallocate to **data composition / registers** (the v2 direction).
That is the methodology doing its job. Full scorecards: [`results/`](results/).

## Roadmap (v0.1 → v1.0)

- [x] Stage 0 — representative, register-balanced AU corpus
- [x] Falsifiable RLHF-free eval harness (AU-specificity differential, bootstrap CIs)
- [x] **v0** — qualified preview model
- [x] **v1** — scale experiment → *scale is not the lever* (negative result, recorded)
- [ ] **v2 / v1.0** — register-conditioned model, re-baselined eval; must *quantifiably* beat v0.1
- [ ] Teacher → student: distil a small specialist (chatbot / S2S / fast inference)

## Repository layout

```
data/    corpus pipeline — harvesters, cleaning, dedup, register-balancing (the Stage-0 machinery)
train/   QLoRA continued-pretraining + tokenisation/packing
eval/    the falsifiable harness + qualitative generation test
results/ pre-registered scorecards (v0 PASS, v1 FAIL) + generation samples, verbatim
```

> **We publish code + reproduction recipes, never the scraped corpora.** The data has mixed provenance
> (Reddit, news APIs, fan-transcribed scripts); redistributing it would be inappropriate. Bring your own
> sources via the harvesters. Base model is Qwen3 (Apache-2.0), so derivative weights are clean.

## Honest caveats

- The current model is a **thin LoRA adapter**, not a deeply trained model — it rambles like a base model
  and does not follow instructions.
- Most metrics are **log-probability probes**, complemented by a qualitative generation test — not a broad
  human eval.
- The "formality spectrum" (register control) is **v1.0 work**, not yet in v0.1.
- Paths in the scripts (`/Volumes/scratch/...`) are from the original case study; treat them as examples.

## License

Code: Apache-2.0 (see [`LICENSE`](LICENSE)). Built on Qwen3 (Apache-2.0).
