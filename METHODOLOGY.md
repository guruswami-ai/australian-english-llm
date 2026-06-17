# Methodology: falsifiable, stage-gated LLM specialisation

This is the transferable core of the project. It is **domain-agnostic** — "Australian English" is just
the worked example. The aim is to specialise a base model toward a target distribution while spending
compute *only where evidence says it pays*, and to know — falsifiably — whether it's working, overfitting,
or a dead end.

## The principle

> Every stage pre-registers **falsifiable success criteria before spending compute**, then measures and
> decides: **advance · refine · abandon.**

Pre-registration (writing the pass/fail thresholds *before* the run) is what makes results
un-rationalisable after the fact. "It looks better" is not allowed; "P(AU spelling) ≥ 0.60 with CI lower
bound > 0.50" is. **Abandon is a first-class outcome** — the cheapest experiment is the one that stops you
funding a dead direction.

## The stages

### Stage 0 — Data gate (the most important, most under-rated)
A model can only be as good as what it learned from. Gate training on the *corpus* meeting falsifiable
quality criteria **before** any GPU time:
- **Representativeness** — does it span the target's real register range? (here: formal ↔ informal,
  written ↔ transcribed speech)
- **Cleanliness & dedup** — measured exact/near-duplicate rate below a ceiling; bots/boilerplate stripped.
- **Contamination control** — e.g. AI-generated text kept out of a "human" corpus (it teaches the wrong
  distribution). See the "low-background steel" note below.
- **Target-specificity of the data itself** — is the data actually *of the target*, not generic?

If Stage 0 fails, no downstream cleverness recovers it. Spend here first and largest. In this repo the
Stage-0 machinery is [`data/`](data/).

### Stage 1 — Cheap probe run
A small, cheap adaptation (here: LoRA, ~2k steps) to test *whether the approach moves the target metric at
all*, before committing to a long run. Qualify on the harness. (Our `v0`.)

### Stage 2 — Scaled run, pre-registered
Only if Stage 1 passes. Change **one** variable, hold the rest constant, pre-register what "better" means.
(Our `v1`: 6.5× tokens, same corpus → tested the *scale* hypothesis.) **Result: falsified** — scale was not
the lever. The gate said "stop scaling, reallocate to data composition." That is a *win* of the method.

### Stage 3 — Reallocate per evidence
Spend the next unit of compute where Stage 2 pointed (here: register composition, not more tokens).

## RLHF-free qualification (the eval harness)

No human feedback loop → qualification must be automated, deterministic, and falsifiable. Four ideas do
the work (implemented in [`eval/eval_harness.py`](eval/eval_harness.py)):

1. **Held-out by construction.** Never test on trained text; probe vocabulary (slang) is held out of the corpus.

2. **Log-probability probes, not generation judging.** Score `P(target variant | context)` directly — e.g.
   does the model assign higher probability to the Australian spelling than the American one? Deterministic,
   no judge model, no prompt sensitivity.

3. **The specificity differential (the key trick).** A single absolute metric can't tell *target adaptation*
   from *general fitting*. So measure the change on the target distribution **and** on a control distribution,
   and require the target to move **far more**:
   `Δmetric(target) : Δmetric(control) ≫ 1`.
   Here: ΔPPL(Australian held-out) vs ΔPPL(non-Australian control). v0 ≈ 9:1, v1 ≈ 450:1 — proof the
   adaptation is *Australian*, not generic Reddit-format fitting.

4. **Bootstrap confidence intervals + pre-registered thresholds.** Every metric ships with a 95% CI; success
   criteria are fixed before the run (see the scorecards in [`results/`](results/)). A point estimate that
   beats a threshold but whose CI doesn't is reported as a miss.

A final **qualitative generation test** ([`eval/gen_test.py`](eval/gen_test.py)) complements the probes —
because probes can't catch degeneracy or cliché. It compares base vs tuned completions across registers and
is the gate before any release.

## Guarding against overfitting / Goodhart

- **Specificity differential** catches "improved on everything" (which usually means general overfitting, not
  target adaptation).
- **Catastrophic-forgetting check**: control-distribution perplexity must not rise above base — confirms the
  model didn't trade away general ability.
- **Held-out probes + qualitative test** catch a model that games the metrics but generates poorly.

## A note on data: "low-background steel"

Pre-2023 human text is like pre-1945 steel — uncontaminated, and no longer manufacturable once the web fills
with AI output. For a corpus whose value is *human naturalness*, dated pre-AI text is an appreciating asset,
and every new source must pass contamination filtering before joining it. (Synthetic / AI-generated datasets
belong in the *eval* as an "AI anchor", not in the training corpus.)

## Why publish the negative result

Most fine-tuning effort dies of sunk cost: people keep scaling because it *feels* productive, with no gate
that says "this isn't paying off." A pre-registered, falsifiable negative result ("more tokens didn't help")
is rarer and more useful than another hill-climb. The discipline — not any single model — is what transfers.
