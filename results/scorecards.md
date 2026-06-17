# Pre-registered scorecards

Every criterion below was fixed **before** the training run (pre-registration). Results are reported as
measured — passes and failures both. Base-vs-tuned on held-out data, adapter toggled so both see identical
inputs, bootstrap 95% CIs.

---

## v0 — Qwen3-8B (instruct) + LoRA, ~2,000 steps → ✅ QUALIFIED

Pre-registered success: *AU-specificity holds AND at least one of {spelling, slang, AU-PPL} improves beyond its CI.*

| Metric | Base | v0 | Δ |
|---|---|---|---|
| AU spelling preferred | 0.360 | 0.600 | +0.240 |
| P(AU spelling) | 0.384 | 0.592 | +0.208 |
| Slang comprehension (chance 0.25) | 0.342 | 0.483 | +0.141 |
| PPL held-out AU | 61.80 | 29.27 | −32.53 |
| PPL non-AU control (wikitext) | 14.52 | 10.90 | −3.62 |

**AU-specificity:** ΔPPL(AU) −32.5 vs ΔPPL(non-AU) −3.6 → **~9:1**. The adaptation is Australian-specific,
not general fitting. **Verdict: PASS.**

---

## v1 — Qwen3-8B-Base + LoRA, full epoch (13,105 steps, 51.7 h) → ❌ MISSED BAR (2/5)

Hypothesis under test: *scale (6.5× tokens) + training from base (not instruct) beats v0.*

| # | Pre-registered criterion | Result | |
|---|---|---|---|
| 1 | AU-specificity ΔPPL(AU):ΔPPL(non-AU) ≥ 3:1 | −31.5 : −0.07 ≈ **450:1** | ✅ |
| 2 | AU PPL ≤ 27 (beat v0's 29.27) | **32.48** | ❌ |
| 3 | P(AU spelling) ≥ 0.60 AND CI-low > 0.50 | 0.591, CI[0.498, 0.688] | ❌ |
| 4 | Slang ≥ 0.48 | **0.417** | ❌ |
| 5 | No forgetting: PPL(non-AU) ≤ base | 10.02 (base 10.08) | ✅ |

Pre-registration required **all five** → **verdict: the scale hypothesis is NOT supported.**

### What it means (the value is here)
- **Relative** adaptation is ~identical to v0: v0 61.8→29.3 (×0.47), v1 64.0→32.5 (×0.51). **6.5× more
  tokens did not beat v0's short run** — v0 had already captured most of the available gain.
  → **Token scale is not the lever. Reallocate to data composition / registers** (the v2 direction).
- The absolute targets (≤27 PPL, ≥0.48 slang) were calibrated on v0, which used a *different base*
  (instruct, easier starting AU-PPL 61.8) than v1 (base, 64.0). **Lesson: re-baseline eval targets per
  base model** — comparing absolutes across bases conflates base-swap with the variable under test.
- v0 remains the stronger *absolute* model; v1 is the cleaner *base* foundation. Both inform v2.

This is the methodology working: a 51-hour experiment bought the evidence to **stop** funding scale.
