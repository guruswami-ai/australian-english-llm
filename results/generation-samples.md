# Qualitative generation samples (2026-06-17)

The pre-registered probes can't catch degeneracy or cliché, so before any release we **look at actual
generations**. Completion-style (these are continued-pretraining base models, not chat), temperature 0.8,
same prompt to base vs tuned with the adapter toggled. Reproduce with [`eval/gen_test.py`](../eval/gen_test.py).
Verdict: **coherent, non-degenerate, believable (non-cliché) Australian English — gate PASSED.**

The thesis in action — *the Australian-ness is local knowledge, register, and spelling, never ocker cliché:*

### "The weather in Melbourne this time of year is famously…"
- **Generic Qwen3-8B:** *"…hot and humid."* — wrong (Melbourne is famously *changeable*); a second base run
  veered into a Markov-chain maths tangent.
- **v0:** *"…unreliable and can change from one day to another. So I always recommend that if you have a
  choice, choose the dry days for your outdoor activities…"*
- **v1:** *"…changeable. The forecast for the day we flew to Adelaide was 10-25° and it actually reached
  about -4°C (it also snowed on Mt Gambier…)."*

The tuned models know the *local reality*; the generic base reaches for a generic (and wrong) answer.

### "The Reserve Bank of Australia announced today that interest rates would…" (news register)
- **v0:** *"…be kept unchanged at 0.5 per cent, as expected by most market analysts. However the RBA did not
  give a firm indication about how long this low rate environment will last…"*

Reads like genuine Australian financial journalism: correct institution, register, and spelling (*per cent*).

### "She picked up her favourite jumper, the colour of which…" (spelling)
- **Generic Qwen3-8B:** drifted into a reasoning-mode monologue (*"Let me start by breaking down the
  problem…"*); one base run emitted Chinese.
- **v0 / v1:** stayed coherent and kept Australian spelling — *favourite, colour(s)*.

### Informal register
- **v1:** *"This bloke is out to get me! He's going through my pockets — he doesn't know how hard it's been
  for this young family of four…"* — natural informal AU, no stereotype.
- **v0:** *"it's going to be a very close contest between two decent teams… I think we might get 240k in
  attendance."* — plausible AFL register, concise.

## Honest read

- **No ocker clichés** in any output — exactly the goal. AU-ness emerges as register + spelling + local
  knowledge.
- Both models consistently use Australian/British spelling (*favourite, colour, organisation, per cent*).
- **Caveat:** these are thin LoRA adapters over base models — they ramble, don't follow instructions, and
  occasionally bleed pre-training formats (Q&A, multiple-choice). Expected for continued-pretraining; stated
  plainly. v0 is noticeably more concise/controlled than v1.
