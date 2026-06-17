#!/usr/bin/env python3
"""
Clean raw subtitle .srt files (subs_raw/<show>/*.srt) into a `dialogue` register
slice -> subs_dialogue.parquet, ready to fold into build_register_corpus.py.

Cleaning pipeline (the value is here — raw subs are noisy):
  1. Parse .srt: drop index + timecode lines, keep cue text.
  2. Strip markup: <i>/<font> tags, {\anN} position tags, music ♪, SDH brackets
     [DOOR SLAMS] / (sighs), leading "- " dialogue dashes, ALLCAPS speaker labels.
  3. Rejoin cues: subs split sentences across timecodes -> concatenate in order,
     normalise whitespace -> flowing dialogue text per episode.
  4. English filter: drop files that aren't actually English (stopword-ratio heuristic;
     OpenSubtitles 'en' tags are unreliable).
  5. Per-show cap (--cap, default 1500 episodes/show) to stop one serial's
     catchphrases/proper-nouns dominating the register.
  6. Exact-dedup (recap intros / duplicate uploads) and drop too-short.

Output schema {id, show, season, episode, source='subtitle', register='dialogue', text}.

  ./venv/bin/python subs_clean.py [--cap 1500]
"""
import os, re, sys, glob, hashlib
import pyarrow as pa, pyarrow.parquet as pq

RAW = "/Volumes/scratch/pushshiftdumps/subs_raw"
OUT = "/Volumes/scratch/pushshiftdumps/subs_dialogue.parquet"
SCHEMA = pa.schema([("id", pa.string()), ("show", pa.string()), ("season", pa.string()),
                    ("episode", pa.string()), ("source", pa.string()),
                    ("register", pa.string()), ("text", pa.string())])

EN_STOP = set("the a an and or but if of to in is are was were be been have has it that this you i he she they we".split())
TAG = re.compile(r"<[^>]+>|\{\\[^}]+\}")
SDH = re.compile(r"\[[^\]]*\]|\([^)]*\)|♪[^♪]*♪?|[♪#]")
SPK = re.compile(r"^\s*[-\s]*[A-Z][A-Z .']{1,20}:\s*")   # ALLCAPS speaker label "STEVE:"
TIME = re.compile(r"-->")


def parse_srt(path):
    try:
        raw = open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        return ""
    raw = raw.lstrip("﻿")
    lines = []
    for block in re.split(r"\n\s*\n", raw):
        for ln in block.splitlines():
            ln = ln.strip()
            if not ln or ln.isdigit() or TIME.search(ln):
                continue
            ln = TAG.sub(" ", ln)
            ln = SDH.sub(" ", ln)
            ln = SPK.sub("", ln)
            ln = re.sub(r"^-\s*", "", ln)          # leading dialogue dash
            ln = re.sub(r"\s+", " ", ln).strip()
            if ln:
                lines.append(ln)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def is_english(t):
    words = re.findall(r"[a-z']+", t.lower())
    if len(words) < 50:
        return False
    return sum(w in EN_STOP for w in words) / len(words) > 0.06


def main():
    cap = 1500
    if "--cap" in sys.argv:
        cap = int(sys.argv[sys.argv.index("--cap") + 1])
    if not os.path.isdir(RAW):
        print(f"No {RAW}/ — run subs_harvest.py first."); return
    rows, seen, pershow = [], set(), {}
    kept = dropped = 0
    for path in sorted(glob.glob(f"{RAW}/*/*.srt")):
        show = os.path.basename(os.path.dirname(path))
        m = re.search(r"s(\w+?)e(\w+?)_", os.path.basename(path))
        season, episode = (m.group(1), m.group(2)) if m else ("", "")
        text = parse_srt(path)
        if len(text) < 600 or not is_english(text):
            dropped += 1; continue
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen:
            dropped += 1; continue
        if pershow.get(show, 0) >= cap:
            dropped += 1; continue
        seen.add(h); pershow[show] = pershow.get(show, 0) + 1
        rows.append({"id": h[:16], "show": show, "season": season, "episode": episode,
                     "source": "subtitle", "register": "dialogue", "text": text})
        kept += 1
    if not rows:
        print(f"nothing kept (dropped {dropped})."); return
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), OUT, compression="zstd")
    chars = sum(len(r["text"]) for r in rows)
    print(f"kept {kept} episodes, dropped {dropped} -> {OUT}")
    print(f"  per-show: {pershow}")
    print(f"  {chars/1e6:.1f} Mchars (~{chars/4e6:.0f}M tokens rough)")


if __name__ == "__main__":
    main()
