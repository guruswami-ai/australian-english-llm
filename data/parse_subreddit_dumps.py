#!/usr/bin/env python3
"""
Parse Watchful1 per-subreddit .zst dumps (pushshift NDJSON) into a clean,
consolidated parquet corpus for LLM language training.

For each subreddits24/*.zst:
  - stream-decompress (zstd --long=31), parse NDJSON
  - normalise comments & submissions to one schema
  - apply training-quality filters (drop deleted/removed/empty, AutoModerator)
  - write one parquet per input file into PARTS_DIR (crash-safe)

Consolidated schema: id, subreddit, author, created_utc, score, type, text

Run:   ./venv/bin/python parse_subreddit_dumps.py
"""
import os, sys, json, subprocess, glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow as pa
import pyarrow.parquet as pq

BASE = "/Volumes/scratch/pushshiftdumps"
IN_DIR = f"{BASE}/reddit/subreddits24"
PARTS_DIR = f"{BASE}/subreddit_parts"
WORKERS = 6
BATCH = 50_000          # rows per parquet row-group flush

DROP_BODIES = {"[deleted]", "[removed]", "", None}
DROP_AUTHORS = {"AutoModerator", "[deleted]"}   # [deleted] author = no attribution; text often still ok but skip bots

# Only parse our Australian subreddits — the torrent pulled in neighbour files
# via shared-piece boundaries, so glob() alone would include non-AU junk.
AU_SUBS = {
    "australia", "australian", "aussie", "askanaustralian", "aussiecasual", "straya", "ausmemes",
    "sydney", "melbourne", "brisbane", "perth", "adelaide", "canberra", "hobart", "darwin",
    "newcastle", "wollongong", "geelong", "goldcoast", "townsville", "cairns",
    "nsw", "vic", "qld", "westernaustralia", "southaustralia", "tasmania", "northernterritory",
    "ausfinance", "australianpolitics", "auspropertychat", "auslegal", "auspublicservice",
    "afl", "nrl", "aleague", "v8supercars", "cricket",
    "australianwildlife", "australianbirds", "australianplants", "gardeningaustralia",
    "carsaustralia", "fishingaustralia", "aussierock", "ausmusic", "australianpets",
    "circlejerkaustralia", "friendlyjordies", "aboriginal",
}


def _sub_of(fname):
    base = fname.replace(".zst", "")
    for suf in ("_comments", "_submissions"):
        if base.endswith(suf):
            return base[: -len(suf)].lower()
    return base.lower()

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("subreddit", pa.string()),
    ("author", pa.string()),
    ("created_utc", pa.int64()),
    ("score", pa.int64()),
    ("type", pa.string()),
    ("text", pa.string()),
])


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_file(path):
    fname = os.path.basename(path)
    is_comment = "_comments" in fname
    ftype = "comment" if is_comment else "submission"
    out = f"{PARTS_DIR}/{fname.replace('.zst', '')}.parquet"
    if os.path.exists(out):
        return fname, "skip(exists)", 0

    cols = {k: [] for k in SCHEMA.names}
    writer = None
    kept = total = 0

    def flush():
        nonlocal writer, cols
        if not cols["id"]:
            return
        tbl = pa.table(cols, schema=SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(out, SCHEMA, compression="zstd")
        writer.write_table(tbl)
        cols = {k: [] for k in SCHEMA.names}

    proc = subprocess.Popen(["zstd", "-dc", "--long=31", path],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 20)
    try:
        for line in proc.stdout:
            total += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            author = d.get("author")
            if author in DROP_AUTHORS:
                continue
            if is_comment:
                text = d.get("body")
                if text in DROP_BODIES:
                    continue
            else:
                title = (d.get("title") or "").strip()
                selftext = d.get("selftext") or ""
                if selftext in ("[deleted]", "[removed]"):
                    selftext = ""
                if not title and not selftext.strip():
                    continue
                text = (title + "\n\n" + selftext).strip()
            cols["id"].append(d.get("id"))
            cols["subreddit"].append(d.get("subreddit"))
            cols["author"].append(author)
            cols["created_utc"].append(_int(d.get("created_utc")))
            cols["score"].append(_int(d.get("score")))
            cols["type"].append(ftype)
            cols["text"].append(text)
            kept += 1
            if len(cols["id"]) >= BATCH:
                flush()
        flush()
    finally:
        proc.stdout.close()
        proc.wait()
    if writer is not None:
        writer.close()
    return fname, f"{kept}/{total}", kept


def main():
    os.makedirs(PARTS_DIR, exist_ok=True)
    all_zst = sorted(glob.glob(f"{IN_DIR}/*.zst"))
    files = [f for f in all_zst if _sub_of(os.path.basename(f)) in AU_SUBS]
    if not files:
        print(f"No AU .zst files in {IN_DIR}")
        return
    print(f"Parsing {len(files)} AU files (of {len(all_zst)} present) with {WORKERS} workers...")
    grand = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(parse_file, f): f for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            fname, stat, kept = fut.result()
            grand += kept
            print(f"  [{i}/{len(files)}] {fname:40} kept {stat}")
    print(f"\nDone. {grand:,} rows across {len(files)} files -> {PARTS_DIR}")


if __name__ == "__main__":
    main()
