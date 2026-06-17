#!/usr/bin/env python3
"""
Build the REGISTER-CONDITIONED corpus for v2 — a single doc-level parquet where
every document carries a `register` control-code, spanning the formality spectrum:

    vernacular  ->  casual  ->  news  ->  formal
    (ocker Reddit) (Reddit)   (Guardian) (Hansard)

Design:
  * Doc-level only. NO balancing/capping here — keep every doc; register balance
    is a PACK-TIME knob for v2 (depends on final per-register token counts).
  * Tokenizer-agnostic cleaned text (matches the existing pipeline): switching base
    models = repack only. The control code is applied at pack time from `register`.
  * Each source is filtered to the 1990-2020 window and cached to its own
    *_register.parquet, so the heavy 61M-row Reddit pass runs ONCE. Re-running skips
    any per-source file that already exists, then re-merges. Safe to re-run as the
    Guardian/Hansard harvests finish.

Output schema: {id, source, register, date(YYYY-MM-DD), year(int), text}

  ./venv/bin/python build_register_corpus.py            # build all ready sources + merge
  ./venv/bin/python build_register_corpus.py --merge    # just (re)merge existing per-source files
"""
import os, sys, duckdb
import pyarrow.parquet as pq

BASE = "/Volumes/scratch/pushshiftdumps"
REDDIT_IN   = f"{BASE}/au_cpt_train.parquet"
GUARDIAN_IN = f"{BASE}/guardian_au.parquet"
HANSARD_IN  = f"{BASE}/hansard_oa.parquet"
OUT         = f"{BASE}/register_corpus.parquet"

# Reddit register split. `vernacular` = the ocker/slang/satirical informal extreme
# (closest Reddit proxy for "bogan"; a cleaner bogan anchor will later come from the
# slang glossary + film transcripts). Everything else Reddit = `casual`. Easy to tune.
VERNACULAR = {"straya", "aussiecasual", "ausmemes", "circlejerkaustralia",
              "aussierock", "ausmusic", "aussie"}
WINDOW = (2008, 2020)   # Reddit/Guardian; Hansard is already 2006-2020


def readable(path):
    """True only if the parquet has a valid footer (i.e. its harvest has finished)."""
    if not os.path.exists(path):
        return False
    try:
        pq.ParquetFile(path).metadata
        return True
    except Exception:
        return False


def build_reddit(con):
    out = f"{BASE}/reddit_register.parquet"
    if os.path.exists(out):
        print(f"  reddit: cached ({out})"); return out
    vern = ",".join("'%s'" % s for s in VERNACULAR)
    con.execute(f"""
        COPY (
          SELECT id,
                 'reddit' AS source,
                 CASE WHEN subreddit IN ({vern}) THEN 'vernacular' ELSE 'casual' END AS register,
                 strftime(to_timestamp(created_utc), '%Y-%m-%d') AS date,
                 year(to_timestamp(created_utc)) AS year,
                 text
          FROM '{REDDIT_IN}'
          WHERE year(to_timestamp(created_utc)) BETWEEN {WINDOW[0]} AND {WINDOW[1]}
            AND text IS NOT NULL AND length(text) > 0
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
    """)
    print(f"  reddit: built {out}"); return out


def build_guardian(con):
    out = f"{BASE}/guardian_register.parquet"
    if os.path.exists(out):
        print(f"  guardian: cached"); return out
    if not readable(GUARDIAN_IN):
        print("  guardian: SKIP (harvest not finished / parquet unreadable)"); return None
    con.execute(f"""
        COPY (
          SELECT id, 'guardian' AS source, 'news' AS register,
                 date, CAST(substr(date,1,4) AS INT) AS year, text
          FROM '{GUARDIAN_IN}'
          WHERE substr(date,1,4) BETWEEN '{WINDOW[0]}' AND '{WINDOW[1]}'
            AND text IS NOT NULL AND length(text) > 200
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
    """)
    print(f"  guardian: built {out}"); return out


def build_hansard(con):
    out = f"{BASE}/hansard_register.parquet"
    if os.path.exists(out):
        print(f"  hansard: cached"); return out
    if not readable(HANSARD_IN):
        print("  hansard: SKIP (harvest not finished / parquet unreadable)"); return None
    con.execute(f"""
        COPY (
          SELECT gid AS id, 'hansard' AS source, 'formal' AS register,
                 date, CAST(substr(date,1,4) AS INT) AS year, text
          FROM '{HANSARD_IN}'
          WHERE text IS NOT NULL AND length(text) > 200
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
    """)
    print(f"  hansard: built {out}"); return out


def merge(con):
    parts = [p for p in (f"{BASE}/reddit_register.parquet",
                         f"{BASE}/guardian_register.parquet",
                         f"{BASE}/hansard_register.parquet") if os.path.exists(p)]
    if not parts:
        print("nothing to merge"); return
    union = " UNION ALL BY NAME ".join(f"SELECT * FROM '{p}'" for p in parts)
    con.execute(f"COPY ({union}) TO '{OUT}' (FORMAT parquet, COMPRESSION zstd)")
    print(f"\nMERGED {len(parts)} source(s) -> {OUT}")
    print("=== register breakdown (docs / Mchars) ===")
    for r in con.execute(f"""
        SELECT register, source, count(*) docs, round(sum(length(text))/1e6,1) Mchars
        FROM '{OUT}' GROUP BY 1,2 ORDER BY 4 DESC""").fetchall():
        print(f"  {r[0]:11} {r[1]:9} {r[2]:>12,} docs  {r[3]:>9} Mchars")
    tot = con.execute(f"SELECT count(*), round(sum(length(text))/1e6,1) FROM '{OUT}'").fetchone()
    print(f"  {'TOTAL':21} {tot[0]:>12,} docs  {tot[1]:>9} Mchars  (~{tot[1]/4:.0f}M tokens rough)")


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    if "--merge" not in sys.argv:
        print("building per-source register slices...")
        build_reddit(con)
        build_guardian(con)
        build_hansard(con)
    merge(con)


if __name__ == "__main__":
    main()
