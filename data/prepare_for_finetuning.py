#!/usr/bin/env python3
"""
Stage 1 of CPT prep for an Australian Gemma-2 model.

Input : australian_corpus.parquet  (id, subreddit, author, created_utc, score, type, text)
Output: au_cpt_train.parquet / au_cpt_val.parquet  (cleaned `text` + metadata)

Steps: clean text -> quality filter -> exact-text dedup -> rebalance sport to ~25%
       -> time-based train/val split (val = Nov+Dec 2024).
Tokenization/packing for Gemma 2 is a separate stage (tokenize_pack_gemma.py).
"""
import duckdb, os

BASE = "/Volumes/scratch/pushshiftdumps"
SRC = f"{BASE}/australian_corpus.parquet"
SPORT = ("cricket", "afl", "nrl", "aleague", "v8supercars")
SPORT_TARGET_SHARE = 0.25
MIN_CHARS = 30
VAL_CUTOFF = 1730419200  # 2024-11-01 UTC: rows on/after this -> validation

con = duckdb.connect()
con.execute("PRAGMA threads=8")
con.execute("PRAGMA disable_progress_bar")
con.execute("PRAGMA preserve_insertion_order=false")
os.makedirs(f"{BASE}/duckdb_tmp", exist_ok=True)
con.execute(f"PRAGMA temp_directory='{BASE}/duckdb_tmp'")

# --- cleaned-text expression: unescape HTML entities + strip zero-width; &amp; last ---
def clean_expr(col):
    e = col
    for a, b in [("chr(8203)", "''"), ("'&#x200B;'", "''"), ("'&nbsp;'", "' '"),
                 ("'&lt;'", "'<'"), ("'&gt;'", "'>'"), ("'&quot;'", "'\"'"),
                 ("'&#39;'", "''''"), ("'&amp;'", "'&'")]:
        e = f"replace({e}, {a}, {b})"
    return f"trim({e})"

CLEAN = clean_expr("text")
BOT = "lower(text) NOT LIKE '%i am a bot%' AND lower(text) NOT LIKE '%performed automatically%'"

# Base cleaned + filtered + exact-text deduped view (dedup via hash GROUP BY = spills well)
con.execute(f"""
CREATE TEMP TABLE clean AS
SELECT any_value(id) id, any_value(lower(subreddit)) subreddit, any_value(author) author,
       any_value(created_utc) created_utc, any_value(score) score, any_value("type") AS "type",
       ct AS text,
       (any_value(lower(subreddit)) IN {SPORT}) AS is_sport
FROM (
  SELECT id, subreddit, author, created_utc, score, "type", {CLEAN} AS ct
  FROM read_parquet('{SRC}')
  WHERE {BOT}
) WHERE length(ct) >= {MIN_CHARS}
GROUP BY ct
""")

n_total, = con.execute("SELECT count(*) FROM clean").fetchone()
n_sport, = con.execute("SELECT count(*) FROM clean WHERE is_sport").fetchone()
n_ns = n_total - n_sport
# keep fraction so sport ends at SPORT_TARGET_SHARE of the final set
target_sport = SPORT_TARGET_SHARE / (1 - SPORT_TARGET_SHARE) * n_ns
frac = min(1.0, target_sport / n_sport) if n_sport else 1.0
print(f"after clean+dedup: {n_total:,} rows  (sport {n_sport:,} = {100*n_sport/n_total:.0f}%)")
print(f"sport keep fraction -> {frac:.3f}")

con.execute("SELECT setseed(0.42)")
KEEP = f"(NOT is_sport OR random() < {frac})"

for name, where in [("train", f"created_utc < {VAL_CUTOFF}"),
                    ("val",   f"created_utc >= {VAL_CUTOFF}")]:
    out = f"{BASE}/au_cpt_{name}.parquet"
    con.execute(f"""
      COPY (SELECT id, subreddit, author, created_utc, score, "type", text
            FROM clean WHERE {KEEP} AND {where})
      TO '{out}' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)
    """)
    n, ch = con.execute(f"SELECT count(*), sum(length(text)) FROM read_parquet('{out}')").fetchone()
    print(f"{name}: {n:,} rows, ~{ch/1e9:.1f}B chars (~{ch/4/1e9:.2f}B tokens) -> {os.path.basename(out)}")

print("\nFinal subreddit mix (train):")
for s, c in con.execute(f"""
    SELECT subreddit, count(*) c FROM read_parquet('{BASE}/au_cpt_train.parquet')
    GROUP BY 1 ORDER BY 2 DESC LIMIT 10""").fetchall():
    print(f"   r/{s:20} {c:>11,}")
