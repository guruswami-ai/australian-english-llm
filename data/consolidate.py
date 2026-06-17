import duckdb, os, time
con = duckdb.connect()
con.execute("PRAGMA memory_limit='20GB'")
con.execute("PRAGMA threads=6")
con.execute("PRAGMA preserve_insertion_order=false")
os.makedirs('duckdb_tmp', exist_ok=True)
con.execute("PRAGMA temp_directory='/Volumes/scratch/pushshiftdumps/duckdb_tmp'")

# 9 AU subs absent from the per-subreddit dumps -> HF is their only source
MISSING = ("aussie","vic","qld","southaustralia","northernterritory",
           "aussierock","ausmusic","australianpets","aboriginal")
HF_FILTER = """body IS NOT NULL AND body NOT IN ('[deleted]','[removed]','')
               AND author NOT IN ('AutoModerator','[deleted]')
               AND lower(subreddit) IN {m}""".format(m=MISSING)

t=time.time()
print("Building consolidated corpus...")
con.execute(f"""
COPY (
  -- backbone: per-subreddit dumps (41 subs, 2005-2024, comments+submissions)
  SELECT id, subreddit, author, CAST(created_utc AS BIGINT) created_utc,
         CAST(score AS BIGINT) score, type, text
    FROM read_parquet('subreddit_parts/*.parquet')
  UNION ALL
  -- the 9 missing subs from HF (comments only), deduped within themselves
  SELECT id, subreddit, author, created_utc, score, type, text FROM (
    SELECT id, subreddit, author, CAST(created_utc AS BIGINT) created_utc,
           CAST(score AS BIGINT) score, 'comment' AS type, body AS text
      FROM read_parquet(['australian_dataset_full.parquet'], union_by_name=true)
      WHERE {HF_FILTER}
    UNION ALL
    SELECT id, subreddit, author, CAST(created_utc AS BIGINT) created_utc,
           CAST(score AS BIGINT) score, 'comment' AS type, body AS text
      FROM read_parquet('australian_parts_pre2016/*.parquet')
      WHERE {HF_FILTER}
  ) QUALIFY row_number() OVER (PARTITION BY id, type ORDER BY length(text) DESC) = 1
) TO 'australian_corpus.parquet' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)
""")
print(f"Done in {(time.time()-t)/60:.1f} min")
