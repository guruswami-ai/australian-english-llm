#!/usr/bin/env python3
"""
Harvest the MODERN, FREELY-ACCESSIBLE slice of Trove digitised newspapers
(1990-2020, full text present) into trove_au.parquet.

Trove API v3. Key via env TROVE_KEY (header X-API-KEY). Filters:
  - category=newspaper
  - l-availability=y  (freely available online)
  - l-decade=199,200,201,202  (1990s..2020s; dates post-filtered to <=2020)
  - include=articleText  -> only keep articles that actually carry full text

  TROVE_KEY=... ./venv/bin/python trove_harvest.py [--probe]

--probe just prints the total full-text count per decade (do this FIRST on key
arrival to decide if the modern volume is worth a full harvest).
"""
import os, sys, time, json, urllib.request, urllib.parse
import pyarrow as pa, pyarrow.parquet as pq

KEY = os.environ.get("TROVE_KEY", "")
OUT = "/Volumes/scratch/pushshiftdumps/trove_au.parquet"
BASE = "https://api.trove.nla.gov.au/v3/result"
SCHEMA = pa.schema([("id", pa.string()), ("date", pa.string()), ("title", pa.string()),
                    ("newspaper", pa.string()), ("source", pa.string()), ("text", pa.string())])


def call(params):
    url = BASE + "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers={"X-API-KEY": KEY, "Accept": "application/json"})
    for a in range(6):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except Exception as e:
            print(f"  retry {a}: {type(e).__name__}", flush=True); time.sleep(3 * (a + 1))
    return None


def probe():
    for dec in ("199", "200", "201", "202"):
        r = call({"q": " ", "category": "newspaper", "l-availability": "y",
                  "l-decade": dec, "encoding": "json", "n": 0})
        try:
            tot = r["category"][0]["records"]["total"]
        except Exception:
            tot = "?"
        print(f"  decade {dec}0s: {tot} freely-available articles")


def harvest():
    writer = pq.ParquetWriter(OUT, SCHEMA, compression="zstd")
    rows, kept, seen = [], 0, 0
    s = "*"  # cursor
    while True:
        r = call({"q": " ", "category": "newspaper", "l-availability": "y",
                  "l-decade": ["199", "200", "201", "202"], "include": "articleText",
                  "encoding": "json", "n": 100, "s": s, "bulkHarvest": "true"})
        if not r:
            print("FETCH FAILED, stopping", flush=True); break
        cat = r["category"][0]["records"]
        arts = cat.get("article", [])
        seen += len(arts)
        for a in arts:
            txt = a.get("articleText", "")
            date = a.get("date", "")
            if txt and date and "1990" <= date[:4] <= "2020":
                # articleText is light HTML; strip tags
                import re, html
                t = html.unescape(re.sub(r"<[^>]+>", " ", txt))
                t = re.sub(r"\s+", " ", t).strip()
                if len(t) > 200:
                    rows.append({"id": str(a.get("id", "")), "date": date,
                                 "title": a.get("heading", ""), "newspaper": a.get("title", {}).get("value", "") if isinstance(a.get("title"), dict) else "",
                                 "source": "trove", "text": t})
                    kept += 1
        print(f"seen {seen} | kept {kept}", flush=True)
        if len(rows) >= 1500:
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA)); rows = []
        s = cat.get("nextStart")
        if not s:
            break
        time.sleep(0.5)
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
    writer.close()
    print(f"DONE: seen {seen}, kept {kept} (1990-2020 full-text) -> {OUT}")


if __name__ == "__main__":
    if not KEY:
        print("Set TROVE_KEY env (Trove API v3 key)."); sys.exit(1)
    probe() if "--probe" in sys.argv else harvest()
