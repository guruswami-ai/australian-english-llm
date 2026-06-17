#!/usr/bin/env python3
"""
Harvest Guardian Australia articles (production-office=aus) via the Open Platform
API into guardian_au.parquet — the formal/edited register for the AU corpus.

Paginated by year (avoids deep-pagination caps), page-size 200, full body text.
Key via env GUARDIAN_KEY (free developer tier ~5000 calls/day; this needs ~660).

  GUARDIAN_KEY=... ./venv/bin/python guardian_harvest.py
"""
import os, time, json, urllib.request, urllib.parse
import pyarrow as pa, pyarrow.parquet as pq

KEY = os.environ["GUARDIAN_KEY"]
OUT = "/Volumes/scratch/pushshiftdumps/guardian_au.parquet"
BASE = "https://content.guardianapis.com/search"
SCHEMA = pa.schema([("id", pa.string()), ("date", pa.string()), ("section", pa.string()),
                    ("title", pa.string()), ("source", pa.string()), ("text", pa.string())])


def fetch(params):
    url = BASE + "?" + urllib.parse.urlencode(params)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return json.load(r)["response"]
        except Exception as e:
            print(f"  retry {attempt}: {type(e).__name__}", flush=True)
            time.sleep(3 * (attempt + 1))
    return None


def main():
    writer = pq.ParquetWriter(OUT, SCHEMA, compression="zstd")
    rows, kept, seen = [], 0, 0
    for year in range(2013, 2027):
        page = 1
        while True:
            resp = fetch({"production-office": "aus", "from-date": f"{year}-01-01",
                          "to-date": f"{year}-12-31", "page-size": 200, "page": page,
                          "show-fields": "bodyText", "order-by": "oldest", "api-key": KEY})
            if not resp:
                print(f"{year} page {page}: FETCH FAILED, skipping", flush=True)
                break
            results = resp.get("results", [])
            seen += len(results)
            for a in results:
                bt = a.get("fields", {}).get("bodyText", "")
                if bt and len(bt) > 200:
                    rows.append({"id": a.get("id", ""), "date": a.get("webPublicationDate", "")[:10],
                                 "section": a.get("sectionId", ""), "title": a.get("webTitle", ""),
                                 "source": "guardian-au", "text": bt})
                    kept += 1
            pages = resp.get("pages", 1)
            print(f"{year} page {page}/{pages} | seen {seen} | kept {kept}", flush=True)
            if len(rows) >= 1500:
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA)); rows = []
            if page >= pages:
                break
            page += 1
            time.sleep(1.0)
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
    writer.close()
    print(f"DONE: seen {seen}, kept {kept} -> {OUT}")


if __name__ == "__main__":
    main()
