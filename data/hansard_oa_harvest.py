#!/usr/bin/env python3
"""
Harvest Commonwealth Hansard (House of Representatives + Senate) from the
OpenAustralia.org Open Platform API into hansard_oa.parquet.

This is the FORMAL / INSTITUTIONAL register slice of the AU corpus: modern
parliamentary Australian English, ~2006 onward (OpenAustralia's coverage start),
capped at 2020 to stay inside the pre-AI / target window.

API: getDebates (TheyWorkForYou-style).
  - getDebates?type=T&date=D            -> day index: top-level sections (htype=10)
  - getDebates?type=T&gid=G             -> full bodies of every speech under G
Key via env OA_KEY (?key=...). 24h keys are fine; the run is resumable.

  OA_KEY=... ./venv/bin/python hansard_oa_harvest.py [--probe]

Resumable: hansard_oa_done.txt holds "TYPE DATE" lines already processed.
One output doc per top-level section (concatenated speeches), >200 chars.
"""
import os, sys, re, time, json, html, datetime, urllib.request, urllib.parse
import pyarrow as pa, pyarrow.parquet as pq

KEY = os.environ.get("OA_KEY", "")
API = "https://www.openaustralia.org.au/api/getDebates"
OUT = "/Volumes/scratch/pushshiftdumps/hansard_oa.parquet"
DONE = "/Volumes/scratch/pushshiftdumps/hansard_oa_done.txt"
TYPES = ("representatives", "senate")
START = datetime.date(2006, 1, 1)
END = datetime.date(2020, 12, 31)
SCHEMA = pa.schema([("date", pa.string()), ("chamber", pa.string()), ("gid", pa.string()),
                    ("title", pa.string()), ("source", pa.string()), ("text", pa.string())])


def call(params, tries=6):
    params = dict(params, key=KEY, output="js")
    url = API + "?" + urllib.parse.urlencode(params)
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                raw = r.read().decode("utf-8", "replace")
            d = json.loads(raw)
            if isinstance(d, dict) and d.get("error"):
                return None          # "No data to display" -> non-sitting day
            return d
        except Exception:
            time.sleep(2 * (a + 1))  # backoff on transport / rate-limit
    return None


def strip(items):
    bodies = [x.get("body", "") for x in items if isinstance(x, dict) and x.get("htype") in ("11", "12")]
    t = " ".join(bodies)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t).strip()


def probe():
    # quick coverage sanity check: count sitting days sampled across the window
    for yr in (2006, 2010, 2015, 2020):
        hits = 0
        for d in (f"{yr}-02-02", f"{yr}-03-15", f"{yr}-06-15", f"{yr}-09-15", f"{yr}-11-15"):
            for t in TYPES:
                if call({"type": t, "date": d}):
                    hits += 1
        print(f"  {yr}: {hits}/10 sampled chamber-days returned data", flush=True)


def daterange():
    d = START
    while d <= END:
        if d.weekday() < 5:          # parliament effectively never sits Sat/Sun
            yield d.isoformat()
        d += datetime.timedelta(days=1)


def main():
    done = set(open(DONE).read().splitlines()) if os.path.exists(DONE) else set()
    writer = pq.ParquetWriter(OUT, SCHEMA, compression="zstd") if not os.path.exists(OUT) \
        else pq.ParquetWriter(OUT + ".part", SCHEMA, compression="zstd")
    rows, kept, days_with_data, n = [], 0, 0, 0
    for date in daterange():
        for t in TYPES:
            tag = f"{t} {date}"
            if tag in done:
                continue
            n += 1
            idx = call({"type": t, "date": date})
            if idx:
                got = 0
                for item in idx:
                    if not (isinstance(item, dict) and "subs" in item):
                        continue            # full text lives under the htype=11 subsections
                    title = item.get("entry", {}).get("body", "")
                    for sub in item["subs"]:
                        gid = sub.get("gid")
                        if not gid:
                            continue
                        full = call({"type": t, "gid": gid})
                        if not full:
                            continue
                        txt = strip(full)
                        if len(txt) > 200:
                            rows.append({"date": date, "chamber": t, "gid": gid,
                                         "title": title, "source": f"hansard-{t[:3]}",
                                         "text": txt})
                            kept += 1; got += 1
                        time.sleep(0.2)
                if got:
                    days_with_data += 1
            with open(DONE, "a") as f:
                f.write(tag + "\n")
            if len(rows) >= 500:
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA)); rows = []
            if n % 50 == 0:
                print(f"  {n} chamber-days scanned | {days_with_data} with data | kept {kept} docs | at {date} {t}", flush=True)
            time.sleep(0.15)
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
    writer.close()
    print(f"DONE: kept {kept} docs from {days_with_data} sitting chamber-days -> {OUT}", flush=True)


if __name__ == "__main__":
    if not KEY:
        print("Set OA_KEY env (OpenAustralia API key)."); sys.exit(1)
    probe() if "--probe" in sys.argv else main()
