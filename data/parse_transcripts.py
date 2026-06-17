#!/usr/bin/env python3
"""
Parse Australian film transcripts (foreverdreaming.org HTML export) into clean
dialogue text for the AU-English corpus.

Each .html -> one document: stripped of page chrome, <br> -> newlines, tags
removed, HTML entities unescaped, common asterisk-censored profanity restored
(film dialogue is authentic AU speech — we don't want '*'-mangled tokens).

Output: transcripts.parquet  (title, year, source, text)
Run:    ./venv/bin/python parse_transcripts.py [glob ...]   (default: *.html in cwd)
"""
import sys, os, re, glob, html, csv
import pyarrow as pa
import pyarrow.parquet as pq

BASE = "/Volumes/scratch/pushshiftdumps"
OUT = f"{BASE}/transcripts.parquet"
CSV = f"{BASE}/australian_film.csv"

# foreverdreaming censors profanity with '*'; restore for authentic AU dialogue
UNCENSOR = {
    r"\bsh\*t\b": "shit", r"\bf\*ck\b": "fuck", r"\bf\*cking\b": "fucking",
    r"\bf\*cked\b": "fucked", r"\bb\*mb\b": "bomb", r"\bg\*ns\b": "guns",
    r"\ba\*s\b": "ass", r"\ba\*tillery\b": "artillery", r"\bc\*nt\b": "cunt",
    r"\bbullsh\*t\b": "bullshit", r"\bd\*ck\b": "dick", r"\bb\*tch\b": "bitch",
    r"\bpr\*ck\b": "prick", r"\bw\*nker\b": "wanker",
}

CONTENT_RE = re.compile(r'<div class="content">(.*?)</div>', re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")


def load_year_map():
    """title(lower) -> year, from the IMDb CSV (best-effort)."""
    m = {}
    if os.path.exists(CSV):
        with open(CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for key in (row.get("Title"), row.get("Original Title")):
                    if key:
                        m[key.strip().lower()] = row.get("Year", "")
    return m


def clean_block(htmltext):
    t = htmltext.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    t = TAG_RE.sub("", t)            # drop <em>, <a>, etc. but keep their text
    t = html.unescape(t)
    for pat, repl in UNCENSOR.items():
        t = re.sub(pat, repl, t, flags=re.I)
    # tidy: collapse 3+ blank lines to one, strip trailing spaces
    lines = [ln.rstrip() for ln in t.split("\n")]
    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def title_from(path, raw):
    m = re.search(r"<h2>(.*?)</h2>", raw, re.S | re.I) or re.search(r"<h3>(.*?)</h3>", raw, re.S | re.I)
    if m:
        return html.unescape(TAG_RE.sub("", m.group(1)).strip())
    return re.sub(r"\s*\(\d{4}\)$", "", os.path.splitext(os.path.basename(path))[0]).strip()


def main():
    files = []
    for g in (sys.argv[1:] or ["*.html"]):
        files += glob.glob(os.path.join(BASE, g)) if not os.path.isabs(g) else glob.glob(g)
    files = sorted(set(files))
    if not files:
        print("No .html transcript files found.")
        return
    years = load_year_map()
    rows = []
    for p in files:
        raw = open(p, encoding="utf-8", errors="replace").read()
        blocks = CONTENT_RE.findall(raw)
        if not blocks:
            print(f"  skip (no content div): {os.path.basename(p)}")
            continue
        text = "\n\n".join(clean_block(b) for b in blocks).strip()
        if len(text) < 500:
            print(f"  skip (too short): {os.path.basename(p)}")
            continue
        title = title_from(p, raw)
        ym = re.search(r"\((\d{4})\)", os.path.basename(p))
        year = (ym.group(1) if ym else years.get(title.lower(), ""))
        rows.append({"title": title, "year": str(year), "source": "film", "text": text})
        print(f"  {title} ({year}): {len(text):,} chars (~{len(text)//4:,} tokens)")

    if rows:
        schema = pa.schema([("title", pa.string()), ("year", pa.string()),
                            ("source", pa.string()), ("text", pa.string())])
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), OUT, compression="zstd")
        chars = sum(len(r["text"]) for r in rows)
        print(f"\n{len(rows)} transcripts -> {OUT}  (~{chars/1e6:.1f}M chars, ~{chars//4/1e6:.1f}M tokens)")


if __name__ == "__main__":
    main()
