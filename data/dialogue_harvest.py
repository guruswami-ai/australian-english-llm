#!/usr/bin/env python3
"""
Harvest the `dialogue` register (spoken-conversational AU) from two free fan archives:

  * HOME AND AWAY  — backtothebay.net  /episodesarchive/scripts/<year>/<ep>.shtml
      Real VERBATIM scripts: <td id="script"> with <h3>SPEAKER</h3><p>line</p>,
      <h2>stage direction</h2>, <h3>SCENE HEADING</h3>. High quality, continuous
      multi-turn dialogue. BUT coverage is SPARSE/patchy (not every episode exists),
      so we probe each episode and keep the hits. Page encoding is cp1252.
  * NEIGHBOURS     — neighboursepisodes.com  index.php?year=<y>&episode=<ep>
      Fan RECAPS (1985-2008, ~3822 eps) with BRIEF quoted dialogue embedded in
      editorialised summary prose. We extract only the quoted dialogue (the clean
      spoken bits); the recap prose is a different/quirky register and is skipped.

Episode lists come from each site's sitemaps (no guessing). Output ->
dialogue_<show>.parquet {id, show, source, register='dialogue', year, episode, text}.
Resumable via dialogue_done_<show>.txt. Polite ~0.4s pacing, custom UA.

  ./venv/bin/python dialogue_harvest.py --probe           # coverage + yield, no save
  ./venv/bin/python dialogue_harvest.py --show haa        # harvest Home & Away
  ./venv/bin/python dialogue_harvest.py --show neighbours # harvest Neighbours
"""
import os, re, sys, time, html, urllib.request, urllib.error
import pyarrow as pa, pyarrow.parquet as pq

UA = {"User-Agent": "aussie-llm-corpus/0.1 (research; contact paul)"}
BASE = "/Volumes/scratch/pushshiftdumps"
SCHEMA = pa.schema([("id", pa.string()), ("show", pa.string()), ("source", pa.string()),
                    ("register", pa.string()), ("year", pa.int32()),
                    ("episode", pa.int32()), ("text", pa.string())])


def fetch(url, enc="utf-8", tries=4):
    for a in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
                return r.read().decode(enc, "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (a + 1))
        except Exception:
            time.sleep(2 * (a + 1))
    return None


# ---------- Home and Away (backtothebay) ----------
def haa_episode_years():
    """ep_number -> air year, from backtothebay episode sitemaps."""
    pairs = {}
    for i in [""] + [str(n) for n in range(2, 12)]:
        sm = fetch(f"https://www.backtothebay.net/bttb_episodes-sitemap{i}.xml")
        if not sm:
            break
        for y, ep in re.findall(r"/episodes/(\d{4})/episode-(\d+)/", sm):
            pairs.setdefault(int(ep), int(y))
    return pairs


def haa_parse(htmltext):
    """Pull <td id='script'> and turn h3=SPEAKER / p=line into 'SPEAKER: line' turns."""
    m = re.search(r'id="script">(.*?)</td>', htmltext, re.S)
    if not m:
        return ""
    body = m.group(1)
    out, speaker = [], None
    for tag, content in re.findall(r"<(h3|h2|p)[^>]*>(.*?)</\1>", body, re.S):
        t = html.unescape(re.sub(r"<[^>]+>", " ", content))
        t = re.sub(r"\s+", " ", t).strip()
        if not t:
            continue
        if tag == "h3":
            speaker = t                      # speaker name OR scene heading
        elif tag == "p":
            out.append(f"{speaker}: {t}" if speaker and speaker.isupper() else t)
            speaker = None
        # h2 (stage directions) skipped — keep pure dialogue
    return "\n".join(out).strip()


def harvest_haa(probe=False):
    pairs = haa_episode_years()
    print(f"HAA: {len(pairs)} episodes in guide ({min(pairs)}-{max(pairs)})", flush=True)
    done_path = f"{BASE}/dialogue_done_haa.txt"
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    rows, kept, hits, scanned = [], 0, 0, 0
    writer = None if probe else pq.ParquetWriter(f"{BASE}/dialogue_haa.parquet", SCHEMA, compression="zstd")
    for ep in sorted(pairs):
        if str(ep) in done:
            continue
        y = pairs[ep]
        scanned += 1
        h = fetch(f"https://www.backtothebay.net/episodesarchive/scripts/{y}/{ep}.shtml", enc="cp1252")
        if h:
            txt = haa_parse(h)
            if len(txt) > 400:
                hits += 1
                if not probe:
                    rows.append({"id": f"haa-{ep}", "show": "home_and_away", "source": "backtothebay",
                                 "register": "dialogue", "year": y, "episode": ep, "text": txt})
                    kept += 1
        if not probe:
            with open(done_path, "a") as f:
                f.write(str(ep) + "\n")
        if rows and len(rows) >= 300:
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA)); rows = []
        if scanned % 200 == 0:
            print(f"  scanned {scanned} | scripts found {hits}", flush=True)
        if probe and scanned >= 400:           # probe = sample first 400 for a hit-rate
            break
        time.sleep(0.35)
    if probe:
        print(f"  PROBE: {hits}/{scanned} episodes have scripts ({100*hits/max(scanned,1):.0f}% hit rate)", flush=True)
        return
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
    writer.close()
    print(f"HAA DONE: kept {kept} scripts -> dialogue_haa.parquet", flush=True)


# ---------- Neighbours (neighboursepisodes) ----------
def neighbours_urls():
    sm = fetch("https://www.neighboursepisodes.com/sitemap.xml")
    locs = [l.replace("&amp;", "&") for l in re.findall(r"<loc>([^<]+)</loc>", sm or "") if "episode=" in l]
    return locs


def neighbours_quotes(htmltext):
    """Extract quoted spoken dialogue from the recap body (brief but clean)."""
    b = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", htmltext, flags=re.S)
    txt = html.unescape(re.sub(r"<[^>]+>", " ", b))
    qs = re.findall(r'"([^"]{4,400}?)"', txt)
    qs = [re.sub(r"\s+", " ", q).strip() for q in qs]
    qs = [q for q in qs if "http" not in q and "=" not in q and len(q.split()) >= 2]
    return "\n".join(qs)


def harvest_neighbours(probe=False):
    urls = neighbours_urls()
    print(f"NEIGHBOURS: {len(urls)} episode pages in sitemap", flush=True)
    done_path = f"{BASE}/dialogue_done_neighbours.txt"
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    rows, kept, chars, scanned = [], 0, 0, 0
    writer = None if probe else pq.ParquetWriter(f"{BASE}/dialogue_neighbours.parquet", SCHEMA, compression="zstd")
    for u in urls:
        ep = re.search(r"episode=(\d+)", u); yr = re.search(r"year=(\d+)", u)
        if not ep:
            continue
        ep, yr = int(ep.group(1)), int(yr.group(1)) if yr else 0
        if str(ep) in done:
            continue
        scanned += 1
        h = fetch(u)
        if h:
            txt = neighbours_quotes(h)
            if len(txt) > 120:
                chars += len(txt)
                if not probe:
                    rows.append({"id": f"neigh-{ep}", "show": "neighbours", "source": "neighboursepisodes",
                                 "register": "dialogue", "year": yr, "episode": ep, "text": txt})
                    kept += 1
        if not probe:
            with open(done_path, "a") as f:
                f.write(str(ep) + "\n")
        if rows and len(rows) >= 300:
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA)); rows = []
        if probe and scanned >= 60:
            print(f"  PROBE: {scanned} eps sampled, ~{chars/scanned:.0f} dialogue chars/ep "
                  f"-> ~{chars/scanned*len(urls)/4e6:.1f}M tokens projected total", flush=True)
            return
        if scanned % 200 == 0:
            print(f"  scanned {scanned} | kept {kept}", flush=True)
        time.sleep(0.35)
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
    writer.close()
    print(f"NEIGHBOURS DONE: kept {kept} eps, {chars/1e6:.1f}M dialogue chars -> dialogue_neighbours.parquet", flush=True)


if __name__ == "__main__":
    probe = "--probe" in sys.argv
    show = sys.argv[sys.argv.index("--show") + 1] if "--show" in sys.argv else "all"
    if probe:
        harvest_haa(probe=True); harvest_neighbours(probe=True)
    elif show in ("haa", "home_and_away"):
        harvest_haa()
    elif show in ("neighbours",):
        harvest_neighbours()
    else:
        print("use --probe, or --show haa|neighbours")
