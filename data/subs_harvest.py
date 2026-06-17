#!/usr/bin/env python3
"""
Harvest English subtitles for long-running Australian TV serials from the
OpenSubtitles.com REST API -> raw .srt files under subs_raw/<show>/.
This is the SPOKEN-CONVERSATIONAL register the corpus lacks (Reddit=written-informal,
Guardian=written-edited, Hansard=spoken-formal; serials=spoken-informal).

STATUS: DRAFT, pending an OpenSubtitles API key. Run --probe FIRST (search-only,
no downloads, ~free) to measure REAL per-show available episode yield before committing.

Key facts (2026-06, see AUSSIE_LLM_PIPELINE.md):
  * Download quota is the bottleneck: 10/day free, 1000/day VIP. Search is cheap
    (~40 req / 10 s). So --probe is unlimited-ish; full harvest is a multi-day VIP drip.
  * Auth: Api-Key header for search; a Bearer user-token (from /login) for downloads.
  * A custom User-Agent is REQUIRED by the API.

Env:
  OS_KEY   = API consumer key   (required)
  OS_USER  = account username   (required only for downloads, not --probe)
  OS_PASS  = account password   (   "    )

  OS_KEY=... ./venv/bin/python subs_harvest.py --probe
  OS_KEY=... OS_USER=... OS_PASS=... ./venv/bin/python subs_harvest.py --max 900
"""
import os, sys, json, time, urllib.request, urllib.parse

API = "https://api.opensubtitles.com/api/v1"
UA = "aussie-llm-corpus/0.1"           # MUST be a real custom UA, not a browser string
KEY = os.environ.get("OS_KEY", "")
USER, PASS = os.environ.get("OS_USER", ""), os.environ.get("OS_PASS", "")
RAW = "/Volumes/scratch/pushshiftdumps/subs_raw"
DONE = "/Volumes/scratch/pushshiftdumps/subs_done.txt"   # file_ids already downloaded

# IMDb ids are parent_imdb_id (series). --probe prints parent_title so we can VERIFY each.
# Mixed deliberately for register diversity (soap + rural drama + comedy/vernacular);
# cap per-show at clean time to avoid one show's catchphrases/proper-nouns dominating.
SHOWS = {
    "home_and_away":     "0094502",
    "neighbours":        "0086814",
    "blue_heelers":      "0108718",
    "a_country_practice":"0086721",
    "packed_to_rafters": "1198108",
    "kath_and_kim":      "0312081",
}


def req(method, path, body=None, token=None, tries=5):
    url = path if path.startswith("http") else API + path
    data = json.dumps(body).encode() if body is not None else None
    h = {"Api-Key": KEY, "User-Agent": UA, "Accept": "application/json"}
    if body is not None:
        h["Content-Type"] = "application/json"
    if token:
        h["Authorization"] = "Bearer " + token
    for a in range(tries):
        try:
            r = urllib.request.Request(url, data=data, headers=h, method=method)
            with urllib.request.urlopen(r, timeout=45) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):            # rate-limited / busy -> back off
                time.sleep(5 * (a + 1)); continue
            print(f"  HTTP {e.code} {e.reason} on {path}", flush=True); return None
        except Exception as ex:
            time.sleep(3 * (a + 1))
    return None


def search_episodes(series_imdb):
    """Enumerate all English EPISODE subtitles for a series. Search-only (cheap)."""
    out, page, total = [], 1, None
    while True:
        q = urllib.parse.urlencode({"parent_imdb_id": series_imdb, "languages": "en",
                                    "type": "episode", "page": page})
        r = req("GET", "/subtitles?" + q)
        if not r:
            break
        total = r.get("total_count", total)
        for item in r.get("data", []):
            at = item.get("attributes", {})
            files = at.get("files", [])
            fd = at.get("feature_details", {})
            if files:
                out.append({"file_id": files[0].get("file_id"),
                            "title": fd.get("parent_title", ""),
                            "season": fd.get("season_number"), "episode": fd.get("episode_number"),
                            "downloads": at.get("download_count", 0)})
        if page >= r.get("total_pages", 1):
            break
        page += 1
        time.sleep(0.3)
    return out, total


def probe():
    print("PROBE — English episode subtitles available per show (search-only, no downloads):\n", flush=True)
    grand = 0
    for name, imdb in SHOWS.items():
        eps, total = search_episodes(imdb)
        uniq = len({(e["season"], e["episode"]) for e in eps if e["season"]})
        title = eps[0]["title"] if eps else "?"
        print(f"  {name:20} imdb tt{imdb}  [{title}]  files={total}  unique-episodes≈{uniq}", flush=True)
        grand += (total or 0)
        time.sleep(0.5)
    print(f"\n  TOTAL subtitle files across shows ≈ {grand}", flush=True)
    print("  Rough tokens if ~all downloaded: files × ~4k words × ~1.3 ≈ "
          f"{grand*4000*1.3/1e6:.0f}M tokens (upper bound; coverage + cleaning trims it).", flush=True)
    print("\n  Download budget: free=10/day, VIP=1000/day. "
          f"At VIP that's ~{(grand/1000):.0f} days of maxed quota for everything.", flush=True)


def login():
    r = req("POST", "/login", {"username": USER, "password": PASS})
    if not r or "token" not in r:
        print("LOGIN FAILED — downloads need OS_USER/OS_PASS", flush=True); sys.exit(1)
    print(f"logged in (level: {r.get('user',{}).get('level','?')}, "
          f"remaining today: {r.get('user',{}).get('allowed_downloads','?')})", flush=True)
    return r["token"]


def harvest(max_dl):
    token = login()
    done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
    n = 0
    for name, imdb in SHOWS.items():
        eps, _ = search_episodes(imdb)
        os.makedirs(f"{RAW}/{name}", exist_ok=True)
        for e in eps:
            fid = str(e["file_id"])
            if fid in done:
                continue
            if n >= max_dl:
                print(f"hit daily cap {max_dl}; resume tomorrow (resumable via {DONE})", flush=True); return
            link = req("POST", "/download", {"file_id": int(fid)}, token=token)
            if not link or "link" not in link:
                continue
            srt = req("GET", link["link"]) if False else None  # fetch raw text below
            try:
                with urllib.request.urlopen(urllib.request.Request(link["link"], headers={"User-Agent": UA}), timeout=45) as resp:
                    text = resp.read().decode("utf-8", "replace")
            except Exception:
                continue
            s, ep = e["season"], e["episode"]
            open(f"{RAW}/{name}/s{s}e{ep}_{fid}.srt", "w").write(text)
            with open(DONE, "a") as f:
                f.write(fid + "\n")
            n += 1
            if n % 25 == 0:
                print(f"  downloaded {n} (remaining today per API: {link.get('remaining','?')})", flush=True)
            time.sleep(0.5)
    print(f"DONE this run: {n} subtitles -> {RAW}", flush=True)


if __name__ == "__main__":
    if not KEY:
        print("Set OS_KEY (free API key from opensubtitles.com profile > API Consumers)."); sys.exit(1)
    if "--probe" in sys.argv:
        probe()
    else:
        mx = 900
        if "--max" in sys.argv:
            mx = int(sys.argv[sys.argv.index("--max") + 1])
        harvest(mx)
