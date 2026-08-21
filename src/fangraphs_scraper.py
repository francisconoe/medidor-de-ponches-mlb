"""
FanGraphs headless-browser scraper.

FanGraphs sits behind Cloudflare, so plain HTTP requests (and even vanilla
headless Selenium) get a 403 "Just a moment..." interstitial. The reliable way
through - the approach used here - is undetected-chromedriver driving a *headful*
Chrome session: it clears the Cloudflare challenge, then we issue an in-page
fetch() to the FanGraphs JSON data API (same origin, so the Cloudflare cookies
ride along) and pull the full pitching leaderboard.

What we extract per pitcher (season-to-date, authoritative):
    - zone_rate  <- Zone%   (a 0-1 fraction; this feature is otherwise
                             impossible to source from any aggregate site)
    - velo_mean  <- usage-weighted mean of the per-pitch-type velocities
                    (pfxv* weighted by pfx*%), falling back to FBv.

We deliberately DO NOT take the pitch-mix (pct_*) from FanGraphs: its pfx*%
columns double-count sweeper/slurve under overlapping legacy PITCHf/x buckets
and sum to >1. Baseball Savant's arsenal endpoint (native Statcast codes) is the
clean source for mix and is used instead (see predict_realtime.py).

The result is cached to data/raw/fangraphs_pit_<season>.json so Cloudflare is
hit at most once per session / cache window.

This module has no side effects on import; it is used by predict_realtime.py via
the --use-fangraphs flag and can also be run standalone to refresh the cache:
    python -m src.fangraphs_scraper --season 2026 --refresh
"""

import argparse
import glob
import json
import os
import re
import time
from datetime import datetime

import numpy as np

CACHE_TEMPLATE = "data/raw/fangraphs_pit_{season}.json"

# FanGraphs PITCHf/x velocity columns we average for velo_mean, paired with
# their usage-share column. (Mix is NOT used as a feature - see module docstring.)
_PFX_VELO_USAGE = [
    ("pfxvFA", "pfxFA%"), ("pfxvFT", "pfxFT%"), ("pfxvFC", "pfxFC%"),
    ("pfxvSI", "pfxSI%"), ("pfxvSL", "pfxSL%"), ("pfxvCU", "pfxCU%"),
    ("pfxvKC", "pfxKC%"), ("pfxvCH", "pfxCH%"), ("pfxvFS", "pfxFS%"),
    ("pfxvFO", "pfxFO%"), ("pfxvEP", "pfxEP%"), ("pfxvSC", "pfxSC%"),
    ("pfxvKN", "pfxKN%"), ("pfxvST", "pfxST%"),
]


# ---------------------------------------------------------------------------
# Chrome version detection (so the driver matches the installed browser)
# ---------------------------------------------------------------------------
def detect_chrome_major() -> int | None:
    """Find the installed Chrome major version from its Application directory."""
    roots = [
        r"C:\Program Files\Google\Chrome\Application",
        r"C:\Program Files (x86)\Google\Chrome\Application",
    ]
    majors = []
    for root in roots:
        for path in glob.glob(os.path.join(root, "*")):
            m = re.match(r"(\d+)\.\d+\.\d+\.\d+", os.path.basename(path))
            if m:
                majors.append(int(m.group(1)))
    return max(majors) if majors else None


# ---------------------------------------------------------------------------
# Scrape
# ---------------------------------------------------------------------------
def _cache_is_fresh(path: str, cache_hours: float) -> bool:
    if not os.path.exists(path):
        return False
    age_h = (time.time() - os.path.getmtime(path)) / 3600.0
    return age_h <= cache_hours


def scrape_fangraphs_pitching(
    season: int,
    cache_hours: float = 12.0,
    refresh: bool = False,
    qual: int = 0,
    page_load_wait: int = 50,
) -> list[dict]:
    """
    Return the FanGraphs season pitching leaderboard as a list of row dicts.

    Uses a disk cache unless `refresh` is set or the cache is older than
    `cache_hours`. On a cache miss, launches undetected-chromedriver (headful),
    clears Cloudflare, and fetches the JSON data API from within the page.
    """
    cache_path = CACHE_TEMPLATE.format(season=season)

    if not refresh and _cache_is_fresh(cache_path, cache_hours):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  FanGraphs: using cached {cache_path} ({len(data)} pitchers).")
        return data

    try:
        import undetected_chromedriver as uc
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "undetected-chromedriver is required for --use-fangraphs. "
            "Install it with: pip install undetected-chromedriver selenium"
        ) from exc

    major = detect_chrome_major()
    print(f"  FanGraphs: launching headful Chrome (v{major}) to clear Cloudflare...")

    opts = uc.ChromeOptions()
    opts.add_argument("--window-size=1400,950")
    driver = uc.Chrome(options=opts, headless=False, use_subprocess=True, version_main=major)

    try:
        driver.set_page_load_timeout(60)
        leaders = (
            "https://www.fangraphs.com/leaders/major-league?pos=all&stats=pit&lg=all"
            f"&qual={qual}&season={season}&season1={season}&type=8"
        )
        driver.get(leaders)

        # Poll until the Cloudflare challenge clears.
        cleared = False
        waited = 0
        while waited < page_load_wait:
            time.sleep(2)
            waited += 2
            title = driver.title or ""
            if "Just a moment" not in title and title.strip():
                cleared = True
                break
        if not cleared:
            raise RuntimeError("Cloudflare challenge did not clear within timeout.")

        api = (
            "https://www.fangraphs.com/api/leaders/major-league/data?pos=all&stats=pit"
            f"&lg=all&qual={qual}&season={season}&season1={season}"
            "&ind=0&pageitems=5000&month=0&type=8"
        )
        js = (
            "const cb=arguments[arguments.length-1];"
            "fetch(arguments[0],{headers:{'Accept':'application/json'}})"
            ".then(r=>r.text().then(t=>cb(t))).catch(e=>cb('ERR:'+e));"
        )
        driver.set_script_timeout(120)
        body = driver.execute_async_script(js, api)
        if body.startswith("ERR:"):
            raise RuntimeError(f"In-page fetch failed: {body}")

        payload = json.loads(body)
        data = payload.get("data", payload if isinstance(payload, list) else [])
        if not data:
            raise RuntimeError("FanGraphs returned no rows.")

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"  FanGraphs: scraped and cached {len(data)} pitchers -> {cache_path}")
        return data
    finally:
        try:
            driver.quit()
        except Exception:  # noqa: BLE001 - UC can raise a harmless handle error on quit
            pass


# ---------------------------------------------------------------------------
# Map FanGraphs rows -> per-pitcher feature overrides
# ---------------------------------------------------------------------------
def _weighted_velo(row: dict) -> float:
    num = den = 0.0
    for vcol, ucol in _PFX_VELO_USAGE:
        v, u = row.get(vcol), row.get(ucol)
        if v is not None and u is not None and not _isnan(v) and not _isnan(u):
            num += float(u) * float(v)
            den += float(u)
    if den > 0:
        return num / den
    fbv = row.get("FBv")
    return float(fbv) if fbv is not None and not _isnan(fbv) else np.nan


def _isnan(x) -> bool:
    try:
        return np.isnan(float(x))
    except (TypeError, ValueError):
        return True


def build_fangraphs_overrides(data: list[dict]) -> dict:
    """
    Build {mlbam_id: {"zone_rate": float, "velo_mean": float, "season_pitches": float}}
    from raw FanGraphs leaderboard rows. Rows without an MLBAM id are skipped.
    """
    out = {}
    for row in data:
        pid = row.get("xMLBAMID")
        if pid is None:
            continue
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            continue

        zone = row.get("Zone%")
        out[pid] = {
            "zone_rate": float(zone) if zone is not None and not _isnan(zone) else np.nan,
            "velo_mean": _weighted_velo(row),
            "season_pitches": float(row.get("Pitches", np.nan)) if row.get("Pitches") else np.nan,
        }
    return out


def apply_fangraphs_override(vec, feature_cols: list[str], fg: dict) -> list[str]:
    """
    Overwrite zone_rate (and velo_mean) on a pitcher's feature row with the
    FanGraphs season-to-date values. Mutates `vec`; returns overridden names.
    """
    overridden = []
    if "zone_rate" in feature_cols and not _isnan(fg.get("zone_rate", np.nan)):
        vec["zone_rate"] = float(fg["zone_rate"])
        overridden.append("zone_rate")
    if "velo_mean" in feature_cols and not _isnan(fg.get("velo_mean", np.nan)):
        vec["velo_mean"] = float(fg["velo_mean"])
        overridden.append("velo_mean")
    return overridden


# ---------------------------------------------------------------------------
# Standalone: refresh the cache
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Scrape/refresh FanGraphs pitching leaderboard.")
    p.add_argument("--season", type=int, default=datetime.today().year)
    p.add_argument("--refresh", action="store_true", help="Ignore cache and re-scrape.")
    p.add_argument("--qual", type=int, default=0)
    args = p.parse_args()

    data = scrape_fangraphs_pitching(args.season, refresh=args.refresh, qual=args.qual)
    overrides = build_fangraphs_overrides(data)
    print(f"Built overrides for {len(overrides)} pitchers.")
    sample = list(overrides.items())[:3]
    for pid, vals in sample:
        print(f"  {pid}: zone_rate={vals['zone_rate']:.3f} velo_mean={vals['velo_mean']:.2f}")


if __name__ == "__main__":
    main()
