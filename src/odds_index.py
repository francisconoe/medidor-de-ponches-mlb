"""
oddsindex.com strikeout-prop picks - best-effort scraper, NEVER fatal.

The site is a Next.js app that server-renders its picks into the __NEXT_DATA__
JSON blob (props.pageProps.structuredData.strikeoutPicks) - a clean per-pitcher
list of pick side / line / odds / confidence / expected Ks. We parse that JSON
directly (no HTML scraping, no JS rendering). Every failure path returns {} and
prints a note, so a scheduled daily run is never broken by a site change, block,
timeout, or layout tweak.

Used as an independent second opinion: the daily run annotates each of the
model's flagged plays with whether oddsindex agrees, and reports the
agreement-filtered slate.

    python -m src.odds_index
"""

import argparse
import json
import re

import requests

from src.odds_api import norm_name

URL = "https://oddsindex.com/sports/mlb/strikeout-props-today"
TIMEOUT = 25
_NEXT = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.DOTALL)


def fetch_picks(url: str = URL) -> dict:
    """{normalized_pitcher_name: {side, line, odds, confidence, expected_ks}}.
    Returns {} on ANY error - callers must treat an empty dict as 'no data'."""
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=TIMEOUT)
        r.raise_for_status()
        m = _NEXT.search(r.text)
        if not m:
            print("[oddsindex] __NEXT_DATA__ not found (site layout changed?) - skipping.")
            return {}
        picks = (json.loads(m.group(1))
                 .get("props", {}).get("pageProps", {})
                 .get("structuredData", {}).get("strikeoutPicks", []))
    except Exception as exc:  # noqa: BLE001
        print(f"[oddsindex] fetch/parse failed (non-fatal): {exc}")
        return {}

    out = {}
    for p in picks:
        try:
            name = norm_name(p.get("pitcher", ""))
            side = str(p.get("pick", "")).strip().lower()
            side = "over" if side.startswith("o") else ("under" if side.startswith("u") else None)
            line = float(p.get("line"))
            if name and side:
                out[name] = {"side": side, "line": line,
                             "odds": str(p.get("odds", "")),
                             "confidence": str(p.get("confidence", "")).upper(),
                             "expected_ks": p.get("expected_ks")}
        except Exception:  # noqa: BLE001 - one bad row must not drop the rest
            continue
    print(f"[oddsindex] parsed {len(out)} picks.")
    return out


def agreement(call: str, picks: dict, pitcher_name: str):
    """Does oddsindex support the model's bet? Compares oddsindex's projection
    (expected_ks) against the MODEL's line, so a different-line pick isn't a false
    conflict (my 'over 2.5' vs their 'under 4.5' both win at 3-4 K). Falls back to
    side-vs-side when expected_ks is missing.
    Returns ('agree'|'conflict'|'no-pick', oddsindex_call_str)."""
    ix = picks.get(norm_name(pitcher_name))
    if not ix:
        return "no-pick", ""
    side, line = call.split()[0], float(call.split()[1])
    ix_call = (f"{ix['side']} {ix['line']:g} ({ix['confidence']} {ix['odds']}, "
               f"xK={ix['expected_ks']})")
    exp = ix.get("expected_ks")
    try:
        exp = float(exp)
        supports = (exp > line) if side == "over" else (exp < line)
        return ("agree" if supports else "conflict"), ix_call
    except (TypeError, ValueError):
        return ("agree" if side == ix["side"] else "conflict"), ix_call


def main():
    argparse.ArgumentParser(description="Fetch oddsindex strikeout picks.").parse_args()
    for name, p in sorted(fetch_picks().items()):
        print(f"  {name:24s} {p['side']:5s} {p['line']:>4}  {p['confidence']:6s} {p['odds']}")


if __name__ == "__main__":
    main()
