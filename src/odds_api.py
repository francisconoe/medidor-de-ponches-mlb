"""
Real sportsbook strikeout-prop lines from The Odds API.

The v3 model was picking whichever line it was *most confident* about, which
mechanically collapsed every call to the outermost lines (over 3.5 / under 6.5).
Those are accurate but unbettable: a book prices "over 3.5 K" at -600 or worse,
so being right earns almost nothing while the occasional early exit wipes you
out. To actually evaluate against a book you must use the line the book POSTS
(near the pitcher's median, ~5.5) and the PRICE it posts, then bet only when the
calibrated probability beats the vig-free implied probability (positive EV).

This module fetches the posted `pitcher_strikeouts` market for the target date
and returns, per pitcher, the main line and the American over/under prices. It
changes no model code; predict_realtime_v3 consumes it read-only.

Auth: put ODDS_API_KEY=<your key> in a .env file at the project root (gitignored),
or set the ODDS_API_KEY environment variable (free tier at the-odds-api.com).
Without a key, fetch() returns {} and the caller falls back to a standard
5.5 @ -110 assumption so the pipeline still runs offline.

    ODDS_API_KEY=... python -m src.odds_api --date 2026-07-09
"""

import argparse
import os
import unicodedata
from datetime import datetime, timedelta, timezone

import requests

API_BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb"
MARKET = "pitcher_strikeouts"
TIMEOUT = 25
ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def _load_env_key() -> str | None:
    """ODDS_API_KEY from the environment, else from the project-root .env file
    (simple KEY=VALUE lines; no dependency on python-dotenv)."""
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key.strip()
    try:
        with open(ENV_FILE, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if line.startswith("ODDS_API_KEY"):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and "your_key" not in val.lower():
                        return val
    except OSError:
        pass
    return None


def norm_name(s: str) -> str:
    """Accent/case/punctuation-insensitive key so 'Jesús Luzardo' matches
    'Jesus Luzardo' and 'J.T. Ginn' matches 'JT Ginn'."""
    s = (unicodedata.normalize("NFKD", str(s))
         .encode("ascii", "ignore").decode().lower())
    s = "".join(ch for ch in s if ch.isalnum() or ch == " ")
    return " ".join(s.split())


def american_to_decimal(odds: float) -> float:
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / -odds)


def american_to_implied(odds: float) -> float:
    """Raw implied probability (still contains the book's vig)."""
    return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / (-odds + 100.0)


def devig_two_way(over_odds: float, under_odds: float) -> tuple[float, float]:
    """Normalise the over/under implied probs to remove the overround.

    Returns (fair_prob_over, fair_prob_under) summing to 1.0.
    """
    io, iu = american_to_implied(over_odds), american_to_implied(under_odds)
    total = io + iu
    if total <= 0:
        return float("nan"), float("nan")
    return io / total, iu / total


def _get(url: str, params: dict) -> object:
    r = requests.get(url, params=params, timeout=TIMEOUT,
                     headers={"Accept": "application/json"})
    r.raise_for_status()
    return r.json()


def fetch(target_date: str, api_key: str | None = None,
          regions: str = "us", bookmaker_priority=("draftkings", "fanduel")) -> dict:
    """Return {normalized_pitcher_name: {line, over_odds, under_odds, book}} for target_date.

    On any failure (no key, network, no games) returns {} so the caller can fall
    back to a standard assumption. Only the game(s) on target_date (local US date
    as reported by the API's ISO commence_time) are considered.
    """
    api_key = api_key or _load_env_key()
    if not api_key:
        print("[odds] no API key (env var or .env) - no live odds; caller will "
              "use assumed lines @ -110.")
        return {}

    try:
        events = _get(f"{API_BASE}/events", {"apiKey": api_key})
    except Exception as exc:  # noqa: BLE001
        print(f"[odds] events fetch failed: {exc}")
        return {}

    want = target_date[:10]
    event_ids = []
    for ev in events:
        ct = ev.get("commence_time", "")
        try:
            d = datetime.fromisoformat(ct.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            continue
        # MLB night games (~7-10pm ET) cross into the next UTC calendar date
        # (e.g. 8:06pm ET -> 00:06 UTC the next day), so comparing raw UTC
        # dates silently grabs the wrong night's slate. MLB's season runs
        # entirely in daylight time (ET = UTC-4); shift before taking the
        # date so the comparison uses the US "baseball day" instead.
        et_date = (d - timedelta(hours=4)).strftime("%Y-%m-%d")
        if et_date == want:
            event_ids.append(ev["id"])

    out: dict = {}
    for eid in event_ids:
        try:
            data = _get(f"{API_BASE}/events/{eid}/odds", {
                "apiKey": api_key, "regions": regions, "markets": MARKET,
                "oddsFormat": "american",
            })
        except Exception as exc:  # noqa: BLE001
            print(f"[odds] props fetch failed for {eid}: {exc}")
            continue
        _merge_event(data, out, bookmaker_priority)

    print(f"[odds] pulled {MARKET} lines for {len(out)} pitcher(s) on {want}.")
    return out


def _merge_event(data: dict, out: dict, bookmaker_priority) -> None:
    """Pick one bookmaker's line per pitcher (preferred book first) into out."""
    books = data.get("bookmakers", [])
    order = sorted(books, key=lambda b: (
        bookmaker_priority.index(b.get("key")) if b.get("key") in bookmaker_priority
        else len(bookmaker_priority)))
    for bk in order:
        for mk in bk.get("markets", []):
            if mk.get("key") != MARKET:
                continue
            # Group the Over/Under outcomes by pitcher (description) + line (point).
            per_pitcher: dict = {}
            for oc in mk.get("outcomes", []):
                name = norm_name(oc.get("description", ""))
                side = str(oc.get("name", "")).lower()
                pt, price = oc.get("point"), oc.get("price")
                if not name or pt is None or price is None:
                    continue
                slot = per_pitcher.setdefault(name, {"line": pt})
                if side.startswith("over"):
                    slot["over_odds"] = float(price)
                elif side.startswith("under"):
                    slot["under_odds"] = float(price)
            for name, slot in per_pitcher.items():
                # First book wins; don't overwrite an already-captured pitcher.
                if name in out or "over_odds" not in slot or "under_odds" not in slot:
                    continue
                out[name] = {"line": float(slot["line"]),
                             "over_odds": slot["over_odds"],
                             "under_odds": slot["under_odds"],
                             "book": bk.get("key")}


def main():
    ap = argparse.ArgumentParser(description="Fetch MLB pitcher strikeout props.")
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    args = ap.parse_args()
    odds = fetch(args.date)
    for name, o in sorted(odds.items()):
        fo, fu = devig_two_way(o["over_odds"], o["under_odds"])
        print(f"  {name:24s} line {o['line']:>4}  O {o['over_odds']:+.0f} / "
              f"U {o['under_odds']:+.0f}  (fair over {fo:.0%})  [{o['book']}]")


if __name__ == "__main__":
    main()
