"""
run_daily.py - non-interactive daily entrypoint for the scheduled run.

Runs the whole workflow with NO stdin / prompts / manual steps, so a scheduler
(GitHub Actions cron) can drive it:

  1. SETTLE  - grade any past prediction files not yet in the ledger vs boxscores
  2. PREDICT - today's slate with real DraftKings/FanDuel odds + confirmed
               lineups + the short-outing correction (all on by default)
  3. REPORT  - readiness (odds/lineup coverage), the flagged plays, lifetime ROI

Exit code: 0 on success (today's predictions were produced), non-zero on failure
- the scheduler uses this to know something broke. Idempotent: settling skips
already-graded games; re-running just refreshes today's file with newer odds.

------------------------------------------------------------------------------
WHEN TO FIRE IT (the timing this was built around)
------------------------------------------------------------------------------
Both inputs only exist on GAME DAY, and only firm up a few hours out:
  - DraftKings K-props: posted through the morning, most up by early afternoon ET
  - Confirmed lineups (MLB Stats API): post ~3-4h before each first pitch
  - oddsindex analysis (manual cross-check): published game-day, early afternoon

MLB's bulk is night games (~7pm ET / 23:00 UTC first pitch). Their lineups are
up by ~3-4pm ET and odds well before. So a run around **18:00-20:00 UTC**
(~2-4pm ET, ~3-5pm BST in summer) catches the night slate with BOTH posted,
before first pitch. Day games (~1pm ET) start earlier and may already be under
way - a single daily run cannot be pre-game for every start; this targets the
majority and reports coverage so you can see how ready the slate was.

The script does NOT enforce the time - the cron schedule does (see
.github/workflows/daily.yml). run_daily just always targets *today* (US Eastern).

Env:
  ODDS_API_KEY  real odds (falls back to assumed -110 lines if unset)
  RUN_DATE      override target date YYYY-MM-DD (default: today, US Eastern)
  BANKROLL      stake sizing (default 1000)
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.bet_ledger import settle_file, print_summary
from src.daily import dates_needing_settlement
from src.odds_index import fetch_picks, agreement

PRED_DIR = "reports/predictions"


def log(msg):
    print(f"[run_daily] {msg}", flush=True)


def target_date() -> str:
    if os.environ.get("RUN_DATE"):
        return os.environ["RUN_DATE"][:10]
    # US Eastern "baseball day" - the slate belongs to the US calendar date.
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def settle_outstanding() -> None:
    """Grade past prediction files not yet in the ledger. Non-fatal on error -
    a boxscore hiccup must not block today's prediction."""
    try:
        todo = dates_needing_settlement()
        if not todo:
            log("settle: nothing outstanding")
            return
        for path in todo:
            log(f"settle: {path}")
            new = settle_file(path)
            print_summary(new if not new.empty else None)
    except Exception as exc:  # noqa: BLE001
        log(f"settle FAILED (non-fatal, continuing to predict): {exc}")


def predict(date: str, bankroll: float, retries: int = 3) -> bool:
    """Run the v4 serving pipeline for `date`. Retries transient failures.
    Returns True on success."""
    cmd = [sys.executable, "-m", "src.predict_realtime_v4",
           "--date", date, "--bankroll", str(bankroll)]
    for attempt in range(1, retries + 1):
        log(f"predict attempt {attempt}/{retries}: {' '.join(cmd)}")
        r = subprocess.run(cmd)
        if r.returncode == 0:
            return True
        log(f"predict attempt {attempt} exited {r.returncode}")
        if attempt < retries:
            time.sleep(20 * attempt)  # linear backoff
    return False


def lifetime_line() -> str:
    """One-line lifetime record/ROI from the ledger, for the notification."""
    try:
        led = pd.read_csv("reports/bets_ledger.csv")
        s = led[led["result"].isin(["win", "loss"])]
        if s.empty:
            return "Lifetime: no settled bets yet."
        w, l = int((s["result"] == "win").sum()), int((s["result"] == "loss").sum())
        staked, profit = s["stake"].sum(), s["profit"].sum()
        roi = profit / staked * 100 if staked else 0.0
        return (f"Lifetime: {w}-{l} ({w/(w+l)*100:.0f}%) | "
                f"ROI {roi:+.1f}% | P/L {profit:+.2f}")
    except Exception:  # noqa: BLE001
        return "Lifetime: (ledger unavailable)"


def build_report(date: str) -> str:
    """Log the day's slate + oddsindex agreement, write the agreement CSV, and
    return a compact Discord-ready message. Never raises."""
    path = f"{PRED_DIR}/realtime_v4_predictions_{date}.csv"
    if not os.path.exists(path):
        log(f"readiness: no prediction file at {path}")
        return f"⚠️ {date}: no predictions produced."
    df = pd.read_csv(path)
    n = len(df)
    real_odds = int(df.get("book", pd.Series(dtype=str)).isin(["draftkings", "fanduel"]).sum())
    confirmed = int((df.get("lineup", pd.Series(dtype=str)) == "confirmed").sum())
    plays = df[df["PLAY"] == df["call"]] if "PLAY" in df else df.iloc[0:0]
    log(f"readiness: {n} starters | real odds {real_odds}/{n} | "
        f"confirmed lineups {confirmed}/{n} | plays {len(plays)}")
    if real_odds < 0.5 * n:
        log("readiness WARNING: <50% real odds - may have fired too early.")

    head = (f"🎯 **Strikeout slate — {date}**\n"
            f"Plays: {len(plays)} | staked ${plays['stake'].sum():.0f} | "
            f"real odds {real_odds}/{n} | lineups {confirmed}/{n}\n"
            f"_Full slate of all {n} pitchers attached as CSV._")
    if not len(plays):
        return f"{head}\n_No plays cleared the filter today._\n{lifetime_line()}"

    picks = {}
    try:
        picks = fetch_picks()
    except Exception as exc:  # noqa: BLE001 (fetch_picks already guards; belt-and-braces)
        log(f"oddsindex lookup failed (non-fatal): {exc}")
    plays = plays.copy()
    ann = [agreement(r["call"], picks, r["pitcher_name"]) for _, r in plays.iterrows()]
    plays["oddsindex"], plays["agree"] = [a[1] or "-" for a in ann], [a[0] for a in ann]

    cols = [c for c in ["pitcher_name", "call", "odds", "book", "model_prob",
                        "edge", "stake", "oddsindex", "agree"] if c in plays.columns]
    log("flagged plays (with oddsindex second opinion):")
    print(plays[cols].to_string(index=False), flush=True)
    plays[cols].to_csv(f"{PRED_DIR}/agreement_slate_{date}.csv", index=False)

    lines = [head, "", "**Model plays:**"]
    for _, r in plays.iterrows():
        tick = {"agree": "✅", "conflict": "❌", "no-pick": "·"}.get(r["agree"], "·")
        lines.append(f"{tick} {r['pitcher_name']} {r['call']} "
                     f"({int(r['odds']):+d} {r['book']}) p={r['model_prob']:.2f}")
    if picks:
        agreed = plays[plays["agree"] == "agree"]
        lines += ["", f"✅ **Agreement slate (model + oddsindex):** "
                  f"{len(agreed)}/{len(plays)}, ${agreed['stake'].sum():.0f}"]
        lines += [f"• {r['pitcher_name']} {r['call']}" for _, r in agreed.iterrows()]
    else:
        lines += ["", "_oddsindex unavailable — agreement filter skipped._"]
    lines += ["", lifetime_line()]
    return "\n".join(lines)


def notify_discord(message: str, attach_path: str | None = None) -> None:
    """POST the message to the Discord webhook, attaching the full prediction CSV
    (every pitcher) when present. Non-fatal: the data work is already done, so a
    webhook failure only warns."""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        log("no DISCORD_WEBHOOK_URL set - skipping notification (see README).")
        return
    try:
        import requests
        payload = {"content": message[:1990]}
        if attach_path and os.path.exists(attach_path):
            with open(attach_path, "rb") as f:
                r = requests.post(url, data={"payload_json": json.dumps(payload)},
                                  files={"file": (os.path.basename(attach_path), f, "text/csv")},
                                  timeout=30)
        else:
            r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()
        log(f"Discord notification sent{' with full-slate CSV' if attach_path and os.path.exists(attach_path) else ''}.")
    except Exception as exc:  # noqa: BLE001
        log(f"Discord notification FAILED (non-fatal): {exc}")


def main() -> int:
    date = target_date()
    bankroll = float(os.environ.get("BANKROLL", "1000"))
    log(f"=== daily run for {date} (bankroll {bankroll:g}) ===")
    if not os.environ.get("ODDS_API_KEY"):
        log("note: ODDS_API_KEY not set - odds will be assumed -110 (see README).")

    settle_outstanding()

    ok = predict(date, bankroll)
    if not ok:
        log("PREDICT FAILED after retries - exiting non-zero")
        return 1

    message = build_report(date)
    print_summary()
    csv_path = f"{PRED_DIR}/realtime_v4_predictions_{date}.csv"
    notify_discord(message, attach_path=csv_path)
    log("=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
