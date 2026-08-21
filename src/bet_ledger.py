"""
Bet ledger - settle flagged plays against actual boxscores and track ROI.

The single source of truth for whether the model is actually making money.
Reads a realtime prediction CSV, settles every row PLAY'd (win / loss / push)
using the real strikeout totals from the MLB boxscore API, appends the results
to reports/bets_ledger.csv (idempotent - a date+pitcher already settled is
skipped), and prints the night's P/L plus the lifetime record, ROI, and drawdown.

    python -m src.bet_ledger --date 2026-07-11              # settle v4 plays
    python -m src.bet_ledger --file reports/predictions/realtime_v4_predictions_2026-07-11.csv
    python -m src.bet_ledger --summary                       # lifetime report only
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.grade_predictions import scoreboard_starters, actual_k, norm
from src.odds_api import american_to_decimal

LEDGER = "reports/bets_ledger.csv"
COLS = ["game_date", "pitcher_name", "call", "line", "odds", "book",
        "model_prob", "edge", "stake", "actual_k", "result", "profit"]


def settle_file(path: str) -> pd.DataFrame:
    """Settle the PLAY rows of one prediction CSV. Returns the new ledger rows."""
    df = pd.read_csv(path)
    # A play is a row whose PLAY column echoes the call (robust to the various
    # pass labels: "— pass —", "— thin data —", "— off-range line —").
    plays = df[df["PLAY"] == df["call"]].copy()
    if plays.empty:
        print(f"No plays flagged in {path}.")
        return pd.DataFrame(columns=COLS)
    d = str(plays["game_date"].iloc[0])[:10]
    sb = scoreboard_starters(d)

    prior = pd.read_csv(LEDGER) if os.path.exists(LEDGER) else pd.DataFrame(columns=COLS)
    done = set(zip(prior["game_date"].astype(str), prior["pitcher_name"]))

    rows = []
    for _, r in plays.iterrows():
        if (d, r["pitcher_name"]) in done:
            continue
        info = sb.get(norm(r["pitcher_name"]))
        if not info:
            print(f"  [skip] {r['pitcher_name']}: not on {d} scoreboard")
            continue
        pk, pid, state = info
        if state != "Final":
            # Settling off a Live game's in-progress K count would lock in a
            # wrong result before the start is actually over.
            print(f"  [skip] {r['pitcher_name']}: game not finished ({state})")
            continue
        k = actual_k(pk, pid)
        if k is None:
            print(f"  [skip] {r['pitcher_name']}: no boxscore line (didn't pitch?)")
            continue
        side, line = r["call"].split()
        line = float(line)
        stake = float(r.get("stake", 0.0)) or 0.0
        if k == line:
            result, profit = "push", 0.0
        elif (k > line) == (side == "over"):
            result, profit = "win", round(stake * (american_to_decimal(r["odds"]) - 1.0), 2)
        else:
            result, profit = "loss", -stake
        rows.append({"game_date": d, "pitcher_name": r["pitcher_name"], "call": r["call"],
                     "line": line, "odds": int(r["odds"]), "book": r.get("book", ""),
                     "model_prob": r["model_prob"], "edge": r["edge"], "stake": stake,
                     "actual_k": int(k), "result": result, "profit": profit})

    new = pd.DataFrame(rows, columns=COLS)
    if not new.empty:
        ledger = pd.concat([prior, new], ignore_index=True)
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        ledger.to_csv(LEDGER, index=False)
    return new


def print_summary(new: pd.DataFrame | None = None):
    if new is not None and not new.empty:
        d = new["game_date"].iloc[0]
        w = (new["result"] == "win").sum(); l = (new["result"] == "loss").sum()
        print(f"\n=== SETTLED {d}:  {w}-{l}"
              + (f"-{(new['result']=='push').sum()} (pushes)" if (new["result"] == "push").any() else "")
              + f"   staked {new['stake'].sum():.2f}   P/L {new['profit'].sum():+.2f} ===")
        for _, r in new.sort_values("model_prob", ascending=False).iterrows():
            print(f"  {r['pitcher_name']:22s} {r['call']:10s} p={r['model_prob']:.2f} "
                  f"stake {r['stake']:6.2f}  actual {r['actual_k']:2d}  "
                  f"{r['result'].upper():5s} {r['profit']:+8.2f}")

    if not os.path.exists(LEDGER):
        print("\nLedger empty.")
        return
    led = pd.read_csv(LEDGER)
    settled = led[led["result"].isin(["win", "loss", "push"])]
    if settled.empty:
        return
    staked = settled["stake"].sum()
    profit = settled["profit"].sum()
    wins = (settled["result"] == "win").sum(); losses = (settled["result"] == "loss").sum()
    cum = settled["profit"].cumsum()
    drawdown = (cum - cum.cummax()).min()
    print(f"\n=== LIFETIME ({settled['game_date'].nunique()} night(s), {len(settled)} bets) ===")
    print(f"  Record : {wins}-{losses}" + (f"-{(settled['result']=='push').sum()}" if (settled['result'] == 'push').any() else "")
          + f"  ({wins / max(wins + losses, 1) * 100:.0f}%)")
    print(f"  Staked : {staked:.2f}")
    print(f"  Profit : {profit:+.2f}")
    print(f"  ROI    : {profit / staked * 100 if staked else 0:+.1f}% on turnover")
    print(f"  Max drawdown: {drawdown:+.2f}")


def main():
    ap = argparse.ArgumentParser(description="Settle plays and track betting ROI.")
    ap.add_argument("--file", default=None, help="Prediction CSV to settle.")
    ap.add_argument("--date", default=None, help="Date YYYY-MM-DD (with --version).")
    ap.add_argument("--version", default="v4")
    ap.add_argument("--summary", action="store_true", help="Print lifetime summary only.")
    args = ap.parse_args()

    if args.summary:
        print_summary()
        return
    path = args.file or f"reports/predictions/realtime_{args.version}_predictions_{args.date}.csv"
    new = settle_file(path)
    print_summary(new)


if __name__ == "__main__":
    main()
