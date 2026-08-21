"""
One-command daily betting workflow:

  1. SETTLE - find every v4 prediction CSV for past dates whose plays aren't in
     the ledger yet, grade them against real boxscores, update the ledger.
  2. PREDICT - run the v4 model for tomorrow's slate (real odds if ODDS_API_KEY
     is set, assumed lines otherwise) with Kelly stakes.
  3. REPORT - lifetime record / ROI / drawdown.

    python -m src.daily                 # settle everything due, predict tomorrow
    python -m src.daily --no-predict    # just settle + report
"""

import argparse
import glob
import os
import re
import subprocess
import sys
from datetime import date, timedelta

import pandas as pd

from src.bet_ledger import LEDGER, settle_file, print_summary
from src.odds_api import _load_env_key

PRED_GLOB = "reports/predictions/realtime_v4_predictions_*.csv"


def dates_needing_settlement() -> list[str]:
    today = date.today().strftime("%Y-%m-%d")
    prior = pd.read_csv(LEDGER) if os.path.exists(LEDGER) else None
    done_dates = set(prior["game_date"].astype(str)) if prior is not None else set()
    todo = []
    for path in sorted(glob.glob(PRED_GLOB)):
        m = re.search(r"(\d{4}-\d{2}-\d{2})\.csv$", path)
        if not m:
            continue
        d = m.group(1)
        if d < today and d not in done_dates:
            todo.append(path)
    return todo


def main():
    ap = argparse.ArgumentParser(description="Daily settle + predict + report.")
    ap.add_argument("--date", default=None, help="Prediction target (default tomorrow).")
    ap.add_argument("--no-predict", action="store_true")
    args = ap.parse_args()

    print("### 1. Settling outstanding plays ###")
    todo = dates_needing_settlement()
    if not todo:
        print("Nothing to settle.")
    for path in todo:
        print(f"\n-- {path}")
        new = settle_file(path)
        print_summary(new if not new.empty else None)

    if not args.no_predict:
        target = args.date or (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"\n### 2. Predicting slate for {target} ###")
        if not _load_env_key():
            print("[note] no ODDS_API_KEY (env var or .env file) - lines will be "
                  "assumed at -110. Set the key for real edges before betting.")
        subprocess.run([sys.executable, "-m", "src.predict_realtime_v4",
                        "--date", target], check=False)

    print("\n### 3. Lifetime report ###")
    print_summary()


if __name__ == "__main__":
    main()
