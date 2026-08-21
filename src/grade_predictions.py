"""
Grade a saved real-time prediction CSV against actual boxscore strikeouts.

Pulls the actual strikeouts each starter recorded from the MLB Stats API
boxscores, compares to the predictions, and reports over/under accuracy at each
line plus point error. Also reports the SELECTIVE accuracy (accuracy on just the
high-confidence plays), which is the number that matters for a betting model.

Works with any realtime prediction CSV that has columns: pitcher_name,
game_date, pred_strikeouts, and prob_over_<line>. Does not touch the model.

    python -m src.grade_predictions --file reports/realtime_v3_predictions_2026-07-07.csv
    python -m src.grade_predictions --date 2026-07-05 --version v3
"""

import argparse
import unicodedata

import numpy as np
import pandas as pd
import requests

THRESHOLDS = [3.5, 4.5, 5.5, 6.5]
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def norm(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def scoreboard_starters(d):
    url = "https://bdfed.stitch.mlbinfra.com/bdfed/transform-mlb-scoreboard"
    params = [("stitch_env", "prod"), ("sortTemplate", "4"), ("sportId", "1"),
              ("startDate", d), ("endDate", d), ("language", "en")] + \
             [("gameType", g) for g in ("R", "S", "E", "F", "D", "L", "W")]
    j = requests.get(url, params=params, headers=H, timeout=25).json()
    games = j.get("dates", [{}])[0].get("games", []) if "dates" in j else j.get("games", [])
    m = {}
    for g in games:
        for side in ("home", "away"):
            pp = g.get("teams", {}).get(side, {}).get("probablePitcher")
            if pp and pp.get("id"):
                m[norm(pp["fullName"])] = (g["gamePk"], int(pp["id"]),
                                           g.get("status", {}).get("abstractGameState"))
    return m


_box = {}
def actual_k(pk, pid):
    if pk not in _box:
        _box[pk] = requests.get(f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore",
                                headers=H, timeout=25).json()
    for side in ("home", "away"):
        pl = _box[pk]["teams"][side]["players"].get(f"ID{pid}")
        if pl and pl.get("stats", {}).get("pitching"):
            return pl["stats"]["pitching"].get("strikeOuts")
    return None


def main():
    ap = argparse.ArgumentParser(description="Grade a prediction CSV vs actual strikeouts.")
    ap.add_argument("--file", default=None, help="Prediction CSV path.")
    ap.add_argument("--date", default=None, help="Date YYYY-MM-DD (with --version).")
    ap.add_argument("--version", default="v3", help="Prefix for --date lookup (v2/v3).")
    ap.add_argument("--line", type=float, default=5.5)
    args = ap.parse_args()

    path = args.file or f"reports/predictions/realtime_{args.version}_predictions_{args.date}.csv"
    df = pd.read_csv(path)
    d = str(df["game_date"].iloc[0])[:10]
    sb = scoreboard_starters(d)

    recs = []
    for _, r in df.iterrows():
        info = sb.get(norm(r["pitcher_name"]))
        if not info:
            continue
        pk, pid, state = info
        if state != "Final":
            # Preview = not started; Live = in progress. Grading a live game's
            # partial strikeout count as if it were final produces false MISSes
            # (e.g. a starter shown "actual 0" one batter into the 1st inning).
            continue
        k = actual_k(pk, pid)
        if k is None:
            continue
        recs.append({"name": r["pitcher_name"], "pred": r["pred_strikeouts"],
                     "actual": k, **{f"p{t}": r.get(f"prob_over_{t}") for t in THRESHOLDS}})
    if not recs:
        print(f"No finished games to grade for {d} yet.")
        return

    g = pd.DataFrame(recs)
    y = g["actual"].values.astype(float)
    print(f"\nGraded {len(g)} finished starts for {d}")
    print(f"  Point MAE = {np.abs(g['pred'] - y).mean():.2f}   RMSE = {np.sqrt(((g['pred']-y)**2).mean()):.2f}")
    print(f"\n  {'line':>5} {'all-games acc':>14} {'selective (conf>=.8)':>22}")
    for t in THRESHOLDS:
        p = g[f"p{t}"].values
        if np.all(pd.isna(p)):
            continue
        pred_over = (p >= 0.5).astype(int)
        actual_over = (y > t).astype(int)
        acc_all = (pred_over == actual_over).mean()
        conf = np.abs(p - 0.5) >= 0.30            # calibrated conf >= 0.80
        acc_sel = (pred_over[conf] == actual_over[conf]).mean() if conf.sum() else float("nan")
        print(f"  {t:>5} {acc_all*100:>12.0f}%  {('%.0f%% (n=%d)' % (acc_sel*100, conf.sum())) if conf.sum() else 'no plays':>22}")

    # Per-pitcher detail at the headline line.
    t = args.line
    g["call"] = np.where(g[f"p{t}"] >= 0.5, "OVER", "UNDER")
    g["hit"] = (g["call"] == np.where(y > t, "OVER", "UNDER"))
    g = g.sort_values("pred", ascending=False)
    print(f"\n  Detail @ {t}:")
    for _, r in g.iterrows():
        print(f"    {r['name']:22s} pred {r['pred']:4.1f}  actual {int(r['actual']):2d}  "
              f"{r['call']:5s} {'HIT' if r['hit'] else 'MISS'}")


if __name__ == "__main__":
    main()
