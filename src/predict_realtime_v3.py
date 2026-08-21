"""
Real-time strikeout prediction with the v3 calibrated NB model + selective play.

Serves train_bnn_v3.py: rebuilds the v3 features for tomorrow's scheduled
starters, runs the Negative-Binomial MC-Dropout network (ensembled with
XGBoost-Poisson for the point mean), applies the isotonic calibrators, and emits
CALIBRATED P(K > line) for every line at once.

Because the probabilities are now trustworthy, it also outputs a SELECTIVE
recommendation: it only flags a play when the calibrated confidence clears a
threshold (default 0.85 either way) - the diagnostic study showed that betting
just the most-confident slice is the only honest route to 90%+ accuracy.

Feature sourcing at serve time:
  - pitcher form + workload : current-season Statcast (pull_pitcher_starts).
  - opponent team K% (overall + vs hand) : MLB Stats API team hitting.
  - opp_lineup_k : approximated by the opponent's team K% at serve time (posted
    lineups are usually unavailable a day out); documented as a bounded proxy.
  - park_k_factor : learned factor for the game's home venue (by team id).

Self-contained; reuses v2 data-fetch helpers and v3 model helpers read-only.
Changes no existing file.

    python -m src.predict_realtime_v3                       # tomorrow
    python -m src.predict_realtime_v3 --date 2026-07-07 --line 5.5
    python -m src.predict_realtime_v3 --min-confidence 0.80  # looser selective filter
"""

import argparse
import json
import os
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from src.utils import load_config, ensure_directories, set_seed
from src.train_bnn_v3 import NBDropoutNet, mc_nb_predict, nb_prob_over, THRESHOLDS
from src.predict_realtime_v2 import pull_pitcher_starts, fetch_team_k_rates
from src.features_v3 import V3_FEATURES
from src.train_bnn_v2 import ROLL_WINDOWS
from src.odds_api import (fetch as fetch_odds, norm_name, devig_two_way,
                          american_to_decimal)

# MLB team id -> Statcast home-team code (for park factor lookup).
TEAMID_TO_CODE = {
    108: "LAA", 109: "AZ", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN",
    114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KC", 119: "LAD",
    120: "WSH", 121: "NYM", 133: "ATH", 134: "PIT", 135: "SD", 136: "SEA",
    137: "SF", 138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}


# ---------------------------------------------------------------------------
# Scoreboard (starters + team ids + home flag)
# ---------------------------------------------------------------------------
def fetch_scoreboard(target_date: str) -> list[dict]:
    import requests
    url = "https://bdfed.stitch.mlbinfra.com/bdfed/transform-mlb-scoreboard"
    params = [
        ("stitch_env", "prod"), ("sortTemplate", "4"), ("sportId", "1"),
        ("startDate", target_date), ("endDate", target_date), ("language", "en"),
    ] + [("gameType", g) for g in ("E", "S", "R", "F", "D", "L", "W", "A", "P")]
    try:
        r = requests.get(url, params=params, timeout=25,
                         headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] scoreboard fetch failed: {exc}")
        return []
    dates = data.get("dates", [])
    games = dates[0].get("games", []) if dates else data.get("games", [])
    out = []
    for g in games:
        t = g.get("teams", {})
        home, away = t.get("home", {}), t.get("away", {})
        home_t, away_t = home.get("team", {}), away.get("team", {})
        for side, opp, is_home in [(home, away_t, 1), (away, home_t, 0)]:
            pp = side.get("probablePitcher")
            if not pp or pp.get("id") is None:
                continue
            home_id = home_t.get("id")
            out.append({
                "pitcher_id": int(pp["id"]),
                "pitcher_name": pp.get("fullName", str(pp["id"])),
                "team_name": side.get("team", {}).get("name", ""),
                "opp_team_id": opp.get("id"),
                "opp_team_name": opp.get("name", ""),
                "is_home": is_home,
                "park_code": TEAMID_TO_CODE.get(int(home_id)) if home_id else None,
            })
    return out


# ---------------------------------------------------------------------------
# v3 feature row (mirrors features_v3 definitions)
# ---------------------------------------------------------------------------
def build_v3_row(starts: pd.DataFrame, target_date: str, is_home: int,
                 opp: dict, park_factor: float, feature_cols: list[str]):
    src = {"k": "k", "k_rate": "k_rate", "whiff_rate": "whiff_rate",
           "csw_rate": "csw_rate", "velo": "velo", "bf": "bf", "pitches": "pitches"}
    row = {}
    for col, name in src.items():
        vals = starts[col].to_numpy(dtype=float)
        for w in ROLL_WINDOWS:
            row[f"roll{w}_{name}"] = float(np.nanmean(vals[-w:])) if len(vals) else np.nan
        row[f"std_{name}"] = float(np.nanmean(vals)) if len(vals) else np.nan

    last = pd.to_datetime(starts["game_date"].max())
    row["days_rest"] = float(min((pd.to_datetime(target_date) - last).days, 14))
    row["is_home"] = float(is_home)
    throws = str(starts["p_throws"].mode().iloc[0]) if not starts.empty else "R"
    row["throws_R"] = 1.0 if throws == "R" else 0.0
    row["opp_k_rate"] = opp.get("overall", np.nan)
    row["opp_k_rate_hand"] = opp.get(throws, np.nan)
    # Serve-time proxy: team K% stands in for the batter-level lineup K%.
    row["opp_lineup_k"] = opp.get("overall", np.nan)
    row["park_k_factor"] = float(park_factor) if park_factor is not None else 1.0

    meta = {"n_starts": len(starts), "last_start": last.strftime("%Y-%m-%d"),
            "throws": throws, "recent_k": row.get("roll5_k")}
    return pd.Series({c: row.get(c, np.nan) for c in feature_cols}), meta


# ---------------------------------------------------------------------------
# Load v3 bundle
# ---------------------------------------------------------------------------
def load_v3(cfg):
    with open("models/bnn_v3_features.json", encoding="utf-8") as f:
        feats = json.load(f)
    with open("models/bnn_v3_impute.json", encoding="utf-8") as f:
        medians = json.load(f)
    with open("models/bnn_v3_park.json", encoding="utf-8") as f:
        park = json.load(f)
    with open("models/bnn_v3_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    scaler = joblib.load("models/bnn_v3_scaler.pkl")
    calibrators = joblib.load("models/bnn_v3_calibrators.pkl")
    xgbm = xgb.XGBRegressor()
    xgbm.load_model("models/bnn_v3_xgb.json")
    model = NBDropoutNet(len(feats), tuple(meta["hidden_sizes"]), meta["dropout"])
    model.load_state_dict(torch.load("models/bnn_v3_model.pt", map_location="cpu"))
    return model, scaler, feats, medians, park, meta, calibrators, xgbm


def parse_args():
    p = argparse.ArgumentParser(description="Real-time strikeout predictions (v3 calibrated NB).")
    p.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default tomorrow).")
    p.add_argument("--line", type=float, default=None,
                   help="Force a specific over/under line for every pitcher. Default: use "
                        "the real book line pulled from The Odds API (5.5 @ -110 fallback).")
    p.add_argument("--min-edge", type=float, default=0.03,
                   help="Selective filter: flag a PLAY only when the calibrated probability "
                        "beats the book's vig-free implied probability by at least this much "
                        "AND the bet is +EV. Default 0.03 (3 percentage points).")
    p.add_argument("--no-odds", action="store_true",
                   help="Skip The Odds API; assume 5.5 @ -110 for every pitcher.")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    ensure_directories()
    set_seed(cfg["project"]["seed"])

    target_date = args.date or (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    season = datetime.strptime(target_date, "%Y-%m-%d").year

    print(f"Target date: {target_date}")
    print("Loading v3 calibrated NB model...")
    model, scaler, feats, medians, park, meta, calibrators, xgbm = load_v3(cfg)
    mc_samples = meta["mc_samples"]
    w = meta["ensemble_nn_weight"]

    print("Fetching starters (mlb.com/scores)...")
    starters = fetch_scoreboard(target_date)
    if not starters:
        print("No posted starters.")
        return
    print(f"Found {len(starters)} starter(s).")
    print("Fetching opponent team K% (MLB Stats API)...")
    team_k = fetch_team_k_rates(season)

    rows, info = [], []
    for s in starters:
        print(f"  - {s['pitcher_name']} vs {s['opp_team_name']} ...", end=" ")
        starts = pull_pitcher_starts(s["pitcher_id"], target_date)
        if starts.empty:
            print("no data, skip.")
            continue
        opp = team_k.get(int(s["opp_team_id"]), {}) if s.get("opp_team_id") else {}
        pf = park.get(s.get("park_code"), 1.0)
        vec, m = build_v3_row(starts, target_date, s["is_home"], opp, pf, feats)
        rows.append(vec)
        info.append({"pitcher_name": s["pitcher_name"], "team": s["team_name"],
                     "opponent": s["opp_team_name"], "n_starts": m["n_starts"],
                     "last_start": m["last_start"]})
        print(f"{m['n_starts']} starts.")
    if not rows:
        print("No usable pitchers.")
        return

    feat_df = pd.DataFrame(rows)[feats].fillna(pd.Series(medians))
    X = torch.tensor(scaler.transform(feat_df.values.astype(np.float32)))
    mu_s, r_s = mc_nb_predict(model, X, mc_samples)
    nn_mean = mu_s.mean(0)
    xgb_pred = xgbm.predict(feat_df)
    pred_mean = w * nn_mean + (1 - w) * xgb_pred

    out = pd.DataFrame(info)
    out["game_date"] = target_date
    out["pred_strikeouts"] = np.round(pred_mean, 2)
    for line in THRESHOLDS:
        raw = nb_prob_over(mu_s, r_s, line)
        out[f"prob_over_{line}"] = np.round(calibrators[line].predict(raw), 3)

    # ------------------------------------------------------------------
    # Betting-realistic call. Instead of picking the line we are MOST
    # confident about (which always degenerates to over 3.5 / under 6.5 -
    # accurate but priced so short that the book always wins), we use the
    # line the BOOK actually posts (~the pitcher's median) and only flag a
    # play when the calibrated probability beats the vig-free implied
    # probability, i.e. the bet is +EV. Model + calibration are untouched;
    # this only changes how the call is chosen and scored.
    # ------------------------------------------------------------------
    print("Fetching sportsbook strikeout lines (The Odds API)...")
    odds = {} if args.no_odds else fetch_odds(target_date)

    def cal_for(line):  # calibrators exist only at THRESHOLDS; use the nearest.
        return calibrators[min(THRESHOLDS, key=lambda t: abs(t - line))]

    # The realistic set of lines a book posts for strikeout props. When live
    # odds are missing we assume the book centres its line on the projection
    # (nearest of these), NOT a flat 5.5 - so weak arms get a 3.5/4.5 line and
    # aces a 6.5, which is what a book would actually hang.
    BOOK_LINES = list(THRESHOLDS)  # [3.5, 4.5, 5.5, 6.5]

    recs = []
    for i, name in enumerate(out["pitcher_name"].tolist()):
        o = odds.get(norm_name(name))
        proj = float(out["pred_strikeouts"].iloc[i])
        assumed_line = min(BOOK_LINES, key=lambda t: abs(t - proj))
        if args.line is not None:
            book_line = args.line
            over_odds, under_odds = (o["over_odds"], o["under_odds"]) if (
                o and abs(o["line"] - args.line) < 1e-9) else (-110.0, -110.0)
            book = o["book"] if o and abs(o["line"] - args.line) < 1e-9 else "(assumed)"
        elif o:
            book_line, over_odds, under_odds, book = (
                o["line"], o["over_odds"], o["under_odds"], o["book"])
        else:  # no market -> assume book line nearest the projection @ -110/-110
            book_line, over_odds, under_odds, book = assumed_line, -110.0, -110.0, "(assumed)"

        raw = float(nb_prob_over(mu_s[:, i:i + 1], r_s[:, i:i + 1], book_line)[0])
        p_over = float(np.clip(cal_for(book_line).predict([raw])[0], 0.0, 1.0))
        p_under = 1.0 - p_over
        fair_over, fair_under = devig_two_way(over_odds, under_odds)
        dec_over, dec_under = american_to_decimal(over_odds), american_to_decimal(under_odds)
        ev_over = p_over * dec_over - 1.0            # EV per 1u staked
        ev_under = p_under * dec_under - 1.0

        if ev_over >= ev_under:
            side, p, fair, ev, price = "over", p_over, fair_over, ev_over, over_odds
        else:
            side, p, fair, ev, price = "under", p_under, fair_under, ev_under, under_odds
        edge = p - fair
        call = f"{side} {book_line:g}"
        play = call if (ev > 0.0 and edge >= args.min_edge) else "— pass —"
        recs.append({"line": book_line, "odds": int(round(price)), "book": book,
                     "call": call, "model_prob": round(p, 3), "fair_prob": round(fair, 3),
                     "edge": round(edge, 3), "EV": round(ev, 3), "PLAY": play})

    out = pd.concat([out, pd.DataFrame(recs, index=out.index)], axis=1)
    out = out.sort_values("EV", ascending=False).reset_index(drop=True)

    mode = f"fixed line {args.line}" if args.line is not None else "real book line per pitcher"
    print("\n" + "=" * 104)
    print(f"  v3 CALIBRATED PREDICTIONS  -  {target_date}  ({mode}, "
          f"+EV filter, edge >= {args.min_edge})")
    print("=" * 104)
    cols = ["pitcher_name", "opponent", "pred_strikeouts", "line", "odds",
            "call", "model_prob", "fair_prob", "edge", "EV", "PLAY"]
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(out[[c for c in cols if c in out.columns]].to_string(index=False))
    n_plays = int((out["PLAY"] != "— pass —").sum())
    print("=" * 104)
    print(f"+EV plays flagged (edge >= {args.min_edge}): {n_plays} of {len(out)}")

    out_path = args.output or f"reports/predictions/realtime_v3_predictions_{target_date}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
