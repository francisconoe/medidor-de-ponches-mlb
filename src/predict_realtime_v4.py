"""
Real-time strikeout prediction with the v4 model (fresh data, ensemble
probabilities, beta calibration, v4 features) + the odds/EV betting layer.

Same serving flow as v3, with the v4 upgrades wired in:
  - loads bnn_v4_* artifacts (v3 files untouched - run predict_realtime_v3 to
    fall back to the old model at any time);
  - P(K > line) comes from the NB tail of the ENSEMBLE mean (NN+XGB), passed
    through the beta calibrators fit at training time;
  - new features served live: opponent LAST-10-GAMES K% (statsapi lastXGames
    split, season K% fallback), form-vs-career deltas, season month.

    python -m src.predict_realtime_v4                      # tomorrow
    python -m src.predict_realtime_v4 --date 2026-07-13
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
from scipy.stats import nbinom

from src.utils import load_config, ensure_directories, set_seed
from src.train_bnn_v3 import NBDropoutNet, mc_nb_predict, THRESHOLDS
from src.train_bnn_v4 import BetaCalibrator, ens_nb_tail  # noqa: F401 (BetaCalibrator needed to unpickle)
from src.predict_realtime_v2 import pull_pitcher_starts, fetch_team_k_rates
from src.predict_realtime_v3 import fetch_scoreboard, build_v3_row, TEAMID_TO_CODE  # noqa: F401
from src.odds_api import (fetch as fetch_odds, norm_name, devig_two_way,
                          american_to_decimal)

OPP_RECENT_GAMES = 10


def fetch_team_recent_k(season: int, n_games: int = OPP_RECENT_GAMES) -> dict:
    """{team_id: last-N-games batting K%} via statsapi lastXGames split.
    Empty dict on failure (caller falls back to season K%)."""
    import requests
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/teams/stats",
            params={"season": season, "group": "hitting", "stats": "lastXGames",
                    "limit": n_games, "sportIds": 1},
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            timeout=25)
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
    except Exception as exc:  # noqa: BLE001
        print(f"    [WARN] lastXGames team stats failed: {exc}")
        return {}
    out = {}
    for s in splits:
        tid = s.get("team", {}).get("id")
        st = s.get("stat", {})
        pa = st.get("plateAppearances") or 0
        if tid is not None and pa:
            out[int(tid)] = st.get("strikeOuts", 0) / pa
    return out


def fetch_confirmed_lineups(target_date: str) -> dict:
    """{team_id: [batter_id,...]} for teams whose lineup is posted for target_date.
    Uses the MLB schedule 'lineups' hydrate. Empty for teams not yet posted."""
    import requests
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": target_date, "hydrate": "lineups"},
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        print(f"    [WARN] lineup fetch failed: {exc}")
        return {}
    out = {}
    for d in data.get("dates", []):
        for g in d.get("games", []):
            lu = g.get("lineups", {})
            t = g.get("teams", {})
            for side, key in [("home", "homePlayers"), ("away", "awayPlayers")]:
                players = lu.get(key) or []
                tid = t.get(side, {}).get("team", {}).get("id")
                if tid and players:
                    out[int(tid)] = [p["id"] for p in players if p.get("id")]
    return out


def fetch_batter_k_rates(season: int) -> dict:
    """{batter_id: season-to-date K per plate appearance} for all hitters."""
    import requests
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/stats",
            params={"stats": "season", "group": "hitting", "season": season,
                    "sportId": 1, "gameType": "R", "limit": 3000, "playerPool": "All"},
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            timeout=30)
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
    except Exception as exc:  # noqa: BLE001
        print(f"    [WARN] batter K-rate fetch failed: {exc}")
        return {}
    out = {}
    for s in splits:
        pid = s.get("player", {}).get("id")
        st = s.get("stat", {})
        pa = st.get("plateAppearances") or 0
        if pid and pa:
            out[int(pid)] = st.get("strikeOuts", 0) / pa
    return out


def confirmed_lineup_k(opp_team_id, lineups: dict, batter_kr: dict):
    """Mean season-to-date K% of the opponent's posted lineup, or None if the
    lineup isn't posted / no batter rates available."""
    ids = lineups.get(int(opp_team_id)) if opp_team_id else None
    if not ids:
        return None
    rates = [batter_kr[b] for b in ids if b in batter_kr]
    return float(np.mean(rates)) if rates else None


def build_v4_row(starts, target_date, is_home, opp, park_factor, feats,
                 opp_recent):
    """v3 row + the v4 features, aligned to the v4 feature list."""
    vec3, meta = build_v3_row(starts, target_date, is_home, opp, park_factor, feats)
    row = vec3.to_dict()
    row["velo_delta"] = row.get("roll3_velo", np.nan) - row.get("std_velo", np.nan)
    row["k_rate_delta"] = row.get("roll3_k_rate", np.nan) - row.get("std_k_rate", np.nan)
    row["whiff_delta"] = row.get("roll3_whiff_rate", np.nan) - row.get("std_whiff_rate", np.nan)
    row["month"] = float(pd.to_datetime(target_date).month)
    # Last-10-games K% when available, else the season rate (documented proxy).
    row["opp_recent_k"] = opp_recent if opp_recent is not None else opp.get("overall", np.nan)
    return pd.Series({c: row.get(c, np.nan) for c in feats}), meta


def load_v4():
    with open("models/bnn_v4_features.json", encoding="utf-8") as f:
        feats = json.load(f)
    with open("models/bnn_v4_impute.json", encoding="utf-8") as f:
        medians = json.load(f)
    with open("models/bnn_v4_park.json", encoding="utf-8") as f:
        park = json.load(f)
    with open("models/bnn_v4_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    scaler = joblib.load("models/bnn_v4_scaler.pkl")
    calibrators = joblib.load("models/bnn_v4_calibrators.pkl")
    xgbm = xgb.XGBRegressor()
    xgbm.load_model("models/bnn_v4_xgb.json")
    model = NBDropoutNet(len(feats), tuple(meta["hidden_sizes"]), meta["dropout"])
    model.load_state_dict(torch.load("models/bnn_v4_model.pt", map_location="cpu"))
    return model, scaler, feats, medians, park, meta, calibrators, xgbm


def parse_args():
    p = argparse.ArgumentParser(description="Real-time strikeout predictions (v4).")
    p.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default tomorrow).")
    p.add_argument("--line", type=float, default=None,
                   help="Force one line for every pitcher (default: book line / nearest).")
    p.add_argument("--min-edge", type=float, default=0.03,
                   help="Flag a PLAY only when calibrated prob beats the vig-free "
                        "implied prob by this much AND the bet is +EV. Default 0.03.")
    p.add_argument("--min-prob", type=float, default=0.65,
                   help="Strict filter: PLAY also requires calibrated model_prob >= this. "
                        "July-11 audit: every call at p>=0.65 hit, every miss was below. "
                        "Default 0.65.")
    p.add_argument("--bankroll", type=float, default=1000.0,
                   help="Bankroll used to size stakes (default 1000 units).")
    p.add_argument("--kelly", type=float, default=0.25,
                   help="Kelly fraction (default 0.25 = quarter-Kelly).")
    p.add_argument("--max-stake-pct", type=float, default=0.02,
                   help="Hard cap on any single stake as a fraction of bankroll (default 2%%).")
    p.add_argument("--max-prob-std", type=float, default=0.16,
                   help="Uncertainty gate: refuse a play when the MC-dropout spread of "
                        "P(over line) exceeds this (thin history / first game back / unusual "
                        "input). Stake is also scaled down toward this ceiling. Default 0.16.")
    p.add_argument("--max-side-pct", type=float, default=0.06,
                   help="Correlated-exposure cap: total stake on all 'over' (or all 'under') "
                        "plays in one night is capped at this fraction of bankroll; stakes on "
                        "that side are scaled down proportionally if exceeded. Default 6%%.")
    p.add_argument("--no-odds", action="store_true",
                   help="Skip The Odds API; assume nearest realistic line @ -110.")
    p.add_argument("--no-lineups", action="store_true",
                   help="Skip confirmed lineups; use the team-K%% proxy for opp_lineup_k "
                        "(the old day-before behaviour). Default: use posted lineups when "
                        "available, matching what the model was trained on.")
    p.add_argument("--short-fix-k", type=float, default=0.35,
                   help="Short-outing correction: K added to the projection for high-K-rate "
                        "arms coming off recent short outings (measured holdout bias). 0 to "
                        "disable. Default 0.35.")
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
    print("Loading v4 model bundle...")
    model, scaler, feats, medians, park, meta, calibrators, xgbm = load_v4()
    w = meta["ensemble_nn_weight"]

    print("Fetching starters (mlb.com/scores)...")
    starters = fetch_scoreboard(target_date)
    if not starters:
        print("No posted starters.")
        return
    print(f"Found {len(starters)} starter(s).")
    print("Fetching opponent team K% (season + last-10)...")
    team_k = fetch_team_k_rates(season)
    team_recent = fetch_team_recent_k(season)
    if args.no_lineups:
        lineups, batter_kr = {}, {}
    else:
        print("Fetching confirmed lineups + batter K% (MLB Stats API)...")
        lineups = fetch_confirmed_lineups(target_date)
        batter_kr = fetch_batter_k_rates(season) if lineups else {}
        print(f"  {len(lineups)} team lineup(s) posted, {len(batter_kr)} batter rates.")

    rows, info = [], []
    for s in starters:
        print(f"  - {s['pitcher_name']} vs {s['opp_team_name']} ...", end=" ")
        starts = pull_pitcher_starts(s["pitcher_id"], target_date)
        if starts.empty:
            print("no data, skip.")
            continue
        tid = int(s["opp_team_id"]) if s.get("opp_team_id") else None
        opp = team_k.get(tid, {}) if tid else {}
        recent = team_recent.get(tid) if tid else None
        pf = park.get(s.get("park_code"), 1.0)
        vec, m = build_v4_row(starts, target_date, s["is_home"], opp, pf, feats, recent)
        # Override the team-K% proxy with the real confirmed-lineup K% when posted
        # - this is the top-5 feature the model was trained on but live serving
        # normally can't see a day out.
        conf_lk = confirmed_lineup_k(tid, lineups, batter_kr)
        lineup_src = "proxy"
        if conf_lk is not None:
            vec["opp_lineup_k"] = conf_lk
            lineup_src = "confirmed"
        rows.append(vec)
        info.append({"pitcher_name": s["pitcher_name"], "team": s["team_name"],
                     "opponent": s["opp_team_name"], "n_starts": m["n_starts"],
                     "last_start": m["last_start"], "lineup": lineup_src})
        print(f"{m['n_starts']} starts [{lineup_src}].")
    if not rows:
        print("No usable pitchers.")
        return

    feat_df = pd.DataFrame(rows)[feats].fillna(pd.Series(medians))
    X = torch.tensor(scaler.transform(feat_df.values.astype(np.float32)))
    mu_s, r_s = mc_nb_predict(model, X, meta["mc_samples"])
    xgb_pred = xgbm.predict(feat_df)
    blend = w * mu_s.mean(0) + (1 - w) * xgb_pred

    # Short-outing correction: the 765-game holdout showed the model
    # under-projects HIGH K-rate arms coming off recent SHORT outings by ~0.35 K
    # (it reads blowup-shortened starts as reduced form, when the K-rate is
    # intact - the Harrison/Warren miss). Nudge the ensemble mean up for exactly
    # that quadrant; every other quadrant is already unbiased so leave it alone.
    if args.short_fix_k > 0:
        r3bf = feat_df["roll3_bf"].to_numpy(float)
        sbf = feat_df["std_bf"].to_numpy(float)
        kr = feat_df["roll5_k_rate"].to_numpy(float)
        short_hi = (r3bf < sbf) & (kr >= medians["roll5_k_rate"])
        blend = blend + np.where(short_hi, args.short_fix_k, 0.0)
        print(f"Short-outing correction (+{args.short_fix_k:g}K) applied to "
              f"{int(short_hi.sum())} of {len(blend)} pitchers.")

    out = pd.DataFrame(info)
    out["game_date"] = target_date
    out["pred_strikeouts"] = np.round(blend, 2)
    for line in THRESHOLDS:
        raw = ens_nb_tail(mu_s, r_s, blend, line)
        out[f"prob_over_{line}"] = np.round(calibrators[line].predict(raw), 3)

    # ---- odds / EV layer (same policy as v3 serving) -----------------------
    print("Fetching sportsbook strikeout lines (The Odds API)...")
    odds = {} if args.no_odds else fetch_odds(target_date)
    BOOK_LINES = list(THRESHOLDS)

    cal_lines = sorted(calibrators.keys())

    def cal_prob_at(i, line):
        raw = ens_nb_tail(mu_s[:, i:i + 1], r_s[:, i:i + 1], blend[i:i + 1], line)
        cal = calibrators[min(cal_lines, key=lambda t: abs(t - line))]
        return float(np.clip(cal.predict(raw)[0], 0.0, 1.0))

    def prob_uncertainty(i, line):
        """MC-dropout spread of P(K > line): std of the per-sample NB tail using
        each dropout sample's own (mu, r). Wide spread = the model itself is
        unsure (thin history, unusual input), distinct from a confident mean."""
        m = int(np.ceil(line))
        mu_col, r_col = mu_s[:, i], r_s[:, i]
        p = r_col / (r_col + mu_col)
        return float(nbinom.sf(m - 1, r_col, p).std())

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
        else:
            book_line, over_odds, under_odds, book = assumed_line, -110.0, -110.0, "(assumed)"

        p_over = cal_prob_at(i, book_line)
        p_under = 1.0 - p_over
        fair_over, fair_under = devig_two_way(over_odds, under_odds)
        ev_over = p_over * american_to_decimal(over_odds) - 1.0
        ev_under = p_under * american_to_decimal(under_odds) - 1.0
        if ev_over >= ev_under:
            side, p, fair, ev, price = "over", p_over, fair_over, ev_over, over_odds
        else:
            side, p, fair, ev, price = "under", p_under, fair_under, ev_under, under_odds
        edge = p - fair
        call = f"{side} {book_line:g}"
        # Role/data-sufficiency guard: openers, spot starters and injury returns
        # (Danny Young: 1 career start, last in 2024) get median-imputed features
        # and fabricated confidence. Require a real recent starter's history.
        n_st = int(out["n_starts"].iloc[i])
        days_gap = (pd.to_datetime(target_date) - pd.to_datetime(out["last_start"].iloc[i])).days
        data_ok = n_st >= 8 and days_gap <= 21
        # Calibration-range guard: only stake on lines we fit calibrators for
        # (2.5-7.5). Anything beyond (Wheeler 9.5) is extrapolation - never bet it.
        line_ok = cal_lines[0] <= book_line <= cal_lines[-1]
        # Elite-arm under caution: 2026 audit showed confident unders on
        # top-quartile arms (proj >= 6) land ~5pp below their stated probability.
        # Demand a stiffer edge and prob before fading a genuine bat-misser.
        bump = 0.05 if (side == "under" and proj >= 6.0) else 0.0
        # Uncertainty gate: even a high-mean bet is refused if the model's own
        # MC-dropout spread on it is wide (the July drawdown was full of these).
        unc = prob_uncertainty(i, book_line)
        unc_ok = unc <= args.max_prob_std
        is_play = (ev > 0.0 and edge >= args.min_edge + bump
                   and p >= args.min_prob + bump and data_ok and line_ok and unc_ok)
        if is_play:
            play = call
        elif not data_ok:
            play = "— thin data —"
        elif not line_ok:
            play = "— off-range line —"
        elif not unc_ok:
            play = "— high uncertainty —"
        else:
            play = "— pass —"
        # Fractional Kelly stake, hard-capped per bet, and scaled DOWN as the
        # model's uncertainty on this bet rises toward the ceiling.
        b = american_to_decimal(price) - 1.0
        kelly_f = max(0.0, (p * b - (1.0 - p)) / b) * args.kelly
        unc_mult = max(0.25, 1.0 - unc / args.max_prob_std)
        stake = round(args.bankroll * min(kelly_f * unc_mult, args.max_stake_pct), 2) if is_play else 0.0
        recs.append({"line": book_line, "odds": int(round(price)), "book": book,
                     "call": call, "model_prob": round(p, 3), "fair_prob": round(fair, 3),
                     "edge": round(edge, 3), "EV": round(ev, 3), "unc": round(unc, 3),
                     "stake": stake, "PLAY": play})

    out = pd.concat([out, pd.DataFrame(recs, index=out.index)], axis=1)

    # Correlated-exposure cap: all 'over' plays (or all 'under' plays) in one
    # night lose together if strikeouts run hot/cold league-wide (weather, ump
    # crew, etc.) - exactly the July 17 / 24-25 / 29 cluster losses. Cap total
    # staked per side at max_side_pct of bankroll, scaling that side down pro
    # rata. Operates only on flagged plays (stake > 0).
    side_cap = args.max_side_pct * args.bankroll
    played = out["PLAY"] == out["call"]
    for sd in ("over", "under"):
        m = played & out["call"].str.startswith(sd)
        tot = out.loc[m, "stake"].sum()
        if tot > side_cap and tot > 0:
            scale = side_cap / tot
            out.loc[m, "stake"] = (out.loc[m, "stake"] * scale).round(2)
            print(f"[exposure] {sd} stake {tot:.2f} > cap {side_cap:.2f} -> scaled x{scale:.2f}")

    # Tiered sort: recommended plays first, then every "over" call (even if
    # not recommended), then the rest - EV descending within each tier.
    is_play = out["PLAY"] == out["call"]
    is_over = out["call"].str.startswith("over")
    out["_tier"] = np.where(is_play, 0, np.where(is_over, 1, 2))
    out = out.sort_values(["_tier", "EV"], ascending=[True, False]).drop(columns="_tier")
    out = out.reset_index(drop=True)

    mode = f"fixed line {args.line}" if args.line is not None else "book line per pitcher"
    print("\n" + "=" * 112)
    print(f"  v4 PREDICTIONS  -  {target_date}  ({mode}; PLAY needs +EV, edge >= "
          f"{args.min_edge}, prob >= {args.min_prob}, unc <= {args.max_prob_std}; "
          f"uncertainty-scaled Kelly, per-side cap {args.max_side_pct:g})")
    print("=" * 112)
    cols = ["pitcher_name", "opponent", "pred_strikeouts", "line", "odds",
            "call", "model_prob", "fair_prob", "edge", "EV", "unc", "stake", "lineup", "PLAY"]
    with pd.option_context("display.max_rows", None, "display.width", 180):
        print(out[[c for c in cols if c in out.columns]].to_string(index=False))
    n_plays = int((out["PLAY"] == out["call"]).sum())
    n_conf = int((out["lineup"] == "confirmed").sum()) if "lineup" in out.columns else 0
    print("=" * 112)
    print(f"Plays flagged: {n_plays} of {len(out)}   total staked: {out['stake'].sum():.2f}"
          f"   confirmed lineups: {n_conf}/{len(out)}")

    out_path = args.output or f"reports/predictions/realtime_v4_predictions_{target_date}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
