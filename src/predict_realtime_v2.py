"""
Real-time strikeout prediction with the v2 (pre-game) BNN.

This serves the model trained by train_bnn_v2.py. It rebuilds the SAME
leakage-safe features used in training, but for tomorrow's scheduled starters:

  - Pitcher recent form (lagged): rolling means over the pitcher's most recent
    starts this season - the very quantities train_bnn_v2 shifts into as the
    "prior games" view (recent K, K/BF, whiff%, CSW%, velo, batters faced),
    plus season-to-date rates and days of rest.
  - Opponent lineup: the opposing team's current-season batting K% (overall and
    vs the starter's handedness), from the MLB Stats API team-hitting endpoint.
  - Context: home/away and handedness, from the mlb.com/scores feed.

Because train and serve build features identically, the model sees the same
kind of inputs it learned on. Outputs a predictive mean, std, and calibrated
P(K > line) via MC-Dropout - a ranked over/under table plus a CSV.

Self-contained: reuses only the v2 model class / inference helpers from
train_bnn_v2 and the read-only config helpers. Changes no existing file.

Run from the strikeout_predictor project root:
    python -m src.predict_realtime_v2                 # tomorrow's slate
    python -m src.predict_realtime_v2 --date 2026-06-26
"""

import argparse
import json
import os
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import torch

from src.utils import load_config, ensure_directories, set_seed
from src.train_bnn_v2 import (
    MCDropoutRegressor, mc_predict, compute_probabilities,
    STARTER_MIN_PITCHES, ROLL_WINDOWS, THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Starters + matchups from the mlb.com/scores feed (with team ids + home flag)
# ---------------------------------------------------------------------------
def fetch_scoreboard_starters(target_date: str) -> list[dict]:
    """One dict per posted starter: pitcher id/name, their team id/name, the
    opponent team id/name, and whether they pitch at home. TBD sides skipped."""
    import requests

    url = "https://bdfed.stitch.mlbinfra.com/bdfed/transform-mlb-scoreboard"
    params = [
        ("stitch_env", "prod"), ("sortTemplate", "4"), ("sportId", "1"),
        ("startDate", target_date), ("endDate", target_date), ("language", "en"),
    ] + [("gameType", gt) for gt in ("E", "S", "R", "F", "D", "L", "W", "A", "P")]

    try:
        resp = requests.get(
            url, params=params, timeout=25,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not reach mlb.com scoreboard API: {exc}")
        return []

    dates = data.get("dates", [])
    games = dates[0].get("games", []) if dates else data.get("games", [])

    out = []
    for game in games:
        teams = game.get("teams", {})
        home, away = teams.get("home", {}), teams.get("away", {})
        home_t, away_t = home.get("team", {}), away.get("team", {})

        for side, opp, is_home in [(home, away_t, 1), (away, home_t, 0)]:
            pp = side.get("probablePitcher")
            if not pp or pp.get("id") is None:
                continue
            out.append({
                "pitcher_id": int(pp["id"]),
                "pitcher_name": pp.get("fullName", str(pp["id"])),
                "team_name": side.get("team", {}).get("name", ""),
                "opp_team_id": opp.get("id"),
                "opp_team_name": opp.get("name", ""),
                "is_home": is_home,
                "game_time": game.get("gameDate", ""),
            })
    return out


# ---------------------------------------------------------------------------
# Opponent batting K% (overall + vs handedness) from the MLB Stats API
# ---------------------------------------------------------------------------
def fetch_team_k_rates(season: int) -> dict:
    """Return {team_id: {"overall": kr, "L": kr_vs_LHP, "R": kr_vs_RHP}} using
    strikeOuts / plateAppearances from the statsapi team-hitting endpoint."""
    import requests

    h = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    url = "https://statsapi.mlb.com/api/v1/teams/stats"

    def pull(params):
        try:
            r = requests.get(url, params=params, headers=h, timeout=25)
            r.raise_for_status()
            return r.json().get("stats", [{}])[0].get("splits", [])
        except Exception as exc:  # noqa: BLE001
            print(f"    [WARN] team stats pull failed: {exc}")
            return []

    def kr(split):
        st = split.get("stat", {})
        pa = st.get("plateAppearances") or 0
        return (st.get("strikeOuts", 0) / pa) if pa else np.nan

    rates = {}
    for s in pull({"season": season, "group": "hitting", "stats": "season", "sportIds": 1}):
        tid = s.get("team", {}).get("id")
        if tid is not None:
            rates[int(tid)] = {"overall": kr(s), "L": np.nan, "R": np.nan}

    for code, hand in [("vl", "L"), ("vr", "R")]:
        for s in pull({"season": season, "group": "hitting", "stats": "statSplits",
                       "sitCodes": code, "sportIds": 1}):
            tid = s.get("team", {}).get("id")
            if tid is not None and int(tid) in rates:
                rates[int(tid)][hand] = kr(s)
    return rates


# ---------------------------------------------------------------------------
# Pitcher recent-form features (mirrors train_bnn_v2's per-start stats)
# ---------------------------------------------------------------------------
def pull_pitcher_starts(pitcher_id: int, target_date: str, max_seasons_back: int = 2) -> pd.DataFrame:
    """Per-start stats for a pitcher (k, bf, whiff%, csw%, velo, pitches), most
    recent season with data. Built identically to build_pitcher_game_table."""
    from pybaseball import statcast_pitcher

    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    for offset in range(max_seasons_back + 1):
        year = target.year - offset
        start = f"{year}-01-01"
        end = min(target, date(year, 12, 31)).strftime("%Y-%m-%d")
        try:
            df = statcast_pitcher(start, end, pitcher_id)
        except Exception as exc:  # noqa: BLE001
            print(f"    [WARN] statcast pull failed for {pitcher_id} ({year}): {exc}")
            continue
        if df is None or df.empty:
            continue

        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["is_whiff"] = df["description"].isin(["swinging_strike", "swinging_strike_blocked"])
        df["is_csw"] = df["description"].isin(
            ["swinging_strike", "swinging_strike_blocked", "called_strike"]
        )
        df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        ab_last = df.groupby(["game_pk", "at_bat_number"], as_index=False).tail(1).copy()
        ab_last["is_k"] = ab_last["events"].fillna("").str.lower().str.contains("strikeout").astype(int)

        per_ab = ab_last.groupby(["game_pk", "game_date"], as_index=False).agg(
            k=("is_k", "sum"), bf=("is_k", "size"))
        per_pitch = df.groupby(["game_pk", "game_date"], as_index=False).agg(
            pitches=("pitch_type", "size"),
            whiff_rate=("is_whiff", "mean"),
            csw_rate=("is_csw", "mean"),
            velo=("release_speed", "mean"),
            p_throws=("p_throws", "first"))
        starts = per_ab.merge(per_pitch, on=["game_pk", "game_date"], how="inner")
        starts["k_rate"] = starts["k"] / starts["bf"].clip(lower=1)
        starts = starts[starts["pitches"] >= STARTER_MIN_PITCHES].copy()
        if not starts.empty:
            return starts.sort_values("game_date").reset_index(drop=True)
    return pd.DataFrame()


def build_feature_row(starts: pd.DataFrame, target_date: str, is_home: int,
                      opp_rates: dict, feature_cols: list[str]) -> tuple[pd.Series, dict]:
    """Assemble one v2 feature row from a pitcher's recent starts + opponent rates."""
    names = {"k": "k", "k_rate": "k_rate", "whiff_rate": "whiff_rate",
             "csw_rate": "csw_rate", "velo": "velo", "bf": "bf"}
    row = {}
    for src, name in names.items():
        vals = starts[src].to_numpy(dtype=float)
        for w in ROLL_WINDOWS:
            row[f"roll{w}_{name}"] = float(np.nanmean(vals[-w:])) if len(vals) else np.nan
        row[f"std_{name}"] = float(np.nanmean(vals)) if len(vals) else np.nan

    last_start = pd.to_datetime(starts["game_date"].max())
    days_rest = (pd.to_datetime(target_date) - last_start).days
    row["days_rest"] = float(min(days_rest, 14))
    row["is_home"] = float(is_home)

    throws = str(starts["p_throws"].mode().iloc[0]) if not starts.empty else "R"
    row["throws_R"] = 1.0 if throws == "R" else 0.0
    row["opp_k_rate"] = opp_rates.get("overall", np.nan)
    row["opp_k_rate_hand"] = opp_rates.get(throws, np.nan)

    meta = {
        "n_starts": len(starts),
        "last_start": last_start.strftime("%Y-%m-%d"),
        "throws": throws,
        "recent_k": row["roll5_k"],
    }
    return pd.Series({c: row.get(c, np.nan) for c in feature_cols}), meta


# ---------------------------------------------------------------------------
# Load model + main
# ---------------------------------------------------------------------------
def load_v2(cfg):
    with open("models/bnn_v2_features.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    with open("models/bnn_v2_impute.json", "r", encoding="utf-8") as f:
        medians = json.load(f)
    scaler = joblib.load("models/bnn_v2_scaler.pkl")
    model = MCDropoutRegressor(
        input_dim=len(feature_cols),
        hidden_sizes=tuple(cfg["nn"]["hidden_sizes"]),
        dropout=cfg["nn"]["dropout"],
    )
    model.load_state_dict(torch.load("models/bnn_v2_model.pt", map_location="cpu"))
    return model, scaler, feature_cols, medians


def parse_args():
    p = argparse.ArgumentParser(description="Real-time strikeout predictions (v2 pre-game BNN).")
    p.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: tomorrow).")
    p.add_argument("--line", type=float, default=5.5, help="Over/under line (default: 5.5).")
    p.add_argument("--mc-samples", type=int, default=None, help="MC-Dropout samples.")
    p.add_argument("--output", default=None, help="Output CSV path.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    ensure_directories()
    set_seed(cfg["project"]["seed"])

    target_date = args.date or (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    season = datetime.strptime(target_date, "%Y-%m-%d").year
    mc_samples = args.mc_samples or cfg["bnn"]["mc_samples"]

    print(f"Target date: {target_date}")
    print("Loading v2 (pre-game) BNN...")
    model, scaler, feature_cols, medians = load_v2(cfg)

    print("Fetching starters from mlb.com/scores feed...")
    starters = fetch_scoreboard_starters(target_date)
    if not starters:
        print("No posted starters for this date.")
        return
    print(f"Found {len(starters)} starter(s).")

    print("Fetching opponent team batting K% (overall + vs handedness)...")
    team_k = fetch_team_k_rates(season)
    if not team_k:
        print("  [WARN] no team K rates; opponent features will be imputed.")

    rows, meta = [], []
    for s in starters:
        print(f"  - {s['pitcher_name']} vs {s['opp_team_name']} ...", end=" ")
        starts = pull_pitcher_starts(s["pitcher_id"], target_date)
        if starts.empty:
            print("no Statcast starts, skipping.")
            continue
        opp = team_k.get(int(s["opp_team_id"]), {}) if s.get("opp_team_id") else {}
        vec, m = build_feature_row(starts, target_date, s["is_home"], opp, feature_cols)
        rows.append(vec)
        meta.append({
            "pitcher_name": s["pitcher_name"], "team": s["team_name"],
            "opponent": s["opp_team_name"], "is_home": s["is_home"],
            "n_starts": m["n_starts"], "last_start": m["last_start"],
            "recent_avg_k": round(m["recent_k"], 2) if pd.notna(m["recent_k"]) else None,
        })
        print(f"{m['n_starts']} starts, last {m['last_start']}.")

    if not rows:
        print("No pitchers had usable data.")
        return

    feat = pd.DataFrame(rows)[feature_cols]
    feat = feat.fillna(pd.Series(medians))          # same impute as training
    X = scaler.transform(feat.values.astype(np.float32))
    X_t = torch.tensor(X, dtype=torch.float32)

    mc_preds, pred_mean, pred_std = mc_predict(model, X_t, n_samples=mc_samples)
    probs = compute_probabilities(mc_preds, THRESHOLDS)

    out = pd.DataFrame(meta)
    out["game_date"] = target_date
    out["pred_strikeouts"] = np.round(pred_mean, 2)
    out["pred_std"] = np.round(pred_std, 2)
    for t in THRESHOLDS:
        out[f"prob_over_{t}"] = np.round(probs[f"prob_over_{t}"], 3)
    line_col = f"prob_over_{args.line}"
    if line_col in out.columns:
        out["ou_call"] = np.where(out[line_col] >= 0.5, "OVER", "UNDER")
        out["ou_conf"] = np.where(out[line_col] >= 0.5, out[line_col], 1 - out[line_col])
    out = out.sort_values("pred_strikeouts", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 82)
    print(f"  PREDICTED STRIKEOUTS (v2)  -  {target_date}  (line = {args.line})")
    print("=" * 82)
    cols = ["pitcher_name", "opponent", "pred_strikeouts", "pred_std", line_col]
    if "ou_call" in out.columns:
        cols += ["ou_call", "ou_conf"]
    with pd.option_context("display.max_rows", None, "display.width", 130):
        print(out[cols].to_string(index=False))
    print("=" * 82)

    out_path = args.output or f"reports/predictions/realtime_v2_predictions_{target_date}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
