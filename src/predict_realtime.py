"""
Real-time strikeout prediction.

Predicts how many strikeouts each probable starting pitcher will record in
the *next* day's MLB games, using the trained Bayesian Neural Network
(MC-Dropout regressor) from `train_bnn.py`.

----------------------------------------------------------------------------
IMPORTANT MODELLING NOTE (read this)
----------------------------------------------------------------------------
The BNN was trained on *same-game* features: every feature in
`models/bnn_features.json` (velo_mean, spin_mean, pitch-mix pct_*, zone_rate,
pct_two_strike, n_pitches, ...) is aggregated from the pitches actually thrown
in the game whose strikeout total is the label.

For a game that has not happened yet those pitches do not exist, so we cannot
compute the true features. Instead we *approximate* each pitcher's feature
vector from their recent form: we rebuild the exact same per-game features from
their most recent starts and take an exponentially-decay-weighted average
(most recent start weighted highest). This is a standard, defensible proxy, but
it introduces a small train/serve mismatch that you should acknowledge when
reporting results. Nothing here retrains or changes the model.

----------------------------------------------------------------------------
Pipeline
----------------------------------------------------------------------------
1. Determine the target date (default: tomorrow).
2. Get the probable starting pitchers:
     - auto-fetch from the MLB Stats API (statsapi.mlb.com), OR
     - use a manual list passed with --pitchers (names or MLBAM ids).
3. For each pitcher, pull their Statcast from the most recent season that has
   data, rebuild the per-game features (identical logic to make_features.py),
   and decay-weight-average the recent starts into one feature vector.
4. Scale with the saved StandardScaler and run MC-Dropout inference to get a
   predictive mean, std, and P(strikeouts > line) for sportsbook-style lines.
5. Print a ranked table and save a CSV to reports/.

Run from the strikeout_predictor project root, e.g.:
    python -m src.predict_realtime
    python -m src.predict_realtime --date 2026-06-25
    python -m src.predict_realtime --pitchers "Gerrit Cole, Tarik Skubal"
    python -m src.predict_realtime --pitchers "543037, 669373"
"""

import argparse
import json
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import torch

# Reuse the project's own code so this stays in lockstep with training.
from src.utils import load_config, ensure_directories, set_seed
from src.train_bnn import MCDropoutRegressor, mc_predict, compute_probabilities
from src.make_features import (
    zone_rate_from_zone,
    safe_nanquantile,
    STARTER_MIN_PITCHES,
)
from src.fangraphs_scraper import (
    scrape_fangraphs_pitching,
    build_fangraphs_overrides,
    apply_fangraphs_override,
)

# Same sportsbook-style lines used during training/evaluation.
THRESHOLDS = [3.5, 4.5, 5.5, 6.5]

# Columns dropped before modelling in train_bnn.py (kept for identification).
ID_COLS = ["game_pk", "game_date", "game_year", "pitcher"]


# ---------------------------------------------------------------------------
# Probable-pitcher discovery
# ---------------------------------------------------------------------------
def fetch_probable_pitchers(target_date: str) -> list[dict]:
    """
    Query the public MLB Stats API for probable starting pitchers on a date.

    Returns a list of dicts:
        {pitcher_id, pitcher_name, team, opponent, game_time}
    One entry per probable starter (home and away).
    """
    import requests  # pulled in transitively by pybaseball

    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": target_date,
        "hydrate": "probablePitcher,team",
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        print(f"[WARN] Could not reach MLB Stats API: {exc}")
        return []

    out = []
    for day in data.get("dates", []):
        for game in day.get("games", []):
            game_time = game.get("gameDate", "")
            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})

            home_name = home.get("team", {}).get("name", "Home")
            away_name = away.get("team", {}).get("name", "Away")

            for side, opp_name in [(home, away_name), (away, home_name)]:
                pp = side.get("probablePitcher")
                if not pp:
                    continue
                out.append(
                    {
                        "pitcher_id": int(pp["id"]),
                        "pitcher_name": pp.get("fullName", str(pp["id"])),
                        "team": side.get("team", {}).get("name", ""),
                        "opponent": opp_name,
                        "game_time": game_time,
                    }
                )
    return out


def fetch_scoreboard_pitchers(target_date: str) -> list[dict]:
    """
    Get the starting pitchers shown on mlb.com/scores/<date>.

    mlb.com/scores is a JS app backed by the public "stitch" scoreboard API
    (bdfed.stitch.mlbinfra.com) - the same feed the page renders, returned as
    clean JSON with MLBAM ids, so we read it directly rather than scraping HTML.
    Preferred over the statsapi probablePitcher hydrate because it mirrors
    exactly what the user sees on the scoreboard (and updates to the actual
    starter closer to game time).

    Returns the same shape as fetch_probable_pitchers(): one dict per posted
    starter; games with a TBD starter are skipped for that side.
    """
    import requests

    url = "https://bdfed.stitch.mlbinfra.com/bdfed/transform-mlb-scoreboard"
    # gameType repeated to cover spring/regular/postseason slates.
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
        game_time = game.get("gameDate", "")
        teams = game.get("teams", {})
        home, away = teams.get("home", {}), teams.get("away", {})
        home_name = home.get("team", {}).get("name", "Home")
        away_name = away.get("team", {}).get("name", "Away")

        for side, opp_name in [(home, away_name), (away, home_name)]:
            pp = side.get("probablePitcher")
            if not pp or pp.get("id") is None:
                continue  # TBD
            out.append(
                {
                    "pitcher_id": int(pp["id"]),
                    "pitcher_name": pp.get("fullName", str(pp["id"])),
                    "team": side.get("team", {}).get("name", ""),
                    "opponent": opp_name,
                    "game_time": game_time,
                }
            )
    return out


def resolve_manual_pitchers(tokens: list[str]) -> list[dict]:
    """
    Turn a manual list of names or MLBAM ids into pitcher dicts.

    A token of pure digits is treated as an MLBAM id; otherwise it is looked up
    by name via pybaseball.playerid_lookup ("First Last").
    """
    from pybaseball import playerid_lookup, playerid_reverse_lookup

    out = []
    for raw in tokens:
        token = raw.strip()
        if not token:
            continue

        if token.isdigit():
            pid = int(token)
            name = str(pid)
            try:
                rev = playerid_reverse_lookup([pid], key_type="mlbam")
                if not rev.empty:
                    name = f"{rev.iloc[0]['name_first']} {rev.iloc[0]['name_last']}".title()
            except Exception:  # noqa: BLE001
                pass
            out.append(
                {"pitcher_id": pid, "pitcher_name": name, "team": "", "opponent": "", "game_time": ""}
            )
            continue

        parts = token.split()
        if len(parts) < 2:
            print(f"[WARN] Cannot parse pitcher name '{token}' (need 'First Last'). Skipping.")
            continue

        first, last = parts[0], " ".join(parts[1:])
        try:
            look = playerid_lookup(last, first)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Lookup failed for '{token}': {exc}. Skipping.")
            continue

        look = look.dropna(subset=["key_mlbam"])
        if look.empty:
            print(f"[WARN] No MLBAM id found for '{token}'. Skipping.")
            continue

        # Prefer the most recently active matching player.
        if "mlb_played_last" in look.columns:
            look = look.sort_values("mlb_played_last", ascending=False)
        row = look.iloc[0]
        out.append(
            {
                "pitcher_id": int(row["key_mlbam"]),
                "pitcher_name": f"{first} {last}".title(),
                "team": "",
                "opponent": "",
                "game_time": "",
            }
        )
    return out


# ---------------------------------------------------------------------------
# Authoritative season-to-date aggregates from an external leaderboard
# (Baseball Savant arsenal endpoints - the same Statcast feed FanGraphs shows).
#
# NOTE ON SOURCING: FanGraphs' own leaderboard is currently 403-blocked via
# pybaseball, and in any case only ~13 of the BNN's 32 features (the pitch-mix
# pct_* columns and velo_mean) exist as published aggregates anywhere. The other
# ~19 (velo_std, velo percentiles, spin_std, pfx/plate location, zone_rate,
# pct_two_strike, per-game counts) are pitch-level distribution stats that no
# site publishes and that can only come from raw Statcast. So this is an
# OVERRIDE of the sourceable subset, layered on top of the recent-form vector.
# ---------------------------------------------------------------------------

# Savant arsenal speed columns -> Statcast pitch-type code.
_ARSENAL_SPEED_TO_CODE = {
    "ff_avg_speed": "FF",
    "si_avg_speed": "SI",
    "fc_avg_speed": "FC",
    "sl_avg_speed": "SL",
    "ch_avg_speed": "CH",
    "cu_avg_speed": "CU",
    "fs_avg_speed": "FS",
    "kn_avg_speed": "KN",
    "st_avg_speed": "ST",
    "sv_avg_speed": "SV",
}


def fetch_savant_season_aggregates(target_date: str, max_seasons_back: int) -> dict:
    """
    Pull season-to-date pitch mix and per-pitch-type velocity for ALL pitchers
    from Baseball Savant, for the most recent season (<= target year) with data.

    Returns a dict keyed by MLBAM id:
        {pid: {"pct": {code: fraction}, "velo_mean": float, "season": year}}

    Pitch usage is published as a percent (0-100); we convert to the 0-1
    fractions the model's pct_* features use. velo_mean is the usage-weighted
    mean of the per-pitch-type average speeds.
    """
    from pybaseball import statcast_pitcher_arsenal_stats, statcast_pitcher_pitch_arsenal

    target = datetime.strptime(target_date, "%Y-%m-%d").date()

    for offset in range(max_seasons_back + 1):
        year = target.year - offset
        try:
            usage = statcast_pitcher_arsenal_stats(year)
            speeds = statcast_pitcher_pitch_arsenal(year)
        except Exception as exc:  # noqa: BLE001
            print(f"    [WARN] Savant arsenal pull failed for {year}: {exc}")
            continue
        if usage is None or usage.empty:
            continue

        print(f"  Savant season-to-date aggregates: using {year} "
              f"({usage['player_id'].nunique()} pitchers).")

        # speeds: one row per pitcher, wide columns per pitch type
        speed_lookup = {}
        if speeds is not None and not speeds.empty:
            for _, r in speeds.iterrows():
                pid = int(r["pitcher"])
                speed_lookup[pid] = {
                    code: r[col]
                    for col, code in _ARSENAL_SPEED_TO_CODE.items()
                    if col in speeds.columns and pd.notna(r[col])
                }

        out = {}
        for pid, grp in usage.groupby("player_id"):
            pid = int(pid)
            pct = {}
            for _, row in grp.iterrows():
                code = str(row["pitch_type"])
                pct[code] = float(row["pitch_usage"]) / 100.0

            # usage-weighted velocity over pitch types we have a speed for
            spd = speed_lookup.get(pid, {})
            num = den = 0.0
            for code, frac in pct.items():
                if code in spd:
                    num += frac * float(spd[code])
                    den += frac
            velo_mean = (num / den) if den > 0 else np.nan

            out[pid] = {"pct": pct, "velo_mean": velo_mean, "season": year}
        return out

    print("  [WARN] No Savant arsenal data found; falling back to Statcast only.")
    return {}


def apply_savant_override(vec: pd.Series, feature_cols: list[str], agg: dict) -> list[str]:
    """
    Overwrite the sourceable features (pct_* pitch mix and velo_mean) in a
    pitcher's feature row with authoritative Savant season-to-date values.

    Mutates `vec` in place and returns the list of feature names overridden.
    """
    overridden = []

    # Full pitch-mix override: every pct_* feature is set from the season mix
    # (pitch types the pitcher never threw are correctly 0).
    pct_cols = [c for c in feature_cols if c.startswith("pct_") and c != "pct_two_strike"]
    if pct_cols:
        for col in pct_cols:
            code = col[len("pct_"):]
            vec[col] = float(agg["pct"].get(code, 0.0))
            overridden.append(col)

    if "velo_mean" in feature_cols and not np.isnan(agg.get("velo_mean", np.nan)):
        vec["velo_mean"] = float(agg["velo_mean"])
        overridden.append("velo_mean")

    return overridden


# ---------------------------------------------------------------------------
# Feature rebuilding (mirrors make_features.py exactly, per game)
# ---------------------------------------------------------------------------
def compute_game_features(pitches: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild per-game starter features for a single pitcher's Statcast pitches.

    This intentionally replicates make_features.py so the inference-time
    features match the training-time features column-for-column. The
    starters-only filter (n_pitches >= STARTER_MIN_PITCHES) is applied so that
    short relief outings don't contaminate a starter's recent-form average.
    """
    gkeys = ["game_pk", "game_date", "game_year", "pitcher"]

    agg = (
        pitches.groupby(gkeys)
        .agg(
            n_pitches=("pitch_type", "size"),
            n_at_bats=("at_bat_number", "nunique"),
            velo_mean=("release_speed", "mean"),
            velo_std=("release_speed", "std"),
            velo_p10=("release_speed", lambda x: safe_nanquantile(x, 0.10)),
            velo_p50=("release_speed", lambda x: safe_nanquantile(x, 0.50)),
            velo_p90=("release_speed", lambda x: safe_nanquantile(x, 0.90)),
            spin_mean=("release_spin_rate", "mean"),
            spin_std=("release_spin_rate", "std"),
            pfx_x_mean=("pfx_x", "mean"),
            pfx_z_mean=("pfx_z", "mean"),
            plate_x_mean=("plate_x", "mean"),
            plate_z_mean=("plate_z", "mean"),
            pct_two_strike=("strikes", lambda x: (x == 2).mean()),
            zone_rate=("zone", zone_rate_from_zone),
        )
        .reset_index()
    )

    mix = pitches.groupby(gkeys + ["pitch_type"]).size().reset_index(name="count")
    mix["pct"] = mix["count"] / mix.groupby(gkeys)["count"].transform("sum")
    mix_pivot = mix.pivot_table(
        index=gkeys, columns="pitch_type", values="pct", fill_value=0.0
    )
    mix_pivot.columns = [f"pct_{c}" for c in mix_pivot.columns]
    mix_pivot = mix_pivot.reset_index()

    features = agg.merge(mix_pivot, on=gkeys, how="left")

    # Starters-only, exactly as in training.
    features = features[features["n_pitches"] >= STARTER_MIN_PITCHES].copy()
    return features


def decay_weights(n: int, decay: float) -> np.ndarray:
    """
    Exponential-decay weights for n starts ordered most-recent first.

    weight_i = decay ** i  (i = 0 is the most recent start), normalised to 1.
    decay in (0, 1]; smaller decay => recent starts dominate more.
    """
    w = decay ** np.arange(n, dtype=np.float64)
    return w / w.sum()


def weighted_feature_vector(
    game_features: pd.DataFrame, feature_cols: list[str], recent_starts: int, decay: float
) -> pd.Series | None:
    """
    Collapse a pitcher's per-game features into one decay-weighted row.

    Most recent `recent_starts` games are kept; each feature is a weighted
    average over the games where it is present (NaNs ignored). Returns None if
    the pitcher has no qualifying starts.
    """
    if game_features.empty:
        return None

    gf = game_features.sort_values("game_date", ascending=False).head(recent_starts)
    n = len(gf)
    w = decay_weights(n, decay)

    result = {}
    for col in feature_cols:
        if col not in gf.columns:
            result[col] = np.nan
            continue
        vals = gf[col].to_numpy(dtype=np.float64)
        mask = ~np.isnan(vals)
        if not mask.any():
            result[col] = np.nan
        else:
            ww = w[mask]
            result[col] = float(np.dot(vals[mask], ww) / ww.sum())

    out = pd.Series(result)
    out["_n_starts_used"] = n
    out["_last_start"] = gf["game_date"].max()
    return out


# ---------------------------------------------------------------------------
# Statcast pull with most-recent-available-season fallback
# ---------------------------------------------------------------------------
def pull_recent_form(
    pitcher_id: int, target_date: str, max_seasons_back: int
) -> pd.DataFrame:
    """
    Pull a pitcher's Statcast for the most recent season (<= target year) that
    actually has data, looking back up to `max_seasons_back` extra seasons.

    Returns raw pitch-level rows (possibly empty).
    """
    from pybaseball import statcast_pitcher

    target = datetime.strptime(target_date, "%Y-%m-%d").date()

    for offset in range(max_seasons_back + 1):
        year = target.year - offset
        start = f"{year}-01-01"
        end = min(target, date(year, 12, 31)).strftime("%Y-%m-%d")
        try:
            df = statcast_pitcher(start, end, pitcher_id)
        except Exception as exc:  # noqa: BLE001
            print(f"    [WARN] Statcast pull failed for {pitcher_id} ({year}): {exc}")
            continue

        if df is not None and not df.empty:
            # game_year is occasionally missing/typed oddly; normalise.
            if "game_year" not in df.columns:
                df["game_year"] = pd.to_datetime(df["game_date"]).dt.year
            return df

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_bnn(cfg):
    """Load the trained BNN, its scaler, and its feature order."""
    with open("models/bnn_features.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    scaler = joblib.load("models/bnn_scaler.pkl")

    model = MCDropoutRegressor(
        input_dim=len(feature_cols),
        hidden_sizes=tuple(cfg["nn"]["hidden_sizes"]),
        dropout=cfg["nn"]["dropout"],
    )
    state = torch.load("models/bnn_model.pt", map_location="cpu")
    model.load_state_dict(state)
    return model, scaler, feature_cols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Real-time BNN strikeout predictions.")
    p.add_argument(
        "--date",
        default=None,
        help="Target game date YYYY-MM-DD (default: tomorrow).",
    )
    p.add_argument(
        "--pitchers",
        default=None,
        help="Manual override: comma-separated names or MLBAM ids. "
        "If omitted, starters are auto-fetched (see --pitcher-source).",
    )
    p.add_argument(
        "--pitcher-source",
        choices=["scoreboard", "probable"],
        default="scoreboard",
        help="Where to auto-fetch starters from: 'scoreboard' = the mlb.com/scores "
        "feed (default, matches the website); 'probable' = the statsapi "
        "probablePitcher hydrate (can be less accurate).",
    )
    p.add_argument(
        "--recent-starts",
        type=int,
        default=8,
        help="Max number of recent starts to weight (default: 8).",
    )
    p.add_argument(
        "--decay",
        type=float,
        default=0.7,
        help="Exponential decay factor in (0,1]; lower = recent starts dominate (default: 0.7).",
    )
    p.add_argument(
        "--max-seasons-back",
        type=int,
        default=2,
        help="How many extra seasons to fall back to for form data (default: 2).",
    )
    p.add_argument(
        "--use-savant-arsenal",
        action="store_true",
        help="Override the sourceable features (pitch-mix pct_* and velo_mean) "
        "with Baseball Savant season-to-date authoritative values "
        "(the same Statcast feed FanGraphs shows). The remaining ~19 pitch-level "
        "features still come from the recent-form Statcast pull.",
    )
    p.add_argument(
        "--use-fangraphs",
        action="store_true",
        help="Scrape FanGraphs (headless Chrome via undetected-chromedriver) and "
        "override zone_rate (FanGraphs Zone%%, otherwise unsourceable) and velo_mean "
        "with season-to-date values. Combine with --use-savant-arsenal for max coverage.",
    )
    p.add_argument(
        "--fg-refresh",
        action="store_true",
        help="Force a fresh FanGraphs scrape (ignore the local cache).",
    )
    p.add_argument(
        "--fg-cache-hours",
        type=float,
        default=12.0,
        help="Reuse the cached FanGraphs scrape if newer than this many hours (default: 12).",
    )
    p.add_argument(
        "--mc-samples",
        type=int,
        default=None,
        help="MC-Dropout samples (default: config bnn.mc_samples).",
    )
    p.add_argument(
        "--line",
        type=float,
        default=5.5,
        help="Strikeout line used for the headline Over/Under call (default: 5.5).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: reports/realtime_predictions_<date>.csv).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    ensure_directories()
    set_seed(cfg["project"]["seed"])

    target_date = args.date or (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    mc_samples = args.mc_samples or cfg["bnn"]["mc_samples"]

    print(f"Target date: {target_date}")
    print("Loading trained Bayesian Neural Network...")
    model, scaler, feature_cols = load_bnn(cfg)

    # --- 1. Who is pitching? -------------------------------------------------
    if args.pitchers:
        print("Using manual pitcher list (override).")
        pitchers = resolve_manual_pitchers(args.pitchers.split(","))
    elif args.pitcher_source == "scoreboard":
        print(f"Fetching starters from the mlb.com/scores feed for {target_date}...")
        pitchers = fetch_scoreboard_pitchers(target_date)
    else:
        print("Auto-fetching probable starters from statsapi...")
        pitchers = fetch_probable_pitchers(target_date)

    if not pitchers:
        print(
            "\nNo pitchers to predict. Either no probable starters are posted for "
            f"{target_date} yet, or the manual list was empty/unresolved.\n"
            "Tip: probables usually appear ~1 day out; or pass --pitchers \"Name, Name\"."
        )
        return

    print(f"Found {len(pitchers)} probable starter(s).\n")

    # Optional: authoritative season-to-date aggregates from Baseball Savant.
    savant_agg = {}
    if args.use_savant_arsenal:
        print("Fetching season-to-date pitch mix + velocity from Baseball Savant...")
        savant_agg = fetch_savant_season_aggregates(target_date, args.max_seasons_back)

    # Optional: FanGraphs (headless-browser scrape) for zone_rate + velo_mean.
    fg_overrides = {}
    if args.use_fangraphs:
        season = datetime.strptime(target_date, "%Y-%m-%d").year
        print("Scraping FanGraphs for zone_rate + velocity...")
        try:
            fg_data = scrape_fangraphs_pitching(
                season, cache_hours=args.fg_cache_hours, refresh=args.fg_refresh
            )
            fg_overrides = build_fangraphs_overrides(fg_data)
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] FanGraphs scrape failed ({exc}); continuing without it.")

    # --- 2. Build a feature vector per pitcher from recent form --------------
    rows = []
    meta = []
    for p in pitchers:
        pid = p["pitcher_id"]
        print(f"  - {p['pitcher_name']} (id={pid}) ...", end=" ")

        pitches = pull_recent_form(pid, target_date, args.max_seasons_back)
        if pitches.empty:
            print("no Statcast data found, skipping.")
            continue

        game_feats = compute_game_features(pitches)
        vec = weighted_feature_vector(game_feats, feature_cols, args.recent_starts, args.decay)
        if vec is None:
            print("no qualifying starts, skipping.")
            continue

        n_used = int(vec["_n_starts_used"])
        last_start = pd.to_datetime(vec["_last_start"]).strftime("%Y-%m-%d")

        # Override sourceable features with authoritative season-to-date values.
        src_notes = []
        n_savant = n_fg = 0
        if args.use_savant_arsenal and pid in savant_agg:
            ov = apply_savant_override(vec, feature_cols, savant_agg[pid])
            n_savant = len(ov)
            src_notes.append(f"{n_savant} Savant ({savant_agg[pid]['season']})")
        if args.use_fangraphs and pid in fg_overrides:
            ov = apply_fangraphs_override(vec, feature_cols, fg_overrides[pid])
            n_fg = len(ov)
            src_notes.append(f"{n_fg} FanGraphs")

        suffix = (", " + ", ".join(src_notes)) if src_notes else ""
        print(f"using {n_used} start(s), last {last_start}{suffix}.")

        rows.append(vec[feature_cols])
        meta.append(
            {
                "pitcher_id": pid,
                "pitcher_name": p["pitcher_name"],
                "team": p.get("team", ""),
                "opponent": p.get("opponent", ""),
                "n_starts_used": n_used,
                "last_start": last_start,
                "savant_features": n_savant,
                "fangraphs_features": n_fg,
            }
        )

    if not rows:
        print("\nNo pitchers had usable recent-form data. Nothing to predict.")
        return

    feat_df = pd.DataFrame(rows).reset_index(drop=True)
    # Same NaN handling as training (fillna(0.0)) then identical scaling.
    X = feat_df[feature_cols].fillna(0.0).values
    X = scaler.transform(X)
    X_t = torch.tensor(X, dtype=torch.float32)

    # --- 3. MC-Dropout inference (identical to train_bnn.mc_predict) ---------
    mc_preds, pred_mean, pred_std = mc_predict(model, X_t, n_samples=mc_samples)
    prob_dict = compute_probabilities(mc_preds, THRESHOLDS)

    # --- 4. Assemble + present ----------------------------------------------
    out = pd.DataFrame(meta)
    out["game_date"] = target_date
    out["pred_strikeouts"] = np.round(pred_mean, 2)
    out["pred_std"] = np.round(pred_std, 2)
    for thr in THRESHOLDS:
        out[f"prob_over_{thr}"] = np.round(prob_dict[f"prob_over_{thr}"], 3)
        out[f"prob_under_{thr}"] = np.round(1.0 - prob_dict[f"prob_over_{thr}"], 3)

    line_col = f"prob_over_{args.line}"
    if line_col in out.columns:
        out["ou_call"] = np.where(out[line_col] >= 0.5, "OVER", "UNDER")
        out["ou_confidence"] = np.where(
            out[line_col] >= 0.5, out[line_col], 1.0 - out[line_col]
        )
    out = out.sort_values("pred_strikeouts", ascending=False).reset_index(drop=True)

    # Console table
    print("\n" + "=" * 78)
    print(f"  PREDICTED STRIKEOUTS  -  {target_date}  (line = {args.line})")
    print("=" * 78)
    show_cols = ["pitcher_name", "opponent", "pred_strikeouts", "pred_std", line_col]
    if "ou_call" in out.columns:
        show_cols += ["ou_call", "ou_confidence"]
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(out[show_cols].to_string(index=False))
    print("=" * 78)
    print(
        "Note: features are decay-weighted recent-form proxies, not same-game "
        "values; pred_std is the BNN's predictive uncertainty."
    )

    # --- 5. Save -------------------------------------------------------------
    out_path = args.output or f"reports/realtime_predictions_{target_date}.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved predictions to: {out_path}")


if __name__ == "__main__":
    main()
