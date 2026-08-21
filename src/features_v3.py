"""
v3 feature engineering - enhanced, leakage-safe, pre-game features.

Builds on the proven v2 pipeline (imported read-only from train_bnn_v2) and adds
the highest-signal features identified in the diagnostic study:

  - opp_lineup_k : opponent BATTER-level strikeout tendency. Instead of a single
    team-season K%, we average the season-to-date (lagged) K% of the individual
    batters the pitcher faces. This is far more granular than the team average,
    which the study showed was too coarse (importance ~0.03).
  - roll{w}_pitches / std_pitches : recent pitch-count workload. Strikeouts scale
    with batters faced, and pitch-count limits (rookies, injury returns) cap Ks -
    the single biggest source of "confident miss" outings.
  - park_k_factor : ballpark strikeout factor (e.g. Coors suppresses breaking
    balls -> fewer Ks). Learned from TRAIN games only, then applied by venue.

Everything is lagged / shifted so no information from the target game leaks in.
Umpire strike-zone tendency was intentionally left out: there is no data source
that is reliably available for BOTH historical training and future serving
without fragile scraping, so including it would break train/serve consistency.

This module is imported by train_bnn_v3.py (training) and predict_realtime_v3.py
(serving) so features are built identically on both sides. No existing file is
modified.
"""

import numpy as np
import pandas as pd

from src.train_bnn_v2 import (
    build_pitcher_game_table,
    add_pitcher_rolling_features,
    add_opponent_features,
    ROLL_WINDOWS,
    MIN_PRIOR_STARTS,
)

# Base v2 feature columns (recreated here so v3 is a strict superset).
_V2_FEATURES = (
    [f"roll{w}_{n}" for n in ["k", "k_rate", "whiff_rate", "csw_rate", "velo", "bf"]
     for w in ROLL_WINDOWS]
    + ["std_k", "std_k_rate", "std_whiff_rate", "std_csw_rate", "std_velo", "std_bf"]
    + ["days_rest", "is_home", "throws_R", "opp_k_rate", "opp_k_rate_hand"]
)
# New v3 features layered on top.
_V3_NEW = [f"roll{w}_pitches" for w in ROLL_WINDOWS] + ["std_pitches", "opp_lineup_k", "park_k_factor"]
V3_FEATURES = _V2_FEATURES + _V3_NEW


# ---------------------------------------------------------------------------
# Opponent batter-level lineup K%
# ---------------------------------------------------------------------------
def build_batter_lineup_k(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (game_pk, pitcher): mean of the season-to-date (lagged) batting K% of the
    batters that pitcher faced. Leakage-safe - each batter's rate uses only their
    games before this one.
    """
    d = df.copy()
    d["game_date"] = pd.to_datetime(d["game_date"])
    d = d.sort_values(["game_pk", "at_bat_number", "pitch_number"])

    # One row per plate appearance (last pitch of the at-bat).
    ab = d.groupby(["game_pk", "at_bat_number"], as_index=False).tail(1).copy()
    ab["is_k"] = ab["events"].fillna("").str.lower().str.contains("strikeout").astype(int)

    # Batter season-to-date K% (lagged).
    bg = ab.groupby(["batter", "game_pk", "game_date"], as_index=False).agg(
        pa=("is_k", "size"), k=("is_k", "sum"))
    bg = bg.sort_values(["batter", "game_date"]).reset_index(drop=True)
    g = bg.groupby("batter", group_keys=False)
    bg["cum_k"] = g["k"].apply(lambda s: s.shift(1).cumsum())
    bg["cum_pa"] = g["pa"].apply(lambda s: s.shift(1).cumsum())
    bg["bat_k_rate"] = bg["cum_k"] / bg["cum_pa"]

    # Attach each faced batter's lagged rate, then average per pitcher-game.
    pairs = ab[["game_pk", "pitcher", "batter"]].drop_duplicates()
    pairs = pairs.merge(bg[["batter", "game_pk", "bat_k_rate"]], on=["batter", "game_pk"], how="left")
    lineup = pairs.groupby(["game_pk", "pitcher"], as_index=False)["bat_k_rate"].mean()
    return lineup.rename(columns={"bat_k_rate": "opp_lineup_k"})


# ---------------------------------------------------------------------------
# Recent pitch-count workload
# ---------------------------------------------------------------------------
def add_roll_pitches(starts: pd.DataFrame) -> pd.DataFrame:
    """Add roll{w}_pitches and std_pitches (lagged) to the (already start-sorted) frame."""
    starts = starts.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    shifted = starts.groupby("pitcher")["pitches"].shift(1)
    for w in ROLL_WINDOWS:
        starts[f"roll{w}_pitches"] = (
            shifted.groupby(starts["pitcher"]).rolling(w, min_periods=2).mean()
            .reset_index(level=0, drop=True))
    starts["std_pitches"] = (
        shifted.groupby(starts["pitcher"]).expanding(min_periods=1).mean()
        .reset_index(level=0, drop=True))
    return starts


# ---------------------------------------------------------------------------
# Ballpark strikeout factor (learned from TRAIN games only)
# ---------------------------------------------------------------------------
def build_park_factors(pg_all: pd.DataFrame, test_start) -> dict:
    """
    park_k_factor[home_team] = (K/BF at that park) / (league K/BF), using only
    games before test_start so it never sees the test period.
    """
    train = pg_all[pd.to_datetime(pg_all["game_date"]) < pd.to_datetime(test_start)]
    league = train["k"].sum() / max(train["bf"].sum(), 1)
    by_park = train.groupby("home_team").agg(k=("k", "sum"), bf=("bf", "sum"))
    factors = (by_park["k"] / by_park["bf"].clip(lower=1) / league)
    return factors.to_dict()


# ---------------------------------------------------------------------------
# Assemble the full training table
# ---------------------------------------------------------------------------
def assemble_training_table(df: pd.DataFrame, test_start):
    """Return (table, feature_cols, park_factors). All features leakage-safe."""
    pg_all = build_pitcher_game_table(df)
    starts = add_pitcher_rolling_features(pg_all)
    starts = add_opponent_features(pg_all, starts)
    starts = add_roll_pitches(starts)

    lineup = build_batter_lineup_k(df)
    starts = starts.merge(lineup, on=["game_pk", "pitcher"], how="left")

    park = build_park_factors(pg_all, test_start)
    league_factor = 1.0
    starts["park_k_factor"] = starts["home_team"].map(park).fillna(league_factor)

    starts["throws_R"] = (starts["p_throws"] == "R").astype(int)
    starts = starts[starts["prior_starts"] >= MIN_PRIOR_STARTS].copy().reset_index(drop=True)
    return starts, V3_FEATURES, park
