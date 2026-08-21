"""
v4 feature engineering - v3 plus recency/trend signals, built for a model that
trains on 2022-2025 and serves 2026.

Adds on top of features_v3 (all leakage-safe, all serveable at prediction time):

  - opp_recent_k : opponent team K% over their LAST 10 GAMES (lagged rolling),
    not season-to-date. By mid-season the season aggregate is stale - it can't
    see a lineup slumping, surging, or churned by trades/injuries. Served via
    the MLB Stats API lastXGames split (season K% fallback).
  - velo_delta / k_rate_delta / whiff_delta : recent form (roll3) minus career
    baseline (std_). "Is this pitcher currently better or worse than his
    identity?" - velocity decline is the classic fatigue/injury tell. Trees and
    MLPs struggle to synthesise differences of correlated columns on their own.
  - month : season phase (April cold starts, September fatigue both move K rates).

Used by train_bnn_v4.py (training) and predict_realtime_v4.py (serving).
No existing file is modified; v3 remains fully intact for rollback.
"""

import numpy as np
import pandas as pd

from src.train_bnn_v2 import (
    build_pitcher_game_table,
    add_pitcher_rolling_features,
    add_opponent_features,
    MIN_PRIOR_STARTS,
)
from src.features_v3 import (
    V3_FEATURES,
    add_roll_pitches,
    build_batter_lineup_k,
    build_park_factors,
)

OPP_RECENT_GAMES = 10

_V4_NEW = ["opp_recent_k", "velo_delta", "k_rate_delta", "whiff_delta", "month"]
V4_FEATURES = V3_FEATURES + _V4_NEW


# ---------------------------------------------------------------------------
# Opponent recent-form K% (last N team games, lagged)
# ---------------------------------------------------------------------------
def add_opp_recent_k(pg_all: pd.DataFrame, starts: pd.DataFrame) -> pd.DataFrame:
    """Rolling (lagged) K% of the opposing lineup over its last OPP_RECENT_GAMES
    games. Mirrors add_opponent_features' merge; every team bats every game."""
    tg = pg_all.groupby(["opponent_team", "game_pk", "game_date"], as_index=False).agg(
        k=("k", "sum"), pa=("bf", "sum")
    ).rename(columns={"opponent_team": "batting_team"})
    tg = tg.sort_values(["batting_team", "game_date"]).reset_index(drop=True)
    gb = tg.groupby("batting_team", group_keys=False)
    k_sh, pa_sh = gb["k"].shift(1), gb["pa"].shift(1)  # exclude current game
    roll_k = (k_sh.groupby(tg["batting_team"]).rolling(OPP_RECENT_GAMES, min_periods=3)
              .sum().reset_index(level=0, drop=True))
    roll_pa = (pa_sh.groupby(tg["batting_team"]).rolling(OPP_RECENT_GAMES, min_periods=3)
               .sum().reset_index(level=0, drop=True))
    tg["opp_recent_k"] = roll_k / roll_pa
    out = starts.merge(
        tg[["batting_team", "game_pk", "opp_recent_k"]],
        left_on=["opponent_team", "game_pk"],
        right_on=["batting_team", "game_pk"], how="left",
    ).drop(columns=["batting_team"])
    return out


# ---------------------------------------------------------------------------
# Trend deltas + season phase (from columns that already exist)
# ---------------------------------------------------------------------------
def add_trend_features(starts: pd.DataFrame) -> pd.DataFrame:
    starts["velo_delta"] = starts["roll3_velo"] - starts["std_velo"]
    starts["k_rate_delta"] = starts["roll3_k_rate"] - starts["std_k_rate"]
    starts["whiff_delta"] = starts["roll3_whiff_rate"] - starts["std_whiff_rate"]
    starts["month"] = pd.to_datetime(starts["game_date"]).dt.month.astype(float)
    return starts


# ---------------------------------------------------------------------------
# Assemble the full v4 training table
# ---------------------------------------------------------------------------
def assemble_training_table_v4(df: pd.DataFrame, park_train_before):
    """Return (table, feature_cols, park_factors). Same construction as v3's
    assemble_training_table plus the v4 features. park_train_before bounds the
    games used to learn park factors (pass the test-period start)."""
    pg_all = build_pitcher_game_table(df)
    starts = add_pitcher_rolling_features(pg_all)
    starts = add_opponent_features(pg_all, starts)
    starts = add_roll_pitches(starts)
    starts = add_opp_recent_k(pg_all, starts)

    lineup = build_batter_lineup_k(df)
    starts = starts.merge(lineup, on=["game_pk", "pitcher"], how="left")

    park = build_park_factors(pg_all, park_train_before)
    starts["park_k_factor"] = starts["home_team"].map(park).fillna(1.0)

    starts["throws_R"] = (starts["p_throws"] == "R").astype(int)
    starts = add_trend_features(starts)
    starts = starts[starts["prior_starts"] >= MIN_PRIOR_STARTS].copy().reset_index(drop=True)
    return starts, V4_FEATURES, park
