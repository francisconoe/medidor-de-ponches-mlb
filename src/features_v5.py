"""
v5 feature engineering - v4 plus the two signals that a retrospective backtest of
the recent live losses showed the model was missing (both leakage-safe, both
computable from data already on hand):

  - vs_opp_career_k : the pitcher's lagged expanding-mean strikeouts against THIS
    specific opponent. The Aug backtest found 5 of 7 recent under-losses (Webb,
    King, Fedde, Senga, Perez) had a vs-opponent history clearly above the line -
    the team-season opp_k_rate could not see it. This isolates the pitcher's
    personal track record vs the matchup.
  - opp_whiff_rate : the opponent lineup's lagged season-to-date swinging-strike
    rate (contact quality), distinct from opp_k_rate. A team can strike out at a
    league-average clip yet whiff a lot (the "bad at contact overall" signal the
    Chase Burns writeup leaned on).
  - roll5_brk_whiff : the pitcher's own whiff-per-swing on SLIDERS + SWEEPERS
    over his last 5 starts (lagged). Aggregate whiff rate can hide one elite
    put-away breaking ball (Cease's slider, Burns' stuff) that drives strikeouts.
  - opp_break_whiff : the opponent lineup's lagged season-to-date whiff-per-swing
    AGAINST sliders/sweepers - the team's specific vulnerability to the pitch a
    breaking-ball pitcher lives on. roll5_brk_whiff x opp_break_whiff is the
    matchup the analysts were reading by hand.

Everything else is inherited unchanged from features_v4. v4 stays intact for
rollback.
"""

import numpy as np
import pandas as pd

from src.train_bnn_v2 import (
    build_pitcher_game_table,
    add_pitcher_rolling_features,
    add_opponent_features,
    MIN_PRIOR_STARTS,
)
from src.features_v3 import add_roll_pitches, build_batter_lineup_k, build_park_factors
from src.features_v4 import V4_FEATURES, add_opp_recent_k, add_trend_features

# Statcast pitch-type codes for the breaking balls of interest.
BREAKING = ["SL", "ST"]  # slider, sweeper
WHIFF_DESC = ["swinging_strike", "swinging_strike_blocked"]
SWING_DESC = WHIFF_DESC + ["foul", "foul_tip", "hit_into_play", "foul_bunt", "missed_bunt"]

_V5_NEW = ["vs_opp_career_k", "opp_whiff_rate", "roll5_brk_whiff", "opp_break_whiff"]
V5_FEATURES = V4_FEATURES + _V5_NEW


def add_vs_opp_history(pg_all: pd.DataFrame, starts: pd.DataFrame) -> pd.DataFrame:
    """Pitcher's lagged expanding-mean K against this exact opponent team."""
    d = pg_all[["pitcher", "opponent_team", "game_pk", "game_date", "k"]].copy()
    d = d.sort_values(["pitcher", "opponent_team", "game_date"]).reset_index(drop=True)
    g = d.groupby(["pitcher", "opponent_team"], group_keys=False)
    d["vs_opp_career_k"] = g["k"].apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
    return starts.merge(d[["pitcher", "game_pk", "vs_opp_career_k"]],
                        on=["pitcher", "game_pk"], how="left")


def add_opp_whiff(pg_all: pd.DataFrame, starts: pd.DataFrame) -> pd.DataFrame:
    """Opponent lineup's lagged season-to-date swinging-strike rate.
    Built like add_opponent_features but on whiffs/pitches, not K/BF."""
    tmp = pg_all.copy()
    tmp["whiffs"] = tmp["whiff_rate"] * tmp["pitches"]
    tg = tmp.groupby(["opponent_team", "game_pk", "game_date"], as_index=False).agg(
        whiffs=("whiffs", "sum"), pitches=("pitches", "sum")
    ).rename(columns={"opponent_team": "batting_team"})
    tg = tg.sort_values(["batting_team", "game_date"]).reset_index(drop=True)
    gb = tg.groupby("batting_team", group_keys=False)
    tg["cw"] = gb["whiffs"].apply(lambda s: s.shift(1).cumsum())
    tg["cp"] = gb["pitches"].apply(lambda s: s.shift(1).cumsum())
    tg["opp_whiff_rate"] = tg["cw"] / tg["cp"]
    return starts.merge(
        tg[["batting_team", "game_pk", "opp_whiff_rate"]],
        left_on=["opponent_team", "game_pk"], right_on=["batting_team", "game_pk"],
        how="left").drop(columns=["batting_team"])


def add_breaking_whiff(df: pd.DataFrame, starts: pd.DataFrame) -> pd.DataFrame:
    """Pitcher's lagged last-5-start whiff-per-swing on sliders + sweepers."""
    d = df[df["pitch_type"].isin(BREAKING)].copy()
    d["game_date"] = pd.to_datetime(d["game_date"])
    d["whiff"] = d["description"].isin(WHIFF_DESC)
    d["swing"] = d["description"].isin(SWING_DESC)
    pgb = d.groupby(["game_pk", "pitcher", "game_date"], as_index=False).agg(
        w=("whiff", "sum"), sw=("swing", "sum"))
    pgb["brk"] = pgb["w"] / pgb["sw"].clip(lower=1)
    pgb = pgb.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    sh = pgb.groupby("pitcher")["brk"].shift(1)              # lag: exclude this game
    pgb["roll5_brk_whiff"] = (sh.groupby(pgb["pitcher"]).rolling(5, min_periods=2)
                              .mean().reset_index(level=0, drop=True))
    return starts.merge(pgb[["game_pk", "pitcher", "roll5_brk_whiff"]],
                        on=["game_pk", "pitcher"], how="left")


def add_opp_break_vuln(df: pd.DataFrame, starts: pd.DataFrame) -> pd.DataFrame:
    """Opponent lineup's lagged season-to-date whiff-per-swing vs sliders/sweepers
    - the team's specific vulnerability to breaking balls."""
    d = df[df["pitch_type"].isin(BREAKING)].copy()
    d["game_date"] = pd.to_datetime(d["game_date"])
    d["batting_team"] = np.where(d["inning_topbot"] == "Top", d["away_team"], d["home_team"])
    d["whiff"] = d["description"].isin(WHIFF_DESC)
    d["swing"] = d["description"].isin(SWING_DESC)
    tg = d.groupby(["batting_team", "game_pk", "game_date"], as_index=False).agg(
        w=("whiff", "sum"), sw=("swing", "sum"))
    tg = tg.sort_values(["batting_team", "game_date"]).reset_index(drop=True)
    gb = tg.groupby("batting_team", group_keys=False)
    cw = gb["w"].apply(lambda s: s.shift(1).cumsum())
    csw = gb["sw"].apply(lambda s: s.shift(1).cumsum())
    tg["opp_break_whiff"] = cw / csw.clip(lower=1)
    return starts.merge(
        tg[["batting_team", "game_pk", "opp_break_whiff"]],
        left_on=["opponent_team", "game_pk"], right_on=["batting_team", "game_pk"],
        how="left").drop(columns=["batting_team"])


def assemble_training_table_v5(df: pd.DataFrame, park_train_before):
    """v4 assembly + the two v5 features. Returns (table, feature_cols, park)."""
    pg_all = build_pitcher_game_table(df)
    starts = add_pitcher_rolling_features(pg_all)
    starts = add_opponent_features(pg_all, starts)
    starts = add_roll_pitches(starts)
    starts = add_opp_recent_k(pg_all, starts)
    starts = add_vs_opp_history(pg_all, starts)
    starts = add_opp_whiff(pg_all, starts)
    starts = add_breaking_whiff(df, starts)
    starts = add_opp_break_vuln(df, starts)

    lineup = build_batter_lineup_k(df)
    starts = starts.merge(lineup, on=["game_pk", "pitcher"], how="left")

    park = build_park_factors(pg_all, park_train_before)
    starts["park_k_factor"] = starts["home_team"].map(park).fillna(1.0)

    starts["throws_R"] = (starts["p_throws"] == "R").astype(int)
    starts = add_trend_features(starts)
    starts = starts[starts["prior_starts"] >= MIN_PRIOR_STARTS].copy().reset_index(drop=True)
    return starts, V5_FEATURES, park
