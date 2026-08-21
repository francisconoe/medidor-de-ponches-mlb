"""
BNN v2 - a real-time-native strikeout model (self-contained, standalone).

WHY THIS EXISTS
---------------
The original BNN (train_bnn.py) is trained on SAME-GAME features: velocity,
spin, pitch mix and zone-rate aggregated from the pitches thrown in the very
game whose strikeout total is the label. Those values do not exist before a
game is played, and they mostly encode "who the pitcher is" rather than what
will happen on a given night. That makes the model (a) impossible to serve
honestly in real time and (b) blind to the single biggest driver of strikeouts:
the opposing lineup.

This version fixes both. Every feature is computable BEFORE first pitch and is
built the same way at train and serve time (no leakage), using two ideas:

  1. Pitcher recent form, LAGGED - rolling stats over the pitcher's PRIOR
     starts only (shifted so the current game is excluded):
     recent strikeouts, K-per-batter, whiff%, CSW% (called+swinging strike
     rate), velocity, batters faced (a workload proxy), plus season-to-date
     rate and days of rest.

  2. Opponent lineup strikeout tendency, LAGGED - the opposing team's
     season-to-date batting K% (overall, and specifically versus the current
     starter's handedness), computed only from games before the target game.

Same model class as the original (MC-Dropout Bayesian NN) so this is a clean
"same model, better inputs" comparison, and it still yields a predictive
distribution -> calibrated P(K > line) for over/under.

NOTHING ELSE IS IMPORTED FROM THE PROJECT except the read-only config/seed
helpers, and NO existing file is modified. All artifacts are written with a
`_v2` suffix so the original model's outputs are untouched.

Run it (from the strikeout_predictor project root):
    python -m src.train_bnn_v2
"""

import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils import load_config, ensure_directories, set_seed

STARTER_MIN_PITCHES = 40          # a start (matches make_features.py)
MIN_PRIOR_STARTS = 3              # need this much pitcher history to predict
THRESHOLDS = [3.5, 4.5, 5.5, 6.5]  # sportsbook-style lines
ROLL_WINDOWS = (3, 5, 10)


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


# ---------------------------------------------------------------------------
# Model (self-contained: MC-Dropout Bayesian NN)
# ---------------------------------------------------------------------------
class MCDropoutRegressor(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(256, 128), dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.out = nn.Linear(hidden_sizes[1], 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.out(x).squeeze(1)


def mc_predict(model, X_tensor, n_samples=100):
    """MC-Dropout inference: keep dropout ON and sample. Returns (samples, mean, std)."""
    model.train()  # dropout active at test time
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(X_tensor).cpu().numpy())
    preds = np.stack(preds, axis=0)
    return preds, preds.mean(axis=0), preds.std(axis=0)


def compute_probabilities(mc_preds, thresholds):
    return {f"prob_over_{t}": (mc_preds > t).mean(axis=0) for t in thresholds}


# ---------------------------------------------------------------------------
# Feature engineering (all leakage-safe / pre-game)
# ---------------------------------------------------------------------------
def build_pitcher_game_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (game, pitcher) with same-game raw stats (used only to derive
    LAGGED rolling features later) plus the identity/context columns.
    """
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Batting team = the side at the plate in that half-inning.
    df["batting_team"] = np.where(df["inning_topbot"] == "Top", df["away_team"], df["home_team"])
    df["pitching_team"] = np.where(df["inning_topbot"] == "Top", df["home_team"], df["away_team"])

    # Pitch-level flags.
    df["is_whiff"] = df["description"].isin(["swinging_strike", "swinging_strike_blocked"])
    df["is_csw"] = df["description"].isin(
        ["swinging_strike", "swinging_strike_blocked", "called_strike"]
    )

    # Last pitch of each at-bat carries the outcome (events).
    df = df.sort_values(["game_pk", "pitcher", "at_bat_number", "pitch_number"])
    ab_last = df.groupby(["game_pk", "pitcher", "at_bat_number"], as_index=False).tail(1).copy()
    ab_last["is_k"] = ab_last["events"].fillna("").str.lower().str.contains("strikeout").astype(int)

    # Strikeouts + batters faced per pitcher-game.
    per_ab = ab_last.groupby(["game_pk", "game_date", "pitcher"], as_index=False).agg(
        k=("is_k", "sum"), bf=("is_k", "size")
    )

    # Pitch-level per pitcher-game aggregates.
    per_pitch = df.groupby(["game_pk", "game_date", "pitcher"], as_index=False).agg(
        pitches=("pitch_type", "size"),
        whiff_rate=("is_whiff", "mean"),
        csw_rate=("is_csw", "mean"),
        velo_mean=("release_speed", "mean"),
        pitching_team=("pitching_team", "first"),
        opponent_team=("batting_team", "first"),
        home_team=("home_team", "first"),
        p_throws=("p_throws", "first"),
    )

    pg = per_ab.merge(per_pitch, on=["game_pk", "game_date", "pitcher"], how="inner")
    pg["is_home"] = (pg["pitching_team"] == pg["home_team"]).astype(int)
    pg["k_rate"] = pg["k"] / pg["bf"].clip(lower=1)
    return pg


def add_pitcher_rolling_features(pg: pd.DataFrame) -> pd.DataFrame:
    """
    Lagged rolling form per pitcher (uses only prior starts via shift(1)).
    Operates on STARTS only so the history is comparable game to game.
    """
    starts = pg[pg["pitches"] >= STARTER_MIN_PITCHES].copy()
    starts = starts.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

    roll_cols = {
        "k": "k", "k_rate": "k_rate", "whiff_rate": "whiff_rate",
        "csw_rate": "csw_rate", "velo_mean": "velo", "bf": "bf",
    }
    g = starts.groupby("pitcher", group_keys=False)

    for src, name in roll_cols.items():
        shifted = g[src].shift(1)  # exclude current game
        for w in ROLL_WINDOWS:
            starts[f"roll{w}_{name}"] = (
                shifted.groupby(starts["pitcher"])
                .rolling(w, min_periods=2)
                .mean()
                .reset_index(level=0, drop=True)
            )
        # Season-to-date (expanding) rate.
        starts[f"std_{name}"] = (
            shifted.groupby(starts["pitcher"]).expanding(min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )

    # Days of rest since previous start.
    starts["prev_date"] = g["game_date"].shift(1)
    starts["days_rest"] = (starts["game_date"] - starts["prev_date"]).dt.days
    starts["days_rest"] = starts["days_rest"].clip(upper=14)

    # Count of prior starts (experience / history sufficiency).
    starts["prior_starts"] = g.cumcount()
    return starts


def add_opponent_features(pg_all: pd.DataFrame, starts: pd.DataFrame) -> pd.DataFrame:
    """
    Opponent lineup strikeout tendency, lagged (season-to-date before the game):
      - opp_k_rate: opposing team's overall batting K% to date.
      - opp_k_rate_hand: opposing team's batting K% to date vs the current
        starter's handedness (L/R).
    Built from ALL pitcher-game rows (every team bats every game).
    """
    # Team-game batting totals (overall).
    tg = pg_all.groupby(["opponent_team", "game_pk", "game_date"], as_index=False).agg(
        k=("k", "sum"), pa=("bf", "sum")
    ).rename(columns={"opponent_team": "batting_team"})
    tg = tg.sort_values(["batting_team", "game_date"]).reset_index(drop=True)
    gb = tg.groupby("batting_team", group_keys=False)
    tg["cum_k"] = gb["k"].apply(lambda s: s.shift(1).cumsum())
    tg["cum_pa"] = gb["pa"].apply(lambda s: s.shift(1).cumsum())
    tg["opp_k_rate"] = tg["cum_k"] / tg["cum_pa"]
    opp_overall = tg[["batting_team", "game_pk", "opp_k_rate"]]

    # Team-game batting totals split by the handedness of the pitcher faced.
    tgh = pg_all.groupby(
        ["opponent_team", "p_throws", "game_pk", "game_date"], as_index=False
    ).agg(k=("k", "sum"), pa=("bf", "sum")).rename(
        columns={"opponent_team": "batting_team", "p_throws": "opp_hand"}
    )
    tgh = tgh.sort_values(["batting_team", "opp_hand", "game_date"]).reset_index(drop=True)
    gbh = tgh.groupby(["batting_team", "opp_hand"], group_keys=False)
    tgh["cum_k"] = gbh["k"].apply(lambda s: s.shift(1).cumsum())
    tgh["cum_pa"] = gbh["pa"].apply(lambda s: s.shift(1).cumsum())
    tgh["opp_k_rate_hand"] = tgh["cum_k"] / tgh["cum_pa"]
    opp_hand = tgh[["batting_team", "opp_hand", "game_pk", "opp_k_rate_hand"]]

    out = starts.merge(
        opp_overall, left_on=["opponent_team", "game_pk"],
        right_on=["batting_team", "game_pk"], how="left",
    ).drop(columns=["batting_team"])
    out = out.merge(
        opp_hand,
        left_on=["opponent_team", "p_throws", "game_pk"],
        right_on=["batting_team", "opp_hand", "game_pk"], how="left",
    ).drop(columns=["batting_team", "opp_hand"])
    return out


def assemble_model_table(df: pd.DataFrame):
    pg_all = build_pitcher_game_table(df)
    starts = add_pitcher_rolling_features(pg_all)
    starts = add_opponent_features(pg_all, starts)

    feature_cols = (
        [f"roll{w}_{n}" for n in ["k", "k_rate", "whiff_rate", "csw_rate", "velo", "bf"]
         for w in ROLL_WINDOWS]
        + ["std_k", "std_k_rate", "std_whiff_rate", "std_csw_rate", "std_velo", "std_bf"]
        + ["days_rest", "is_home", "throws_R", "opp_k_rate", "opp_k_rate_hand"]
    )
    starts["throws_R"] = (starts["p_throws"] == "R").astype(int)

    # Require enough pitcher history for the rolling features to be meaningful.
    starts = starts[starts["prior_starts"] >= MIN_PRIOR_STARTS].copy()
    starts = starts.reset_index(drop=True)
    return starts, feature_cols


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def main():
    cfg = load_config()
    ensure_directories()
    set_seed(cfg["project"]["seed"])

    clean_path = cfg["data"]["clean_path"]
    test_start = pd.to_datetime(cfg["split"]["test_start_date"])
    mc_samples = cfg["bnn"]["mc_samples"]

    print("Loading cleaned pitches...")
    df = pd.read_parquet(clean_path)

    print("Building leakage-safe pre-game features (pitcher form + opponent)...")
    table, feature_cols = assemble_model_table(df)
    print(f"Model table: {table.shape[0]} starter-games, {len(feature_cols)} features.")

    table["game_date"] = pd.to_datetime(table["game_date"])
    train_df = table[table["game_date"] < test_start].copy()
    test_df = table[table["game_date"] >= test_start].copy()
    print(f"Train: {len(train_df)}   Test: {len(test_df)}")

    # Impute remaining NaNs (mostly early-season opponent rates) with TRAIN medians.
    medians = train_df[feature_cols].median(numeric_only=True)
    X_train = train_df[feature_cols].fillna(medians).values.astype(np.float32)
    X_test = test_df[feature_cols].fillna(medians).values.astype(np.float32)
    y_train = train_df["k"].values.astype(np.float32)
    y_test = test_df["k"].values.astype(np.float32)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=cfg["nn"]["batch_size"], shuffle=True,
    )

    model = MCDropoutRegressor(
        input_dim=X_train_s.shape[1],
        hidden_sizes=tuple(cfg["nn"]["hidden_sizes"]),
        dropout=cfg["nn"]["dropout"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["nn"]["lr"], weight_decay=cfg["nn"]["weight_decay"]
    )
    loss_fn = nn.MSELoss()

    print("Training BNN v2...")
    model.train()
    for epoch in range(cfg["nn"]["epochs"]):
        losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:02d}/{cfg['nn']['epochs']} loss={np.mean(losses):.4f}")

    mc_preds, pred_mean, pred_std = mc_predict(model, X_test_t, n_samples=mc_samples)
    prob_dict = compute_probabilities(mc_preds, THRESHOLDS)

    # --- Metrics + honest baselines ---------------------------------------
    mae = mean_absolute_error(y_test, pred_mean)
    r = rmse(y_test, pred_mean)

    # Baseline A: predict the pitcher's own recent-average strikeouts (roll5_k).
    base_recent = test_df["roll5_k"].fillna(medians["roll5_k"]).values
    mae_recent = mean_absolute_error(y_test, base_recent)
    rmse_recent = rmse(y_test, base_recent)
    # Baseline B: predict the global training mean.
    base_mean = np.full_like(y_test, y_train.mean())
    mae_mean = mean_absolute_error(y_test, base_mean)

    print("\n[BNN v2 RESULTS]  (test = games on/after", cfg["split"]["test_start_date"], ")")
    print(f"  Model            MAE={mae:.3f}  RMSE={r:.3f}")
    print(f"  Recent-avg base  MAE={mae_recent:.3f}  RMSE={rmse_recent:.3f}")
    print(f"  Global-mean base MAE={mae_mean:.3f}")
    print(f"  Mean predictive std={pred_std.mean():.3f}")

    # Over/under accuracy vs the recent-average baseline.
    print("\n  Over/Under accuracy (model vs recent-avg baseline):")
    for t in THRESHOLDS:
        yt = (y_test > t).astype(int)
        acc_m = ((pred_mean > t).astype(int) == yt).mean()
        acc_b = ((base_recent > t).astype(int) == yt).mean()
        acc_p = ((prob_dict[f"prob_over_{t}"] >= 0.5).astype(int) == yt).mean()
        print(f"    line {t}:  model={acc_m:.3f}  prob={acc_p:.3f}  baseline={acc_b:.3f}")

    # --- Save artifacts (all _v2, nothing overwritten) --------------------
    torch.save(model.state_dict(), "models/bnn_v2_model.pt")
    joblib.dump(scaler, "models/bnn_v2_scaler.pkl")
    with open("models/bnn_v2_features.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    with open("models/bnn_v2_impute.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in medians.items()}, f, indent=2)

    pred_out = test_df[["game_pk", "game_date", "pitcher", "opponent_team", "k"]].copy()
    pred_out = pred_out.rename(columns={"k": "strikeouts"})
    pred_out["pred_mean"] = pred_mean
    pred_out["pred_std"] = pred_std
    for t in THRESHOLDS:
        pred_out[f"prob_over_{t}"] = prob_dict[f"prob_over_{t}"]
        pred_out[f"prob_under_{t}"] = 1.0 - prob_dict[f"prob_over_{t}"]
    pred_out.to_csv("reports/bnn_v2_predictions.csv", index=False)

    pd.DataFrame([{
        "model": "bnn_v2_pregame", "n_test": len(test_df),
        "mae": mae, "rmse": r,
        "mae_recent_avg_baseline": mae_recent,
        "mae_global_mean_baseline": mae_mean,
        "mean_predictive_std": float(pred_std.mean()),
    }]).to_csv("reports/metrics_bnn_v2.csv", index=False)

    print("\nSaved: models/bnn_v2_model.pt, bnn_v2_scaler.pkl, bnn_v2_features.json,")
    print("       bnn_v2_impute.json, reports/bnn_v2_predictions.csv, reports/metrics_bnn_v2.csv")


if __name__ == "__main__":
    main()
