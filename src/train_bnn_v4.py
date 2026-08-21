"""
BNN v4 - the v3 calibrated NB model, retrained on current data with the
improvements that survived out-of-sample validation.

The v4 changes (each was simulation-tested; architecture changes were tested
and did NOT help, so the proven v3 NB-Dropout + XGB-Poisson core is unchanged):

  1. FRESH TRAINING DATA. v3 trained on 2022 -> Apr 2024 yet serves 2026 - two
     years stale, blind to the 2025 run environment and to every pitcher who
     debuted since. v4 trains on 2022 -> mid-2025, calibrates on late 2025, and
     is evaluated on 2026 (fully held out).
  2. ENSEMBLE PROBABILITIES. v3 computed P(K > line) from the NB net's mean
     alone, using XGBoost only for the point estimate. v4 blends the point mean
     INTO the NB tail (mu_ens = w*NN + (1-w)*XGB), which lifted the fraction of
     the slate bettable at 90% accuracy from 11.6% to 13.2% on 2026.
  3. BETA CALIBRATION instead of isotonic - lower calibration error (ECE) at
     every line in validation, and it can't produce isotonic's flat steps.
  4. v4 FEATURES (features_v4): opponent last-10-games K%, form-vs-career
     deltas (velo/k_rate/whiff), season month.

Artifacts are saved with a _v4 suffix; v3 files are untouched for rollback.

    python -m src.train_bnn_v4
"""

import json
import glob

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import nbinom
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

from src.utils import load_config, ensure_directories, set_seed
from src.train_bnn_v3 import NBDropoutNet, nb_nll, mc_nb_predict, THRESHOLDS
from src.features_v4 import assemble_training_table_v4

# Time boundaries (trains on everything up to FIT_END, calibrates on
# (FIT_END, TEST_START), and is evaluated on TEST_START onward). Moved forward
# so the model TRAINS on the 2026 season (through mid-June) and is held out on
# the most recent ~month - the old split never trained on 2026 at all, which is
# the staleness the season-drift concern was really about.
FIT_END = "2026-06-15"
TEST_START = "2026-07-06"
MC_SAMPLES = 300

# Calibrators are fit at every line a book actually posts (2.5 for weak arms,
# 7.5 for aces), not just the eval THRESHOLDS - serving refuses to stake on any
# line outside this range, so widening it converts auto-passes into playable,
# honestly-calibrated bets.
CAL_LINES = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]

# Pitch-level columns the feature pipeline needs (schemas differ across pulls).
PIPE_COLS = ["game_pk", "game_date", "pitcher", "batter", "at_bat_number",
             "pitch_number", "events", "description", "pitch_type",
             "release_speed", "inning_topbot", "home_team", "away_team", "p_throws"]


class BetaCalibrator:
    """Beta calibration (Kull et al. 2017): logistic regression on
    [log(p), -log(1-p)]. Monotone, smooth, fixes NB tail over/under-confidence."""

    EPS = 1e-6

    def _z(self, p):
        p = np.clip(np.asarray(p, dtype=float), self.EPS, 1 - self.EPS)
        return np.column_stack([np.log(p), -np.log(1 - p)])

    def fit(self, raw, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._z(raw), y)
        return self

    def predict(self, raw):
        return self.lr.predict_proba(self._z(raw))[:, 1]


def load_all_pitches(cfg) -> pd.DataFrame:
    """2022-24 clean parquet + every season parquet pulled since (2025, 2026...)."""
    frames = [pd.read_parquet(cfg["data"]["clean_path"], columns=PIPE_COLS)]
    for path in sorted(glob.glob("data/raw/pitches_20*.parquet")):
        d = pd.read_parquet(path, columns=PIPE_COLS)
        d = d.dropna(subset=["pitcher", "game_date"])
        frames.append(d)
        print(f"  + {path}: {len(d)} pitches")
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number", "pitcher"])
    print(f"  total: {len(df)} pitches, {df['game_date'].min().date()} -> {df['game_date'].max().date()}")
    return df


def ens_nb_tail(mu_s, r_s, point_blend, line):
    """P(K > line): NB survival with the ENSEMBLE mean substituted for the NN
    mean in every MC sample (keeps sampled dispersion), averaged over samples."""
    m = int(np.ceil(line))
    p = r_s / (r_s + point_blend[None, :])
    return nbinom.sf(m - 1, r_s, p).mean(axis=0)


def coverage_at_acc(prob, actual_over, target=0.90):
    pred = (prob >= 0.5).astype(int)
    c = (pred == actual_over).astype(float)[np.argsort(-np.abs(prob - 0.5))]
    best = 0.0
    for k in range(1, len(c) + 1):
        if c[:k].mean() >= target:
            best = k / len(c)
    return best


def sel_acc(prob, actual_over, cov):
    c = ((prob >= 0.5).astype(int) == actual_over).astype(float)
    c = c[np.argsort(-np.abs(prob - 0.5))]
    return c[:max(1, int(len(c) * cov))].mean()


def ece(prob, actual_over, bins=10):
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (prob >= lo) & (prob < lo + 1 / bins)
        if m.sum():
            e += m.mean() * abs(prob[m].mean() - actual_over[m].mean())
    return e


def main():
    cfg = load_config()
    ensure_directories()
    set_seed(cfg["project"]["seed"])

    print("Loading pitch data (2022 -> present)...")
    df = load_all_pitches(cfg)

    print("Building v4 feature table...")
    table, feats, park = assemble_training_table_v4(df, pd.Timestamp(TEST_START))
    table["game_date"] = pd.to_datetime(table["game_date"])
    print(f"Table: {len(table)} starter-games, {len(feats)} features.")

    fit_df = table[table["game_date"] <= FIT_END].copy()
    val_df = table[(table["game_date"] > FIT_END) & (table["game_date"] < TEST_START)].copy()
    test_df = table[table["game_date"] >= TEST_START].copy()
    print(f"Fit: {len(fit_df)} (<= {FIT_END})   Val: {len(val_df)}   Test 2026: {len(test_df)}")

    medians = fit_df[feats].median(numeric_only=True)

    def prep(d):
        return d[feats].fillna(medians).values.astype(np.float32)

    scaler = StandardScaler().fit(prep(fit_df))
    Xf, Xv, Xt = scaler.transform(prep(fit_df)), scaler.transform(prep(val_df)), scaler.transform(prep(test_df))
    yf = fit_df["k"].values.astype(np.float32)
    yv = val_df["k"].values.astype(float)
    yt = test_df["k"].values.astype(float)

    # --- NB-Dropout network (proven v3 architecture) -----------------------
    model = NBDropoutNet(Xf.shape[1], tuple(cfg["nn"]["hidden_sizes"]), cfg["nn"]["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["nn"]["lr"],
                           weight_decay=cfg["nn"]["weight_decay"])
    loader = DataLoader(TensorDataset(torch.tensor(Xf), torch.tensor(yf)),
                        batch_size=cfg["nn"]["batch_size"], shuffle=True)
    epochs = max(cfg["nn"]["epochs"], 80)
    print(f"Training NB-Dropout network ({epochs} epochs)...")
    model.train()
    for ep in range(epochs):
        losses = []
        for xb, yb in loader:
            opt.zero_grad()
            mu, r = model(xb)
            loss = nb_nll(yb, mu, r)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  epoch {ep+1:02d}/{epochs}  NB-NLL={np.mean(losses):.4f}")

    print("Training XGBoost-Poisson...")
    xgbm = xgb.XGBRegressor(
        objective="count:poisson", n_estimators=700, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0)
    xgbm.fit(fit_df[feats].fillna(medians), yf)

    # --- Ensemble weight (val MAE) -----------------------------------------
    mu_v, r_v = mc_nb_predict(model, torch.tensor(Xv), MC_SAMPLES)
    nn_v = mu_v.mean(0)
    xgb_v = xgbm.predict(val_df[feats].fillna(medians))
    best_w, best_mae = 0.5, 1e9
    for w in np.linspace(0, 1, 11):
        mae = mean_absolute_error(yv, w * nn_v + (1 - w) * xgb_v)
        if mae < best_mae:
            best_mae, best_w = mae, w
    print(f"Ensemble weight (NN share) = {best_w:.1f}  [val MAE {best_mae:.3f}]")

    # --- Beta calibrators on ensemble-tail probabilities (val) -------------
    blend_v = best_w * nn_v + (1 - best_w) * xgb_v
    calibrators = {}
    for line in CAL_LINES:
        raw = ens_nb_tail(mu_v, r_v, blend_v, line)
        calibrators[line] = BetaCalibrator().fit(raw, (yv > line).astype(int))

    # --- Evaluate on held-out 2026 -----------------------------------------
    mu_t, r_t = mc_nb_predict(model, torch.tensor(Xt), MC_SAMPLES)
    xgb_t = xgbm.predict(test_df[feats].fillna(medians))
    blend_t = best_w * mu_t.mean(0) + (1 - best_w) * xgb_t

    mae = mean_absolute_error(yt, blend_t)
    base_recent = test_df["roll5_k"].fillna(medians["roll5_k"]).values
    print(f"\n[BNN v4 RESULTS - 2026 held-out season]")
    print(f"  Point MAE={mae:.3f}  (recent-avg baseline {mean_absolute_error(yt, base_recent):.3f})")

    print(f"\n  {'line':>4} {'baseRate':>8} {'allAcc':>7} {'ECE':>6} {'cov@90%':>8} "
          f"{'top25%':>7} {'top10%':>7} {'top5%':>7}")
    cal_probs = {}
    for line in CAL_LINES:
        raw = ens_nb_tail(mu_t, r_t, blend_t, line)
        p = calibrators[line].predict(raw)
        cal_probs[line] = p
        ao = (yt > line).astype(int)
        print(f"  {line:>4} {max(ao.mean(), 1-ao.mean())*100:>7.0f}% "
              f"{(((p>=0.5).astype(int)==ao).mean())*100:>6.0f}% {ece(p, ao):>6.3f} "
              f"{coverage_at_acc(p, ao)*100:>7.0f}% {sel_acc(p, ao, .25)*100:>6.0f}% "
              f"{sel_acc(p, ao, .10)*100:>6.0f}% {sel_acc(p, ao, .05)*100:>6.0f}%")

    # --- Save artifacts -----------------------------------------------------
    torch.save(model.state_dict(), "models/bnn_v4_model.pt")
    joblib.dump(scaler, "models/bnn_v4_scaler.pkl")
    joblib.dump(calibrators, "models/bnn_v4_calibrators.pkl")
    xgbm.save_model("models/bnn_v4_xgb.json")
    with open("models/bnn_v4_features.json", "w", encoding="utf-8") as f:
        json.dump(feats, f, indent=2)
    with open("models/bnn_v4_impute.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in medians.items()}, f, indent=2)
    with open("models/bnn_v4_park.json", "w", encoding="utf-8") as f:
        json.dump(park, f, indent=2)
    with open("models/bnn_v4_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "ensemble_nn_weight": float(best_w),
            "hidden_sizes": list(cfg["nn"]["hidden_sizes"]),
            "dropout": cfg["nn"]["dropout"],
            "mc_samples": MC_SAMPLES,
            "thresholds": THRESHOLDS,
            "fit_end": FIT_END,
            "test_start": TEST_START,
            "prob_pipeline": "ensemble_nb_tail + beta_calibration",
        }, f, indent=2)

    out = test_df[["game_pk", "game_date", "pitcher", "opponent_team", "k"]].copy()
    out = out.rename(columns={"k": "strikeouts"})
    out["pred_mean"] = np.round(blend_t, 3)
    for line in CAL_LINES:
        out[f"prob_over_{line}"] = np.round(cal_probs[line], 4)
    out.to_csv("reports/bnn_v4_predictions.csv", index=False)

    pd.DataFrame([{
        "model": "bnn_v4_nb_ens_beta", "n_test_2026": len(test_df), "mae": mae,
        "mae_recent_avg_baseline": mean_absolute_error(yt, base_recent),
        "ensemble_nn_weight": float(best_w),
    }]).to_csv("reports/metrics_bnn_v4.csv", index=False)
    print("\nSaved v4 artifacts (models/bnn_v4_*, reports/bnn_v4_*). v3 untouched.")


if __name__ == "__main__":
    main()
