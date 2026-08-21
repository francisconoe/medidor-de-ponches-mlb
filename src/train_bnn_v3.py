"""
BNN v3 - a calibrated, count-aware Bayesian strikeout model.

The v2 model's point accuracy was already near the irreducible noise floor, but
its PROBABILITIES were badly miscalibrated: an MSE point-regressor with MC-Dropout
has no observation-noise model, so its predictive spread (std ~0.45) was far too
narrow for a count whose true noise SD is ~2.0 -> it said 93% when it meant 56%.

v3 fixes this at the source by modelling strikeouts as a proper count:

  1. Negative-Binomial likelihood. The network outputs (mu, r) - a mean and a
     dispersion - and is trained by NB negative log-likelihood. NB models the
     real over-dispersed count noise, so P(K > line) is principled, not a
     Gaussian afterthought. MC-Dropout still supplies epistemic uncertainty on
     top, giving a full predictive distribution.
  2. Isotonic post-hoc calibration. Fit on a held-out validation slice (by date)
     so the reported probabilities match reality on unseen data.
  3. XGBoost-Poisson ensemble for the point mean (the study showed a gradient
     boosted count model edges the NN on MAE); the calibrated NB head owns the
     probabilities.

Features come from features_v3 (batter-level lineup K%, workload, park factor).
Trains, evaluates honestly (MAE vs baselines, calibration table, and the
coverage-accuracy curve that defines the selective-prediction path), and saves
everything with a _v3 suffix. No existing file is modified.

Run from the strikeout_predictor project root:
    python -m src.train_bnn_v3
"""

import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import nbinom
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

from src.utils import load_config, ensure_directories, set_seed
from src.features_v3 import assemble_training_table

THRESHOLDS = [3.5, 4.5, 5.5, 6.5]
VAL_FRACTION = 0.15          # last 15% of train (by date) used for calibration
MC_SAMPLES_DEFAULT = 200


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------------------------------------------------------------------------
# Negative-Binomial MC-Dropout network
# ---------------------------------------------------------------------------
class NBDropoutNet(nn.Module):
    """Shared trunk with dropout, two positive heads: mean (mu) and dispersion (r)."""

    def __init__(self, input_dim, hidden_sizes=(256, 128), dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.mu_head = nn.Linear(hidden_sizes[1], 1)
        self.r_head = nn.Linear(hidden_sizes[1], 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        mu = torch.nn.functional.softplus(self.mu_head(x)).squeeze(1) + 1e-4
        r = torch.nn.functional.softplus(self.r_head(x)).squeeze(1) + 1e-2
        r = torch.clamp(r, max=1e4)
        return mu, r


def nb_nll(y, mu, r):
    """Negative-Binomial negative log-likelihood (mean-dispersion parameterisation)."""
    return -(
        torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0)
        + r * (torch.log(r) - torch.log(r + mu))
        + y * (torch.log(mu) - torch.log(r + mu))
    ).mean()


def mc_nb_predict(model, X_tensor, n_samples=200):
    """MC-Dropout sampling -> stacked mu and r arrays of shape [n_samples, n]."""
    model.train()  # keep dropout ON
    mus, rs = [], []
    with torch.no_grad():
        for _ in range(n_samples):
            mu, r = model(X_tensor)
            mus.append(mu.cpu().numpy())
            rs.append(r.cpu().numpy())
    return np.stack(mus, 0), np.stack(rs, 0)


def nb_prob_over(mu_s, r_s, line):
    """Mean over MC samples of P(Y > line) under NB(mu, r). line is x.5 -> Y >= ceil(line)."""
    m = int(np.ceil(line))                     # e.g. line 5.5 -> need Y >= 6
    p = r_s / (r_s + mu_s)                      # scipy nbinom(n=r, p) has mean mu
    sf = nbinom.sf(m - 1, r_s, p)              # P(Y >= m) = sf(m-1)
    return sf.mean(axis=0)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def main():
    cfg = load_config()
    ensure_directories()
    set_seed(cfg["project"]["seed"])

    test_start = pd.to_datetime(cfg["split"]["test_start_date"])
    mc_samples = cfg.get("bnn", {}).get("mc_samples", MC_SAMPLES_DEFAULT) or MC_SAMPLES_DEFAULT
    mc_samples = max(mc_samples, MC_SAMPLES_DEFAULT)

    print("Loading pitches + building v3 features...")
    df = pd.read_parquet(cfg["data"]["clean_path"])
    table, feature_cols, park = assemble_training_table(df, test_start)
    table["game_date"] = pd.to_datetime(table["game_date"])
    print(f"Table: {table.shape[0]} starter-games, {len(feature_cols)} features.")

    test_df = table[table["game_date"] >= test_start].copy()
    train_all = table[table["game_date"] < test_start].sort_values("game_date").copy()

    # Validation slice (last VAL_FRACTION by date) for calibration + ensemble weight.
    cut = train_all["game_date"].quantile(1 - VAL_FRACTION)
    fit_df = train_all[train_all["game_date"] <= cut].copy()
    val_df = train_all[train_all["game_date"] > cut].copy()
    print(f"Fit: {len(fit_df)}   Val: {len(val_df)}   Test: {len(test_df)}")

    medians = fit_df[feature_cols].median(numeric_only=True)

    def prep(d):
        return d[feature_cols].fillna(medians).values.astype(np.float32)

    scaler = StandardScaler().fit(prep(fit_df))
    Xf, Xv, Xt = scaler.transform(prep(fit_df)), scaler.transform(prep(val_df)), scaler.transform(prep(test_df))
    yf = fit_df["k"].values.astype(np.float32)
    yv = val_df["k"].values.astype(np.float32)
    yt = test_df["k"].values.astype(np.float32)

    # --- NB network -------------------------------------------------------
    model = NBDropoutNet(Xf.shape[1], tuple(cfg["nn"]["hidden_sizes"]), cfg["nn"]["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["nn"]["lr"], weight_decay=cfg["nn"]["weight_decay"])
    loader = DataLoader(
        TensorDataset(torch.tensor(Xf), torch.tensor(yf)),
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

    # --- XGBoost-Poisson (point mean) ------------------------------------
    print("Training XGBoost-Poisson...")
    xgbm = xgb.XGBRegressor(
        objective="count:poisson", n_estimators=700, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0)
    xgbm.fit(fit_df[feature_cols].fillna(medians), yf)

    # --- Ensemble weight for the point mean (chosen on val) --------------
    mu_val, r_val = mc_nb_predict(model, torch.tensor(Xv), mc_samples)
    nn_val = mu_val.mean(0)
    xgb_val = xgbm.predict(val_df[feature_cols].fillna(medians))
    best_w, best_mae = 0.5, 1e9
    for w in np.linspace(0, 1, 11):
        mae = mean_absolute_error(yv, w * nn_val + (1 - w) * xgb_val)
        if mae < best_mae:
            best_mae, best_w = mae, w
    print(f"Ensemble weight (NN share) = {best_w:.1f}  [val MAE {best_mae:.3f}]")

    # --- Isotonic calibration per line (fit on val) ----------------------
    calibrators = {}
    for line in THRESHOLDS:
        raw = nb_prob_over(mu_val, r_val, line)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(raw, (yv > line).astype(int))
        calibrators[line] = iso

    # --- Evaluate on test -------------------------------------------------
    mu_t, r_t = mc_nb_predict(model, torch.tensor(Xt), mc_samples)
    nn_t = mu_t.mean(0)
    xgb_t = xgbm.predict(test_df[feature_cols].fillna(medians))
    pred_mean = best_w * nn_t + (1 - best_w) * xgb_t

    raw_probs = {l: nb_prob_over(mu_t, r_t, l) for l in THRESHOLDS}
    cal_probs = {l: calibrators[l].predict(raw_probs[l]) for l in THRESHOLDS}

    mae = mean_absolute_error(yt, pred_mean)
    r_ = rmse(yt, pred_mean)
    base_recent = test_df["roll5_k"].fillna(medians["roll5_k"]).values
    mae_recent = mean_absolute_error(yt, base_recent)

    print("\n[BNN v3 RESULTS]")
    print(f"  Model MAE={mae:.3f}  RMSE={r_:.3f}   | recent-avg baseline MAE={mae_recent:.3f}")

    print("\n  Calibration of P(over 5.5) after isotonic:  bin  n  pred  emp")
    p55 = cal_probs[5.5]; over55 = (yt > 5.5).astype(int)
    for lo in np.arange(0, 1, 0.2):
        m = (p55 >= lo) & (p55 < lo + 0.2)
        if m.sum():
            print(f"     {lo:.1f}-{lo+0.2:.1f}  {m.sum():4d}  {p55[m].mean():.2f}  {over55[m].mean():.2f}")

    print("\n  Coverage-accuracy (selective prediction) using calibrated confidence:")
    print(f"  {'line':>5} {'100%':>6} {'50%':>6} {'25%':>6} {'10%':>6} {'5%':>6}")
    for line in THRESHOLDS:
        pc = cal_probs[line]
        correct = ((pc >= 0.5).astype(int) == (yt > line).astype(int)).astype(float)
        order = np.argsort(-np.abs(pc - 0.5))
        c = correct[order]
        cells = [f"{c[:max(1, int(len(c)*cov))].mean()*100:.0f}%" for cov in [1, .5, .25, .10, .05]]
        print(f"  {line:>5} {cells[0]:>6} {cells[1]:>6} {cells[2]:>6} {cells[3]:>6} {cells[4]:>6}")

    # --- Save artifacts ---------------------------------------------------
    torch.save(model.state_dict(), "models/bnn_v3_model.pt")
    joblib.dump(scaler, "models/bnn_v3_scaler.pkl")
    joblib.dump(calibrators, "models/bnn_v3_calibrators.pkl")
    xgbm.save_model("models/bnn_v3_xgb.json")
    with open("models/bnn_v3_features.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    with open("models/bnn_v3_impute.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in medians.items()}, f, indent=2)
    with open("models/bnn_v3_park.json", "w", encoding="utf-8") as f:
        json.dump(park, f, indent=2)
    with open("models/bnn_v3_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "ensemble_nn_weight": float(best_w),
            "hidden_sizes": list(cfg["nn"]["hidden_sizes"]),
            "dropout": cfg["nn"]["dropout"],
            "mc_samples": int(mc_samples),
            "thresholds": THRESHOLDS,
        }, f, indent=2)

    out = test_df[["game_pk", "game_date", "pitcher", "opponent_team", "k"]].copy()
    out = out.rename(columns={"k": "strikeouts"})
    out["pred_mean"] = np.round(pred_mean, 3)
    for line in THRESHOLDS:
        out[f"prob_over_{line}"] = np.round(cal_probs[line], 4)
    out.to_csv("reports/bnn_v3_predictions.csv", index=False)

    pd.DataFrame([{
        "model": "bnn_v3_nb_calibrated", "n_test": len(test_df),
        "mae": mae, "rmse": r_, "mae_recent_avg_baseline": mae_recent,
        "ensemble_nn_weight": float(best_w),
    }]).to_csv("reports/metrics_bnn_v3.csv", index=False)

    print("\nSaved v3 artifacts (models/bnn_v3_*, reports/bnn_v3_*).")


if __name__ == "__main__":
    main()
