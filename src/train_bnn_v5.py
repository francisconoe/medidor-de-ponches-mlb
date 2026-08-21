"""
BNN v5 - identical architecture/pipeline to v4, with two added features
(vs_opp_career_k, opp_whiff_rate) motivated by the Aug backtest of live losses.

Trains on the same split as v4 (fit <= 2025-06-30, calibrate on late 2025,
evaluate on the fully held-out 2026 season) so the holdout numbers are directly
comparable to reports/metrics_bnn_v4.csv. Saves bnn_v5_* artifacts; v4 untouched.

    python -m src.train_bnn_v5
"""

import json
import glob

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

from src.utils import load_config, ensure_directories, set_seed
from src.train_bnn_v3 import NBDropoutNet, nb_nll, mc_nb_predict, THRESHOLDS
from src.train_bnn_v4 import (BetaCalibrator, ens_nb_tail, coverage_at_acc,
                              sel_acc, ece, CAL_LINES, FIT_END, TEST_START,
                              MC_SAMPLES, PIPE_COLS, load_all_pitches)
from src.features_v5 import assemble_training_table_v5


def main():
    cfg = load_config()
    ensure_directories()
    set_seed(cfg["project"]["seed"])

    print("Loading pitch data (2022 -> present)...")
    df = load_all_pitches(cfg)

    print("Building v5 feature table...")
    table, feats, park = assemble_training_table_v5(df, pd.Timestamp(TEST_START))
    table["game_date"] = pd.to_datetime(table["game_date"])
    print(f"Table: {len(table)} starter-games, {len(feats)} features "
          f"(v4 had {len(feats)-2}).")

    fit_df = table[table["game_date"] <= FIT_END].copy()
    val_df = table[(table["game_date"] > FIT_END) & (table["game_date"] < TEST_START)].copy()
    test_df = table[table["game_date"] >= TEST_START].copy()
    print(f"Fit: {len(fit_df)}   Val: {len(val_df)}   Test 2026: {len(test_df)}")

    medians = fit_df[feats].median(numeric_only=True)

    def prep(d):
        return d[feats].fillna(medians).values.astype(np.float32)

    scaler = StandardScaler().fit(prep(fit_df))
    Xf, Xv, Xt = scaler.transform(prep(fit_df)), scaler.transform(prep(val_df)), scaler.transform(prep(test_df))
    yf = fit_df["k"].values.astype(np.float32)
    yv = val_df["k"].values.astype(float)
    yt = test_df["k"].values.astype(float)

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

    mu_v, r_v = mc_nb_predict(model, torch.tensor(Xv), MC_SAMPLES)
    nn_v = mu_v.mean(0)
    xgb_v = xgbm.predict(val_df[feats].fillna(medians))
    best_w, best_mae = 0.5, 1e9
    for wv in np.linspace(0, 1, 11):
        mae = mean_absolute_error(yv, wv * nn_v + (1 - wv) * xgb_v)
        if mae < best_mae:
            best_mae, best_w = mae, wv
    print(f"Ensemble weight (NN share) = {best_w:.1f}  [val MAE {best_mae:.3f}]")

    blend_v = best_w * nn_v + (1 - best_w) * xgb_v
    calibrators = {}
    for line in CAL_LINES:
        raw = ens_nb_tail(mu_v, r_v, blend_v, line)
        calibrators[line] = BetaCalibrator().fit(raw, (yv > line).astype(int))

    mu_t, r_t = mc_nb_predict(model, torch.tensor(Xt), MC_SAMPLES)
    xgb_t = xgbm.predict(test_df[feats].fillna(medians))
    blend_t = best_w * mu_t.mean(0) + (1 - best_w) * xgb_t

    mae = mean_absolute_error(yt, blend_t)
    base_recent = test_df["roll5_k"].fillna(medians["roll5_k"]).values
    print("\n[BNN v5 RESULTS - 2026 held-out season]")
    print(f"  Point MAE={mae:.3f}  (recent-avg baseline {mean_absolute_error(yt, base_recent):.3f})")
    print("  (v4 holdout MAE was 1.700)")

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

    print("\n  (v4 holdout for reference:")
    print("   3.5 65% 70% 0.011  8% 85% 88% 91% | 4.5 52% 66% 0.012  6% 82% 87% 91%")
    print("   5.5 67% 71% 0.020 22% 88% 94% 96% | 6.5 78% 80% 0.016 54% 93% 96% 97%)")

    # feature importance of the two new features (XGB gain)
    imp = pd.Series(xgbm.feature_importances_, index=feats).sort_values(ascending=False)
    print(f"\n  New-feature XGB importance rank (of {len(feats)}):")
    for f in ["vs_opp_career_k", "opp_whiff_rate", "roll5_brk_whiff", "opp_break_whiff"]:
        if f in imp.index:
            print(f"    {f:18s} rank {list(imp.index).index(f)+1:>2}  gain={imp[f]:.4f}")

    torch.save(model.state_dict(), "models/bnn_v5_model.pt")
    joblib.dump(scaler, "models/bnn_v5_scaler.pkl")
    joblib.dump(calibrators, "models/bnn_v5_calibrators.pkl")
    xgbm.save_model("models/bnn_v5_xgb.json")
    with open("models/bnn_v5_features.json", "w", encoding="utf-8") as f:
        json.dump(feats, f, indent=2)
    with open("models/bnn_v5_impute.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in medians.items()}, f, indent=2)
    with open("models/bnn_v5_park.json", "w", encoding="utf-8") as f:
        json.dump(park, f, indent=2)
    with open("models/bnn_v5_meta.json", "w", encoding="utf-8") as f:
        json.dump({"ensemble_nn_weight": float(best_w),
                   "hidden_sizes": list(cfg["nn"]["hidden_sizes"]),
                   "dropout": cfg["nn"]["dropout"], "mc_samples": MC_SAMPLES,
                   "thresholds": THRESHOLDS, "fit_end": FIT_END,
                   "test_start": TEST_START}, f, indent=2)

    pd.DataFrame([{"model": "bnn_v5_vsopp_whiff", "n_test_2026": len(test_df),
                   "mae": mae, "mae_recent_avg_baseline": mean_absolute_error(yt, base_recent),
                   "ensemble_nn_weight": float(best_w)}]).to_csv("reports/metrics_bnn_v5.csv", index=False)
    print("\nSaved v5 artifacts (models/bnn_v5_*, reports/metrics_bnn_v5.csv). v4 untouched.")


if __name__ == "__main__":
    main()
