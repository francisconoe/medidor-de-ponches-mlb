import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils import load_config, ensure_directories


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def main():
    cfg = load_config()
    ensure_directories()

    data_path = cfg["dataset"]["out_path"]
    test_start = pd.to_datetime(cfg["split"]["test_start_date"])

    df = pd.read_parquet(data_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    train_df = df[df["game_date"] < test_start].copy()
    test_df = df[df["game_date"] >= test_start].copy()

    y_train = train_df["strikeouts"].values
    y_test = test_df["strikeouts"].values

    drop_cols = ["game_pk", "game_date", "game_year", "pitcher", "strikeouts"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].fillna(0)
    X_test = test_df[feature_cols].fillna(0)

    model = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=cfg["project"]["seed"],
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r = rmse(y_test, preds)

    print("\n[XGBOOST RESULTS]")
    print(f"MAE = {mae:.3f}")
    print(f"RMSE = {r:.3f}")

    # save model
    model.save_model("models/xgboost_model.json")

    # save feature order
    with open("models/xgboost_features.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    # save predictions with identifiers
    pred_df = test_df[["game_pk", "game_date", "game_year", "pitcher", "strikeouts"]].copy()
    pred_df["pred_xgb"] = preds
    pred_df.to_csv("reports/xgb_predictions.csv", index=False)

    # save metrics
    metrics_df = pd.DataFrame([
        {"model": "xgboost", "mae": mae, "rmse": r}
    ])
    metrics_df.to_csv("reports/metrics_xgboost.csv", index=False)

    print("Saved model to models/xgboost_model.json")
    print("Saved feature list to models/xgboost_features.json")
    print("Saved predictions to reports/xgb_predictions.csv")
    print("Saved metrics to reports/metrics_xgboost.csv")


if __name__ == "__main__":
    main()