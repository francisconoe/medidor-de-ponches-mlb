import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

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

    print("Train rows:", len(train_df), "Test rows:", len(test_df))

    y_train = train_df["strikeouts"].values
    y_test = test_df["strikeouts"].values

    drop_cols = ["game_pk", "game_date", "game_year", "pitcher", "strikeouts"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].fillna(0.0).values
    X_test = test_df[feature_cols].fillna(0.0).values

    # Mean baseline
    mean_pred = np.full_like(y_test, y_train.mean(), dtype=float)

    # Linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    results = [
        ("mean_baseline", mean_absolute_error(y_test, mean_pred), rmse(y_test, mean_pred)),
        ("linear_regression", mean_absolute_error(y_test, lr_pred), rmse(y_test, lr_pred)),
    ]

    print("\n[BASELINE RESULTS]")
    for name, mae, r in results:
        print(f"{name:18s}  MAE={mae:.3f}  RMSE={r:.3f}")

    # Save metrics
    metrics_df = pd.DataFrame(results, columns=["model", "mae", "rmse"])
    metrics_df.to_csv("reports/metrics_baselines.csv", index=False)

    # Save predictions for evaluation
    pred_df = test_df[
        ["game_pk", "game_date", "game_year", "pitcher", "strikeouts"]
    ].copy()

    pred_df["pred_mean"] = mean_pred
    pred_df["pred_linear"] = lr_pred

    pred_df.to_csv("reports/baseline_predictions.csv", index=False)

    print("\nSaved metrics to reports/metrics_baselines.csv")
    print("Saved predictions to reports/baseline_predictions.csv")


if __name__ == "__main__":
    main()