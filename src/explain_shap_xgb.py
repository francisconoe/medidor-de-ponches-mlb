import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb

from src.utils import load_config, ensure_directories


def main():
    cfg = load_config()
    ensure_directories()

    data_path = cfg["dataset"]["out_path"]
    test_start = pd.to_datetime(cfg["split"]["test_start_date"])

    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model table...")
    df = pd.read_parquet(data_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    print("Loading XGBoost feature list...")
    with open("models/xgboost_features.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    print("Loading XGBoost model...")
    model = xgb.XGBRegressor()
    model.load_model("models/xgboost_model.json")

    # Explain test set only
    test_df = df[df["game_date"] >= test_start].copy()
    X_test = test_df[feature_cols].fillna(0.0)

    print("Test shape for SHAP:", X_test.shape)

    # Sample for speed and cleaner plots
    sample_size = min(2000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=cfg["project"]["seed"])
    print("Sample shape used for SHAP:", X_sample.shape)

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 1) Bar plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_xgb_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2) Beeswarm summary plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_xgb_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3) Save feature importance table
    mean_abs_shap = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    mean_abs_shap.to_csv("reports/shap_xgb_feature_importance.csv", index=False)

    print("\nTop 10 SHAP features:")
    print(mean_abs_shap.head(10))

    # 4) Dependence plots for top 3 features
    top_features = mean_abs_shap["feature"].head(3).tolist()

    for feature in top_features:
        plt.figure()
        shap.dependence_plot(
            feature,
            shap_values,
            X_sample,
            show=False
        )
        plt.tight_layout()
        safe_name = feature.replace("/", "_")
        plt.savefig(figures_dir / f"shap_dependence_{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("\nSaved:")
    print(" - reports/figures/shap_xgb_bar.png")
    print(" - reports/figures/shap_xgb_summary.png")
    print(" - reports/shap_xgb_feature_importance.csv")
    for feature in top_features:
        safe_name = feature.replace("/", "_")
        print(f" - reports/figures/shap_dependence_{safe_name}.png")


if __name__ == "__main__":
    main()