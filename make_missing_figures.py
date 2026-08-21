from pathlib import Path
import sys
import json
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
print("SCRIPT IS RUNNING")

print("=" * 70)
print("STARTING make_missing_figures.py")
print("Project root:", ROOT)
print("Figures directory:", FIGURES_DIR)
print("=" * 70)


def check_file(path):
    path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    print("Found:", path)
    return path


def save_and_close(filename):
    output_path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", output_path)


# 1. Local SHAP waterfall plot
def make_local_shap_plot():
    print("\n[1/4] Creating local SHAP waterfall plot...")

    import shap
    import xgboost as xgb

    model_table_path = check_file("data/processed/model_table.parquet")
    features_path = check_file("models/xgboost_features.json")
    model_path = check_file("models/xgboost_model.json")

    df = pd.read_parquet(model_table_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    test_df = df[df["game_date"] >= pd.to_datetime("2024-04-01")].copy()

    with open(features_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    X_test = test_df[feature_cols].fillna(0.0)

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(model_path)

    X_sample = X_test.sample(min(2000, len(X_test)), random_state=42)

    print("X_sample shape:", X_sample.shape)

    explainer = shap.Explainer(xgb_model, X_sample)
    shap_values = explainer(X_sample)

    preds = xgb_model.predict(X_sample)
    i = int(np.argmax(preds))

    print("Selected sample index:", i)
    print("Selected predicted strikeouts:", round(float(preds[i]), 3))

    shap.plots.waterfall(shap_values[i], max_display=10, show=False)
    save_and_close("shap_local_example.png")


# 2. Threshold F1 performance plot
def make_threshold_plot():
    print("\n[2/4] Creating threshold F1 performance plot...")

    metrics_path = check_file("reports/over_under_metrics.csv")
    metrics = pd.read_csv(metrics_path)

    print("over_under_metrics.csv columns:")
    print(metrics.columns.tolist())

    # Expected columns:
    # model, threshold, accuracy, precision, recall, f1, decision_type

    if "decision_type" in metrics.columns:
        plot_df = metrics[
            (metrics["decision_type"] == "thresholded_regression")
            & (metrics["model"] != "mean_baseline")
        ].copy()
    else:
        plot_df = metrics[metrics["model"] != "mean_baseline"].copy()

    name_map = {
        "linear_regression": "Linear Regression",
        "xgboost": "XGBoost",
        "neural_network": "Neural Network",
        "bayesian_nn_mean": "Bayesian Neural Network",
        "bnn": "Bayesian Neural Network",
    }

    plot_df["model_label"] = plot_df["model"].map(name_map).fillna(plot_df["model"])

    print("Rows used for threshold plot:", len(plot_df))
    print(plot_df[["model", "threshold", "f1"]].head())

    plt.figure(figsize=(8, 5))

    for model_name in plot_df["model_label"].dropna().unique():
        model_data = plot_df[plot_df["model_label"] == model_name].sort_values("threshold")

        plt.plot(
            model_data["threshold"],
            model_data["f1"],
            marker="o",
            label=model_name
        )

    plt.xlabel("Strikeout line")
    plt.ylabel("F1-score")
    plt.title("Over/under F1-score across strikeout thresholds")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_and_close("threshold_f1_performance.png")


# 3. BNN MC Dropout predictive distribution
def make_bnn_distribution_plot():
    print("\n[3/4] Creating BNN predictive distribution plot...")

    import joblib
    import torch

    sys.path.append(str(ROOT))

    from src.train_bnn import MCDropoutRegressor, mc_predict

    model_table_path = check_file("data/processed/model_table.parquet")
    features_path = check_file("models/bnn_features.json")
    scaler_path = check_file("models/bnn_scaler.pkl")
    model_path = check_file("models/bnn_model.pt")

    df = pd.read_parquet(model_table_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    test_df = df[df["game_date"] >= pd.to_datetime("2024-04-01")].copy()

    with open(features_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    X_test = test_df[feature_cols].fillna(0.0).values

    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    model = MCDropoutRegressor(
        input_dim=len(feature_cols),
        hidden_sizes=(256, 128),
        dropout=0.2
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    mc_preds, pred_mean, pred_std = mc_predict(model, X_test_tensor, n_samples=100)

    print("MC predictions shape:", mc_preds.shape)

    # Choose the example with the highest uncertainty
    i = int(np.argmax(pred_std))
    example_samples = mc_preds[:, i]

    plt.figure(figsize=(7, 5))
    plt.hist(example_samples, bins=20)
    plt.axvline(
        example_samples.mean(),
        linestyle="--",
        label=f"Mean = {example_samples.mean():.2f}"
    )

    plt.xlabel("Predicted strikeouts")
    plt.ylabel("Frequency")
    plt.title("MC Dropout predictive distribution for one pitcher-game")
    plt.legend()

    save_and_close("bnn_predictive_distribution.png")

    print("Selected example index:", i)

    if "strikeouts" in test_df.columns:
        print("Actual strikeouts:", test_df.iloc[i]["strikeouts"])

    print("Predictive mean:", round(float(example_samples.mean()), 3))
    print("Predictive std:", round(float(example_samples.std()), 3))


# 4. BNN calibration curve
def make_calibration_curve():
    print("\n[4/4] Creating BNN calibration curve...")

    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    bnn_path = check_file("reports/bnn_predictions.csv")
    bnn = pd.read_csv(bnn_path)

    print("bnn_predictions.csv columns:")
    print(bnn.columns.tolist())

    MAIN_LINE = 5.5

    possible_prob_cols = [
        f"prob_over_{MAIN_LINE}",
        "prob_over_5.5",
        "prob_over_5_5",
    ]

    prob_col = None
    for col in possible_prob_cols:
        if col in bnn.columns:
            prob_col = col
            break

    if prob_col is None:
        raise KeyError(
            "Could not find probability column for line 5.5. "
            f"Tried: {possible_prob_cols}. "
            f"Available columns: {bnn.columns.tolist()}"
        )

    if "strikeouts" in bnn.columns:
        actual_col = "strikeouts"
    elif "actual" in bnn.columns:
        actual_col = "actual"
    elif "y_true" in bnn.columns:
        actual_col = "y_true"
    else:
        raise KeyError(
            "Could not find actual strikeouts column. "
            f"Available columns: {bnn.columns.tolist()}"
        )

    p_over = bnn[prob_col]
    y_true_over = (bnn[actual_col] > MAIN_LINE).astype(int)

    brier = brier_score_loss(y_true_over, p_over)
    print("Brier score at 5.5:", round(float(brier), 3))

    prob_true, prob_pred = calibration_curve(
        y_true_over,
        p_over,
        n_bins=10,
        strategy="uniform"
    )

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="BNN")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    plt.xlabel("Predicted probability of over")
    plt.ylabel("Observed frequency of over")
    plt.title("Reliability diagram at strikeout line 5.5")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_and_close("bnn_calibration_curve.png")


# # ------------------------------------------------------------
# # Run everything with error reporting
# # ------------------------------------------------------------
# tasks = [
#     make_local_shap_plot,
#     make_threshold_plot,
#     make_bnn_distribution_plot,
#     make_calibration_curve,
# ]

# for task in tasks:
#     try:
#         task()
#     except Exception as e:
#         print("\nERROR in:", task.__name__)
#         print(type(e).__name__, ":", e)
#         traceback.print_exc()

# print("\nFinished script.")
# print("Files now in reports/figures:")
# for file in sorted(FIGURES_DIR.glob("*.png")):
#     print(" -", file.name)

print("Creating local SHAP explanation plot...")

figures_dir = Path("reports/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

# Load model table
df = pd.read_parquet("data/processed/model_table.parquet")
df["game_date"] = pd.to_datetime(df["game_date"])

# Same test split used in the report
test_df = df[df["game_date"] >= pd.to_datetime("2024-04-01")].copy()

# Load XGBoost feature list
with open("models/xgboost_features.json", "r", encoding="utf-8") as f:
    feature_cols = json.load(f)

# Keep only model features
X_test = test_df[feature_cols].copy()

# IMPORTANT FIX:
# Force all feature columns to numeric float64 for SHAP
for col in feature_cols:
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(0.0).astype(np.float64)

print("X_test shape:", X_test.shape)
print("Any object columns left?", X_test.select_dtypes(include=["object"]).columns.tolist())

# Load trained XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("models/xgboost_model.json")

# Sample for speed
X_sample = X_test.sample(min(1000, len(X_test)), random_state=42)

# Use TreeExplainer directly
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample)

# Choose one example with high predicted strikeouts
preds = xgb_model.predict(X_sample)
i = int(np.argmax(preds))

print("Selected sample index:", i)
print("Predicted strikeouts:", round(float(preds[i]), 3))

# Create a simple local SHAP bar plot
local_values = pd.DataFrame({
    "feature": X_sample.columns,
    "shap_value": shap_values[i]
})

local_values["abs_shap"] = local_values["shap_value"].abs()
local_values = local_values.sort_values("abs_shap", ascending=False).head(10)
local_values = local_values.sort_values("shap_value")

plt.figure(figsize=(8, 6))
plt.barh(local_values["feature"], local_values["shap_value"])
plt.axvline(0, linestyle="--")
plt.xlabel("SHAP value")
plt.title("Local SHAP explanation for one pitcher-game prediction")
plt.tight_layout()

out_path = figures_dir / "shap_local_example.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", out_path)