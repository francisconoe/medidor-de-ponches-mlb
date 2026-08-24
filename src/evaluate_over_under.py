import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.utils import ensure_directories


THRESHOLDS = [3.5, 4.5, 5.5, 6.5]


def evaluate_threshold(y_true, y_pred, threshold):
    """
    Evaluate Over/Under classification at a given strikeout line.

    Over  = strikeouts > threshold
    Under = strikeouts <= threshold
    """
    y_true_bin = (y_true > threshold).astype(int)
    y_pred_bin = (y_pred > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_probability_threshold(y_true, prob_over, threshold, prob_cutoff=0.5):
    """
    Evaluate a probability-based Over/Under decision.
    Predict Over if P(Over) >= prob_cutoff.
    """
    y_true_bin = (y_true > threshold).astype(int)
    y_pred_bin = (prob_over >= prob_cutoff).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "prob_cutoff": prob_cutoff,
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    ensure_directories()

    print("Loading prediction files...")

    base_df = pd.read_csv("reports/baseline_predictions.csv")
    xgb_df = pd.read_csv("reports/xgb_predictions.csv")
    nn_df = pd.read_csv("reports/nn_predictions.csv")
    bnn_df = pd.read_csv("reports/bnn_predictions.csv")

    # Start from baseline file
    merged = base_df[
        ["game_pk", "game_date", "game_year", "pitcher", "strikeouts", "pred_mean", "pred_linear"]
    ].copy()

    # Merge XGBoost predictions
    merged = merged.merge(
        xgb_df[["game_pk", "pitcher", "pred_xgb"]],
        on=["game_pk", "pitcher"],
        how="inner"
    )

    # Merge NN predictions
    merged = merged.merge(
        nn_df[["game_pk", "pitcher", "pred_nn"]],
        on=["game_pk", "pitcher"],
        how="inner"
    )

    # Merge BNN predictions
    bnn_cols = [
        "game_pk", "pitcher", "pred_bnn_mean", "pred_bnn_std",
        "prob_over_3.5", "prob_over_4.5", "prob_over_5.5", "prob_over_6.5",
        "prob_under_3.5", "prob_under_4.5", "prob_under_5.5", "prob_under_6.5"
    ]
    merged = merged.merge(
        bnn_df[bnn_cols],
        on=["game_pk", "pitcher"],
        how="inner"
    )

    print("Merged evaluation dataset shape:", merged.shape)

    y_true = merged["strikeouts"].values

    results = []

    # Regression outputs converted into Over/Under by thresholding the predicted strikeout count
    regression_models = {
        "mean_baseline": "pred_mean",
        "linear_regression": "pred_linear",
        "xgboost": "pred_xgb",
        "neural_network": "pred_nn",
        "bayesian_nn_mean": "pred_bnn_mean",
    }

    for model_name, pred_col in regression_models.items():
        y_pred = merged[pred_col].values

        for threshold in THRESHOLDS:
            row = evaluate_threshold(y_true, y_pred, threshold)
            row["model"] = model_name
            row["decision_type"] = "thresholded_regression"
            results.append(row)

    # BNN probability outputs
    for threshold in THRESHOLDS:
        prob_col = f"prob_over_{threshold}"
        prob_over = merged[prob_col].values

        row = evaluate_probability_threshold(y_true, prob_over, threshold, prob_cutoff=0.5)
        row["model"] = "bayesian_nn_probability"
        row["decision_type"] = "probability_cutoff"
        results.append(row)

    results_df = pd.DataFrame(results)

    # Nice ordering
    results_df = results_df[
        [
            "model", "decision_type", "threshold",
            "accuracy", "precision", "recall", "f1",
            "tn", "fp", "fn", "tp"
        ] + (["prob_cutoff"] if "prob_cutoff" in results_df.columns else [])
    ]

    print("\n[OVER/UNDER RESULTS]")
    print(results_df[["model", "threshold", "accuracy", "precision", "recall", "f1"]])

    out_path = "reports/over_under_metrics.csv"
    results_df.to_csv(out_path, index=False)
    print("\nSaved over/under metrics to:", out_path)


if __name__ == "__main__":
    main()