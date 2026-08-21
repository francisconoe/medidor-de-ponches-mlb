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


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


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
        x = self.out(x)
        return x.squeeze(1)


def mc_predict(model, X_tensor, n_samples=100):
    """
    MC Dropout inference:
    keep dropout ON at test time and sample multiple predictions.
    Returns:
        preds:      [n_samples, n_examples]
        pred_mean:  [n_examples]
        pred_std:   [n_examples]
    """
    model.train()  # keep dropout active
    preds = []

    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X_tensor).cpu().numpy()
            preds.append(pred)

    preds = np.stack(preds, axis=0)
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)

    return preds, pred_mean, pred_std


def compute_probabilities(mc_preds, thresholds):
    """
    Compute P(K > threshold) for each threshold.
    mc_preds shape: [n_samples, n_examples]
    """
    prob_dict = {}
    for threshold in thresholds:
        prob_dict[f"prob_over_{threshold}"] = (mc_preds > threshold).mean(axis=0)
    return prob_dict


def main():
    cfg = load_config()
    ensure_directories()
    set_seed(cfg["project"]["seed"])

    data_path = cfg["dataset"]["out_path"]
    test_start = pd.to_datetime(cfg["split"]["test_start_date"])
    mc_samples = cfg["bnn"]["mc_samples"]

    # sportsbook-style lines
    thresholds = [3.5, 4.5, 5.5, 6.5]

    df = pd.read_parquet(data_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    train_df = df[df["game_date"] < test_start].copy()
    test_df = df[df["game_date"] >= test_start].copy()

    y_train = train_df["strikeouts"].values.astype(np.float32)
    y_test = test_df["strikeouts"].values.astype(np.float32)

    drop_cols = ["game_pk", "game_date", "game_year", "pitcher", "strikeouts"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].fillna(0.0).values
    X_test = test_df[feature_cols].fillna(0.0).values

    # scale inputs
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=cfg["nn"]["batch_size"],
        shuffle=True
    )

    model = MCDropoutRegressor(
        input_dim=X_train.shape[1],
        hidden_sizes=tuple(cfg["nn"]["hidden_sizes"]),
        dropout=cfg["nn"]["dropout"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["nn"]["lr"],
        weight_decay=cfg["nn"]["weight_decay"]
    )
    loss_fn = nn.MSELoss()

    # Train
    model.train()
    for epoch in range(cfg["nn"]["epochs"]):
        epoch_losses = []

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f"Epoch {epoch + 1:02d}/{cfg['nn']['epochs']} - loss={np.mean(epoch_losses):.4f}")

    # MC Dropout inference
    mc_preds, pred_mean, pred_std = mc_predict(model, X_test_t, n_samples=mc_samples)

    # Multi-threshold probabilities
    prob_dict = compute_probabilities(mc_preds, thresholds)

    mae = mean_absolute_error(y_test, pred_mean)
    r = rmse(y_test, pred_mean)

    print("\n[BAYESIAN NEURAL NETWORK RESULTS]")
    print(f"MAE = {mae:.3f}")
    print(f"RMSE = {r:.3f}")
    print(f"Mean predictive std = {pred_std.mean():.3f}")

    # Save model weights
    torch.save(model.state_dict(), "models/bnn_model.pt")

    # Save scaler
    joblib.dump(scaler, "models/bnn_scaler.pkl")

    # Save feature order
    with open("models/bnn_features.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    # Save test predictions
    pred_df = test_df[["game_pk", "game_date", "game_year", "pitcher", "strikeouts"]].copy()
    pred_df["pred_bnn_mean"] = pred_mean
    pred_df["pred_bnn_std"] = pred_std

    for col_name, values in prob_dict.items():
        pred_df[col_name] = values
        pred_df[col_name.replace("prob_over", "prob_under")] = 1.0 - values

    pred_df.to_csv("reports/bnn_predictions.csv", index=False)

    # Save metrics
    metrics_row = {
        "model": "bayesian_neural_network",
        "mae": mae,
        "rmse": r,
        "mean_predictive_std": float(pred_std.mean())
    }

    for threshold in thresholds:
        metrics_row[f"avg_prob_over_{threshold}"] = float(prob_dict[f"prob_over_{threshold}"].mean())

    metrics_df = pd.DataFrame([metrics_row])
    metrics_df.to_csv("reports/metrics_bnn.csv", index=False)

    print("Saved model to models/bnn_model.pt")
    print("Saved scaler to models/bnn_scaler.pkl")
    print("Saved feature list to models/bnn_features.json")
    print("Saved predictions to reports/bnn_predictions.csv")
    print("Saved metrics to reports/metrics_bnn.csv")


if __name__ == "__main__":
    main()