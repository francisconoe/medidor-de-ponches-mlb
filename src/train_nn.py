import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils import load_config, ensure_directories, set_seed


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(256, 128), dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[1], 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def main():
    cfg = load_config()
    ensure_directories()
    set_seed(cfg["project"]["seed"])

    data_path = cfg["dataset"]["out_path"]
    test_start = pd.to_datetime(cfg["split"]["test_start_date"])

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

    model = MLPRegressor(
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

        print(f"Epoch {epoch+1:02d}/{cfg['nn']['epochs']} - loss={np.mean(epoch_losses):.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()

    mae = mean_absolute_error(y_test, preds)
    r = rmse(y_test, preds)

    print("\n[NEURAL NETWORK RESULTS]")
    print(f"MAE = {mae:.3f}")
    print(f"RMSE = {r:.3f}")

    # save model weights
    torch.save(model.state_dict(), "models/nn_model.pt")

    # save scaler
    joblib.dump(scaler, "models/nn_scaler.pkl")

    # save feature order
    with open("models/nn_features.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    # save predictions
    pred_df = test_df[["game_pk", "game_date", "game_year", "pitcher", "strikeouts"]].copy()
    pred_df["pred_nn"] = preds
    pred_df.to_csv("reports/nn_predictions.csv", index=False)

    # save metrics
    metrics_df = pd.DataFrame([
        {"model": "neural_network", "mae": mae, "rmse": r}
    ])
    metrics_df.to_csv("reports/metrics_nn.csv", index=False)

    print("Saved model to models/nn_model.pt")
    print("Saved scaler to models/nn_scaler.pkl")
    print("Saved feature list to models/nn_features.json")
    print("Saved predictions to reports/nn_predictions.csv")
    print("Saved metrics to reports/metrics_nn.csv")


if __name__ == "__main__":
    main()