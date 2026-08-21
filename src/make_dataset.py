import pandas as pd
from src.utils import load_config, ensure_directories


def main():
    cfg = load_config()
    ensure_directories()

    feat_path = cfg["features"]["out_path"]
    lab_path = cfg["labels"]["out_path"]
    out_path = cfg["dataset"]["out_path"]

    print("Loading features:", feat_path)
    X = pd.read_parquet(feat_path)
    print("Features shape:", X.shape)

    print("Loading labels:", lab_path)
    y = pd.read_parquet(lab_path)
    print("Labels shape:", y.shape)

    keys = ["game_pk", "game_date", "pitcher"]
    df = X.merge(y, on=keys, how="inner")

    # Sort by time for later time-based splitting
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    print("Model table shape:", df.shape)
    print(df[["game_date", "game_year", "pitcher", "strikeouts"]].head(5))

    df.to_parquet(out_path, index=False)
    print("Saved model table to:", out_path)

    # Quick sanity stats
    print("Mean strikeouts (starters):", df["strikeouts"].mean())
    print("Max strikeouts (starters):", df["strikeouts"].max())


if __name__ == "__main__":
    main()