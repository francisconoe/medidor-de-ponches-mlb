import pandas as pd
from src.utils import load_config, ensure_directories


def is_strikeout_event(x) -> int:
    
    # Statcast 'events' is populated on the last pitch of an at-bat.
    # We treat anything containing 'strikeout' as a strikeout (includes strikeout_double_play).
    
    if pd.isna(x):
        return 0
    return int("strikeout" in str(x).lower())


def main():
    cfg = load_config()
    ensure_directories()

    clean_path = cfg["data"]["clean_path"]
    out_path = cfg["labels"]["out_path"]

    print("Loading cleaned pitches...")
    df = pd.read_parquet(clean_path)
    print("Pitches shape:", df.shape)

    required = ["game_pk", "game_date", "pitcher", "at_bat_number", "pitch_number", "events"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for labels: {missing}")

    df = df.sort_values(["game_pk", "pitcher", "at_bat_number", "pitch_number"])

    ab_last = df.groupby(["game_pk", "pitcher", "at_bat_number"], as_index=False).tail(1).copy()

    ab_last["is_k"] = ab_last["events"].apply(is_strikeout_event)

    labels = (
        ab_last.groupby(["game_pk", "game_date", "pitcher"], as_index=False)["is_k"]
        .sum()
        .rename(columns={"is_k": "strikeouts"})
    )

    print("Labels shape:", labels.shape)
    print(labels.head(10))

    labels.to_parquet(out_path, index=False)
    print("Saved labels to:", out_path)
    # print("Mean strikeouts:", labels["strikeouts"].mean())
    # print("Max strikeouts:", labels["strikeouts"].max())


if __name__ == "__main__":
    main()
    