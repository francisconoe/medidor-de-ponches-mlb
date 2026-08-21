import pandas as pd

from src.utils import load_config, ensure_directories


COLUMNS_TO_KEEP = [
    "game_pk",
    "game_date",
    "game_year",
    "pitcher",
    "batter",
    "player_name",
    "pitch_type",
    "at_bat_number",
    "pitch_number",
    "release_speed",
    "release_spin_rate",
    "release_pos_x",
    "release_pos_z",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "zone",
    "balls",
    "strikes",
    "inning",
    "inning_topbot",
    "stand",
    "p_throws",
    "events",
    "description",
    "type",
    "home_team",
    "away_team"
]


def main():

    config = load_config()
    ensure_directories()

    raw_path = config["data"]["raw_path"]

    print("Loading raw dataset...")
    df = pd.read_parquet(raw_path)

    print("Original shape:", df.shape)

    # keep only relevant columns
    df = df[COLUMNS_TO_KEEP]

    print("After column selection:", df.shape)

    # remove rows with missing pitcher or game
    df = df.dropna(subset=["pitcher", "game_date"])

    # convert date
    df["game_date"] = pd.to_datetime(df["game_date"])

    print("After basic cleaning:", df.shape)

    output_path = "data/processed/pitches_clean.parquet"

    print("Saving cleaned dataset...")
    df.to_parquet(output_path, index=False)

    print("Saved to:", output_path)


if __name__ == "__main__":
    main()