import pandas as pd
import numpy as np
from src.utils import load_config, ensure_directories


STARTER_MIN_PITCHES = 40  # proxy for starters


def zone_rate_from_zone(series: pd.Series) -> float:
    
    # Statcast 'zone' is 1-14 for pitch location buckets.
    # Zones 1-9 are typically considered in the strike zone.
    
    s = pd.to_numeric(series, errors="coerce")
    return s.between(1, 9).mean()

def safe_nanquantile(x, q):
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() == 0:
        return np.nan
    return np.nanquantile(x, q)


def main():
    cfg = load_config()
    ensure_directories()

    clean_path = cfg["data"]["clean_path"]
    out_path = cfg.get("features", {}).get("out_path", "data/processed/game_features.parquet")

    print("Loading cleaned pitches...")
    df = pd.read_parquet(clean_path)
    print("Pitches shape:", df.shape)

    required = [
        "game_pk", "game_date", "game_year", "pitcher",
        "pitch_type", "release_speed", "release_spin_rate",
        "pfx_x", "pfx_z", "plate_x", "plate_z",
        "balls", "strikes", "zone",
        "at_bat_number"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for features: {missing}")

    gkeys = ["game_pk", "game_date", "game_year", "pitcher"]

    # Basic per-game aggregates
    agg = df.groupby(gkeys).agg(
        n_pitches=("pitch_type", "size"),
        n_at_bats=("at_bat_number", "nunique"),

        velo_mean=("release_speed", "mean"),
        velo_std=("release_speed", "std"),
        velo_p10=("release_speed", lambda x: safe_nanquantile(x, 0.10)),
        velo_p50=("release_speed", lambda x: safe_nanquantile(x, 0.50)),
        velo_p90=("release_speed", lambda x: safe_nanquantile(x, 0.90)),

        spin_mean=("release_spin_rate", "mean"),
        spin_std=("release_spin_rate", "std"),

        pfx_x_mean=("pfx_x", "mean"),
        pfx_z_mean=("pfx_z", "mean"),

        plate_x_mean=("plate_x", "mean"),
        plate_z_mean=("plate_z", "mean"),

        pct_two_strike=("strikes", lambda x: (x == 2).mean()),

        zone_rate=("zone", zone_rate_from_zone),
    ).reset_index()

    # Pitch mix percentages (one column per pitch type)
    mix = df.groupby(gkeys + ["pitch_type"]).size().reset_index(name="count")
    mix["pct"] = mix["count"] / mix.groupby(gkeys)["count"].transform("sum")
    mix_pivot = mix.pivot_table(index=gkeys, columns="pitch_type", values="pct", fill_value=0.0)
    mix_pivot.columns = [f"pct_{c}" for c in mix_pivot.columns]
    mix_pivot = mix_pivot.reset_index()

    features = agg.merge(mix_pivot, on=gkeys, how="left")

    print("Game-level features shape (all pitchers):", features.shape)

    # Starters-only filter
    starters = features[features["n_pitches"] >= STARTER_MIN_PITCHES].copy()
    print(f"Starters-only (n_pitches >= {STARTER_MIN_PITCHES}) shape:", starters.shape)

    starters.to_parquet(out_path, index=False)
    print("Saved features to:", out_path)


if __name__ == "__main__":
    main()