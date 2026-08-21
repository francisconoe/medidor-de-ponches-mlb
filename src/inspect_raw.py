# src/inspect_raw.py
import pandas as pd
from src.utils import load_config, ensure_directories

def main():
    cfg = load_config()
    ensure_directories()

    path = cfg["data"]["raw_path"]
    df = pd.read_parquet(path)

    print("[OK] Loaded:", path)
    print("[OK] Shape:", df.shape)
    print("[OK] Columns:", len(df.columns))
    print(df.head(3))
    print("\nSample columns:", df.columns.tolist()[:25])

if __name__ == "__main__":
    main()