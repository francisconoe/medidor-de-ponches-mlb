from pybaseball import statcast
import pandas as pd
from datetime import datetime, timedelta
from src.utils import load_config, set_seed, ensure_directories

def daterange(start_date, end_date):
    # Yield monthly date ranges between start and end.
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    current = start

    while current < end:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield current.strftime("%Y-%m-%d"), min(next_month, end).strftime("%Y-%m-%d")
        current = next_month


def main():

    # Load config
    config = load_config()

    # Set random seed
    set_seed(config["project"]["seed"])

    # Ensure directories exist
    ensure_directories()

    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    output_path = config["data"]["raw_path"]

    print(f"Pulling Statcast data from {start_date} to {end_date}")

    df = statcast(start_dt=start_date, end_dt=end_date)

    print("Rows downloaded:", len(df))
    print("Saving dataset...")

    df.to_parquet(output_path, index=False)

    print("Saved raw dataset to:", output_path)


if __name__ == "__main__":
    main()