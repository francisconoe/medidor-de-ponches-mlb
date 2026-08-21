from pathlib import Path
import yaml
import random
import numpy as np


def load_config(path: str = "config.yaml"):
    # Load configuration file
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_directories():
    # Ensure required project directories exist
    dirs = [
        "data/raw",
        "data/processed",
        "models",
        "reports/figures",
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)