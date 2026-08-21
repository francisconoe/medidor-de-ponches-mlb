from src.utils import load_config, set_seed, ensure_directories

config = load_config()
set_seed(config["project"]["seed"])
ensure_directories()

print("Config loaded successfully.")
print("Directories ensured.")