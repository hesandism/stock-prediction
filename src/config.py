from pathlib import Path

# This is the place to put File paths, hyperparameters, constants

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT/"data"
RAW_DIR = DATA_DIR/"raw"
PROCESSED_DIR = DATA_DIR/"processed"
FEATURES_DIR = DATA_DIR/"features"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)