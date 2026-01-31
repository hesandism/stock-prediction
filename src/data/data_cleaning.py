import pandas as pd
from pathlib import Path

raw_file = Path("data\raw\TeleCom_2009_2023_Daily.csv")
output_file = Path("data\processed\slt_daily_clean.csv")

def load_raw_data(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"CSV file not found!!!\n"
            "searched file path: {file_path}"
        )
    
    df = pd.read_csv(file_path)
    return df
    