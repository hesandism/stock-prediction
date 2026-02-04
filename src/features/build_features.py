from pathlib import Path
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, FEATURES_DIR

def add_targets(df:pd.DataFrame):
    """
    Targets:
    target_next_close: next day close price.
    target_next_return: next day return price.
    target_direction: 1 if next day return > 0 else 0
    """

    df = df.copy()
    
    df["target_next_close"] = df["close"].shift(-1)
    df["target_next_return"] = (df["target_next_close"]/df["close"])-1.0
    df["target_direction"] = (df["target_next_return"]>0).astype(int)

    return df



