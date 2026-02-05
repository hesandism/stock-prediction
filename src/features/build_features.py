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


def add_features(df:pd.DataFrame):
    df = df.copy()

    df["ret_1d"] = df["close"].pct_change()
    df["logret_1d"] = np.log(df["close"]).diff()

    df["hl_range"] = (df["high"] - df["low"])/df["close"]  # normalized by close.
    df["co_range"] = (df["close"]-df["open"])/df["open"]

    # defining rolling window sizes
    windows = [5, 10, 20]

    for w in windows:
        df[f"ret_mean_{w}"] = df["ret_1d"].rolling(w).mean()
        df[f"ret_std_{w}"] = df["ret_1d"].rolling(w).std()
        df[f"close_sma_{w}"] = df["close"].rolling(w).mean()
        df[f"close_to_sma_{w}"] = (df["close"] / df[f"close_sma_{w}"])-1.0

        df[f"mom_{w}"] = df["close"].pct_change(w)

    if "volume" in df.columns:
        df["vol_chg_1d"] = df["volume"].pct_change()
        for w in windows:
            df[f"vol_sma_{w}"] = df["volume"].rolling(w).mean()
            df[f"vol_to_sma_{w}"] = (df["volume"] / df[f"vol_sma_{w}"])-1.0
    
    return df







