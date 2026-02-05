# verifying features and targets are meaningful

from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score

from src.config import FEATURES_DIR


def time_split(df:pd.DataFrame, train_ratio: float = 0.8):
    n = len(df)
    cut = int(n*train_ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()



def baseline_predict_next_close(test_df:pd.DataFrame):
    """naive baseline: tomorrows close price = todays close price."""
    return test_df["close"]

def baseline_predict_direction(test_df:pd.DataFrame):
    today_ret = test_df["ret_1d"]
    return (today_ret > 0).astype(int)



def main():
    path = FEATURES_DIR/"Toyota_features.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    train_df, test_df = time_split(df, train_ratio=0.8)

    pred_close = baseline_predict_next_close(test_df)
    mae_close = mean_absolute_error(test_df["target_next_close"],pred_close)

    pred_dir = baseline_predict_direction(test_df)
    acc_dir = accuracy_score(test_df["target_direction"], pred_dir)

    print("===== BASELINE RESULTS =====")
    print(f"Test rows: {len(test_df)}")
    print(f"MAE (next close): {mae_close:.4f}")
    print(f"Accuracy (direction): {acc_dir:.4f}")


if __name__ == "__main__":
    main()