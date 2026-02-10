import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

from src.config import FEATURES_DIR


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    # ------------------ Load ------------------
    path = FEATURES_DIR / "Toyota_features.csv"
    df = pd.read_csv(path)

    # ------------------ Target / leakage ------------------
    y_col = "target_next_return"
    leak_cols = {
        "date",
        "target_next_close",
        "target_next_return",
        "target_direction",
    }

    X = df.drop(columns=[c for c in leak_cols if c in df.columns], errors="ignore")
    y = df[y_col]

    # Drop rows with NaNs in X or y
    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(columns=[y_col])
    y = data[y_col].astype(float)

    # ------------------ Time split (train/valid/test) ------------------
    test_size = valid_size = 0.15
    n = len(X)

    test_split_idx = int(n * (1 - test_size))
    valid_split_idx = int(n * (1 - (test_size + valid_size)))

    X_train = X.iloc[:valid_split_idx].copy()
    y_train = y.iloc[:valid_split_idx].copy()

    X_valid = X.iloc[valid_split_idx:test_split_idx].copy()
    y_valid = y.iloc[valid_split_idx:test_split_idx].copy()

    X_test = X.iloc[test_split_idx:].copy()
    y_test = y.iloc[test_split_idx:].copy()

    print(f"Total rows: {len(X)}")
    print(f"Train rows: {len(X_train)}")
    print(f"Valid rows: {len(X_valid)}")
    print(f"Test rows : {len(X_test)}")

    # ------------------ Baselines ------------------
    print("\n===== BASELINE 1 (Predict 0 return) =====")
    baseline_zero = np.zeros(len(y_test), dtype=float)
    print("MAE :", mean_absolute_error(y_test, baseline_zero))
    print("RMSE:", rmse(y_test, baseline_zero))
    print("R2  :", r2_score(y_test, baseline_zero))

    # If your features include ret_1d, use it as a "persistence return" baseline
    if "ret_1d" in X_test.columns:
        print("\n===== BASELINE 2 (Predict next return = today's return) =====")
        baseline_ret = X_test["ret_1d"].astype(float).values
        print("MAE :", mean_absolute_error(y_test, baseline_ret))
        print("RMSE:", rmse(y_test, baseline_ret))
        print("R2  :", r2_score(y_test, baseline_ret))

    # ------------------ Model + search ------------------
    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        random_state=42,
        n_jobs=-1,
    )

    param_dist = {
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [300, 600, 900],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 5, 10],
        "gamma": [0, 0.05, 0.1],
        "reg_lambda": [1, 5, 10],
    }

    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=12,
        scoring="neg_mean_absolute_error",  # minimize MAE on returns
        cv=tscv,
        verbose=2,
        random_state=42,
        n_jobs=1,  # GPU-friendly
        error_score="raise",
    )

    # Tune on train only
    search.fit(X_train, y_train)

    print("\nBest params from search:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    best_model = search.best_estimator_

    # ------------------ Refit on train + valid ------------------
    X_train_full = pd.concat([X_train, X_valid], axis=0)
    y_train_full = pd.concat([y_train, y_valid], axis=0)

    best_model.fit(X_train_full, y_train_full)

    # ------------------ Evaluate on test ------------------
    pred = best_model.predict(X_test)

    print("\n===== XGBOOST REGRESSOR (Next-day Return) =====")
    print("MAE :", mean_absolute_error(y_test, pred))
    print("RMSE:", rmse(y_test, pred))
    print("R2  :", r2_score(y_test, pred))

    # Optional: direction accuracy derived from predicted return sign
    true_dir = (y_test.values > 0).astype(int)
    pred_dir = (pred > 0).astype(int)
    print("\nDerived Direction Accuracy (sign of return):", accuracy_score(true_dir, pred_dir))


if __name__ == "__main__":
    main()
