import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.config import FEATURES_DIR


# =======================
# CONFIG (tune these)
# =======================
DEADZONE = 0.002          # label as 1 if next_return > +DEADZONE, 0 if < -DEADZONE (else drop)
VALID_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

# Confidence-filtering (only "act" on high-confidence days)
CONF_HIGH = 0.60          # act long if proba >= CONF_HIGH
CONF_LOW = 0.40           # act short if proba <= CONF_LOW

# Walk-forward settings (you can tune these)
USE_WALK_FORWARD = True
WF_TEST_SIZE = 500        # number of samples per test window
WF_STEP = 500             # step size to move the window forward


# =======================
# FEATURE ENGINEERING
# =======================
def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a small set of strong, non-leaky regime features from existing columns.
    Uses only rolling windows (past-only), so no leakage.
    """
    df = df.copy()

    # Volatility regime
    if "ret_std_5" in df.columns and "ret_std_20" in df.columns:
        df["vol_ratio_5_20"] = df["ret_std_5"] / df["ret_std_20"].replace(0, np.nan)

    # Trend strength from SMAs
    if "close_sma_5" in df.columns and "close_sma_20" in df.columns:
        df["trend_5_20"] = (df["close_sma_5"] / df["close_sma_20"]) - 1.0
    if "close_sma_10" in df.columns and "close_sma_20" in df.columns:
        df["trend_10_20"] = (df["close_sma_10"] / df["close_sma_20"]) - 1.0

    # Price position in recent range
    if {"close", "high", "low"}.issubset(df.columns):
        roll_high_10 = df["high"].rolling(10).max()
        roll_low_10 = df["low"].rolling(10).min()
        denom = (roll_high_10 - roll_low_10).replace(0, np.nan)
        df["range_pos_10"] = (df["close"] - roll_low_10) / denom

    # ATR-like average range
    if "hl_range" in df.columns:
        df["hl_mean_14"] = df["hl_range"].rolling(14).mean()
        df["hl_z_14"] = df["hl_range"] / df["hl_mean_14"].replace(0, np.nan)

    # Shock / z-score style feature
    if "ret_1d" in df.columns and "ret_std_20" in df.columns:
        df["abs_ret_1d"] = df["ret_1d"].abs()
        df["ret_z_20"] = df["ret_1d"] / df["ret_std_20"].replace(0, np.nan)

    return df


def make_strict_direction_target(df: pd.DataFrame, deadzone: float) -> pd.Series:
    """
    1 if target_next_return > +deadzone
    0 if target_next_return < -deadzone
    else NaN (dropped)
    """
    y = np.where(
        df["target_next_return"] > deadzone, 1,
        np.where(df["target_next_return"] < -deadzone, 0, np.nan)
    )
    return pd.Series(y, index=df.index, name="target_direction_strict")


def tune_threshold(y_true: np.ndarray, proba: np.ndarray):
    """
    Tune threshold on VALID using balanced accuracy.
    Returns best_threshold and metrics at that threshold.
    """
    best_t, best_bal, best_acc = 0.5, -1.0, -1.0
    for t in np.arange(0.35, 0.66, 0.01):
        pred = (proba >= t).astype(int)
        bal = balanced_accuracy_score(y_true, pred)
        acc = accuracy_score(y_true, pred)
        if bal > best_bal:
            best_t, best_bal, best_acc = float(t), float(bal), float(acc)
    return best_t, best_bal, best_acc


def eval_with_confidence_filter(y_true: np.ndarray, proba: np.ndarray, high=0.60, low=0.40):
    """
    Evaluate accuracy only on high-confidence predictions:
    - predict 1 when proba >= high
    - predict 0 when proba <= low
    - ignore middle region
    """
    mask = (proba >= high) | (proba <= low)
    coverage = float(mask.mean())
    if mask.sum() == 0:
        return coverage, np.nan, np.nan, np.nan

    pred = (proba[mask] >= 0.5).astype(int)
    yt = y_true[mask]
    acc = float(accuracy_score(yt, pred))
    bal = float(balanced_accuracy_score(yt, pred))
    auc = float(roc_auc_score(yt, proba[mask])) if len(np.unique(yt)) > 1 else np.nan
    return coverage, acc, bal, auc


# =======================
# XGBOOST TRAINING (native, works on old xgboost)
# =======================
def train_xgb_native(X_train_np, y_train_np, X_valid_np, y_valid_np):
    dtrain = xgb.DMatrix(X_train_np, label=y_train_np)
    dvalid = xgb.DMatrix(X_valid_np, label=y_valid_np)

    pos = int((y_train_np == 1).sum())
    neg = int((y_train_np == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    # Robust, simple, regularized params (finance-friendly)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": RANDOM_STATE,

        # Keep it simple to generalize
        "max_depth": 2,
        "min_child_weight": 20,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "gamma": 0.2,
        "lambda": 20.0,
        "eta": 0.02,

        "scale_pos_weight": scale_pos_weight,

        # GPU (if supported). If errors, switch to "hist"
        "tree_method": "hist",
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=8000,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=300,
        verbose_eval=200,
    )
    return booster


# =======================
# MAIN
# =======================
def main():
    # ---------- Load ----------
    path = FEATURES_DIR / "Toyota_features.csv"
    df = pd.read_csv(path)

    # Sort by time
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    # Check required cols
    if "target_next_return" not in df.columns:
        raise ValueError("target_next_return column is required in Toyota_features.csv")

    # Add features
    df = add_extra_features(df)

    # Target (strict direction)
    y = make_strict_direction_target(df, DEADZONE)

    # Remove leakage
    leak_cols = {
        "date",
        "target_next_close",
        "target_next_return",
        "target_direction",
        "target_direction_strict",
    }
    X = df.drop(columns=[c for c in leak_cols if c in df.columns], errors="ignore")

    # Drop NaNs
    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(columns=[y.name])
    y = data[y.name].astype(int)

    # ---------- Print dataset stats ----------
    n = len(X)
    print(f"Total rows (after deadzone+NaN drop): {n}")
    print(f"Rows kept from original: {n}/{len(df)} ({n/len(df):.2%})")
    print(f"Using strict direction with deadzone = {DEADZONE:.4f}")

    # ---------- Simple Train/Valid/Test split ----------
    test_split_idx = int(n * (1 - TEST_SIZE))
    valid_split_idx = int(n * (1 - (TEST_SIZE + VALID_SIZE)))

    X_train = X.iloc[:valid_split_idx]
    y_train = y.iloc[:valid_split_idx]

    X_valid = X.iloc[valid_split_idx:test_split_idx]
    y_valid = y.iloc[valid_split_idx:test_split_idx]

    X_test = X.iloc[test_split_idx:]
    y_test = y.iloc[test_split_idx:]

    print(f"\nTrain rows: {len(X_train)}")
    print(f"Valid rows: {len(X_valid)}")
    print(f"Test rows : {len(X_test)}")

    # ---------- Baseline ----------
    if "ret_1d" in X_test.columns:
        base_pred = (X_test["ret_1d"].astype(float).values > 0).astype(int)
        print("\n===== BASELINE (sign of ret_1d) =====")
        print("Accuracy:", accuracy_score(y_test, base_pred))
        print("Balanced Accuracy:", balanced_accuracy_score(y_test, base_pred))
        print("Confusion:\n", confusion_matrix(y_test, base_pred))

    # ---------- Convert to float32 numpy ----------
    X_train_np = X_train.astype(np.float32).to_numpy()
    X_valid_np = X_valid.astype(np.float32).to_numpy()
    X_test_np = X_test.astype(np.float32).to_numpy()

    y_train_np = y_train.to_numpy(dtype=np.int32)
    y_valid_np = y_valid.to_numpy(dtype=np.int32)
    y_test_np = y_test.to_numpy(dtype=np.int32)

    # ---------- Train native XGBoost ----------
    try:
        booster = train_xgb_native(X_train_np, y_train_np, X_valid_np, y_valid_np)
    except xgb.core.XGBoostError as e:
        # If GPU isn't available, fall back to CPU hist
        print("\n[GPU not available or gpu_hist unsupported] Falling back to CPU hist.")
        print("Error was:", str(e))

        def train_cpu(Xtr, ytr, Xva, yva):
            dtrain = xgb.DMatrix(Xtr, label=ytr)
            dvalid = xgb.DMatrix(Xva, label=yva)
            pos = int((ytr == 1).sum())
            neg = int((ytr == 0).sum())
            scale_pos_weight = neg / max(pos, 1)
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "seed": RANDOM_STATE,
                "max_depth": 2,
                "min_child_weight": 20,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "gamma": 0.2,
                "lambda": 20.0,
                "eta": 0.02,
                "scale_pos_weight": scale_pos_weight,
                "tree_method": "hist",
            }
            return xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=8000,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=300,
                verbose_eval=200,
            )

        booster = train_cpu(X_train_np, y_train_np, X_valid_np, y_valid_np)

    # ---------- Predict VALID / tune threshold ----------
    dvalid = xgb.DMatrix(X_valid_np, label=y_valid_np)
    valid_proba = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))

    best_t, best_bal, best_acc = tune_threshold(y_valid_np, valid_proba)

    print("\n===== THRESHOLD TUNING (VALID) =====")
    print(f"Best threshold: {best_t:.2f}")
    print(f"VALID Balanced Accuracy: {best_bal:.4f}")
    print(f"VALID Accuracy: {best_acc:.4f}")
    print(f"VALID ROC-AUC: {roc_auc_score(y_valid_np, valid_proba):.4f}")

    cov_v, acc_v, bal_v, auc_v = eval_with_confidence_filter(
        y_valid_np, valid_proba, high=CONF_HIGH, low=CONF_LOW
    )
    print("\n===== CONFIDENCE FILTER (VALID) =====")
    print(f"Coverage: {cov_v:.2%} (fraction of days acted)")
    print(f"Accuracy: {acc_v}")
    print(f"Balanced Accuracy: {bal_v}")
    print(f"ROC-AUC (filtered): {auc_v}")

    # ---------- Predict TEST ----------
    dtest = xgb.DMatrix(X_test_np, label=y_test_np)
    test_proba = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    test_pred = (test_proba >= best_t).astype(int)

    print("\n===== XGBOOST DIRECTION (TEST) =====")
    print("Accuracy:", accuracy_score(y_test_np, test_pred))
    print("Balanced Accuracy:", balanced_accuracy_score(y_test_np, test_pred))
    print("ROC-AUC:", roc_auc_score(y_test_np, test_proba))
    print("\nConfusion matrix:\n", confusion_matrix(y_test_np, test_pred))
    print("\nClassification report:\n", classification_report(y_test_np, test_pred))

    cov_t, acc_t, bal_t, auc_t = eval_with_confidence_filter(
        y_test_np, test_proba, high=CONF_HIGH, low=CONF_LOW
    )
    print("\n===== CONFIDENCE FILTER (TEST) =====")
    print(f"Coverage: {cov_t:.2%} (fraction of days acted)")
    print(f"Accuracy: {acc_t}")
    print(f"Balanced Accuracy: {bal_t}")
    print(f"ROC-AUC (filtered): {auc_t}")

    # ---------- Optional: Walk-forward evaluation ----------
    if USE_WALK_FORWARD:
        print("\n===== WALK-FORWARD EVALUATION =====")
        X_all = X.astype(np.float32).to_numpy()
        y_all = y.to_numpy(dtype=np.int32)

        start_train_end = max(int(n * 0.6), 2000)  # initial training length
        test_size = min(WF_TEST_SIZE, n - start_train_end - 1)
        step = min(WF_STEP, test_size)

        wf_accs, wf_aucs, wf_covs = [], [], []
        i = 0
        for train_end in range(start_train_end, n - test_size, step):
            test_start = train_end
            test_end = train_end + test_size

            Xtr, ytr = X_all[:train_end], y_all[:train_end]
            Xva, yva = X_all[train_end - test_size:train_end], y_all[train_end - test_size:train_end]
            Xte, yte = X_all[test_start:test_end], y_all[test_start:test_end]

            booster_wf = train_xgb_native(Xtr, ytr, Xva, yva)

            dte = xgb.DMatrix(Xte, label=yte)
            proba_te = booster_wf.predict(dte, iteration_range=(0, booster_wf.best_iteration + 1))

            # confidence-filtered accuracy (most meaningful)
            cov, acc, _, _ = eval_with_confidence_filter(yte, proba_te, high=CONF_HIGH, low=CONF_LOW)
            wf_covs.append(cov)
            wf_accs.append(acc if not np.isnan(acc) else 0.0)

            # AUC on full test window
            try:
                wf_aucs.append(roc_auc_score(yte, proba_te))
            except ValueError:
                wf_aucs.append(np.nan)

            i += 1

        print(f"Windows evaluated: {i}")
        print("Avg confidence coverage:", float(np.nanmean(wf_covs)))
        print("Avg confidence-filtered accuracy:", float(np.nanmean(wf_accs)))
        print("Avg ROC-AUC:", float(np.nanmean(wf_aucs)))


if __name__ == "__main__":
    main()
