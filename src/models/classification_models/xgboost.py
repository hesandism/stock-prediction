import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from src.config import FEATURES_DIR


def main():
    path = FEATURES_DIR / "Toyota_features.csv"
    df = pd.read_csv(path)

    y_col = "target_direction"
    leak_cols = {"date", "target_next_close", "target_next_return", "target_direction"}

    
    X = df.drop(columns=[c for c in leak_cols if c in df.columns], errors="ignore")
    y = df[y_col].astype(int)

    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(columns=[y_col])
    y = data[y_col].astype(int)

   
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

    
    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        n_jobs=-1,
        random_state=42,
    )

    param_dist = {
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 400],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 5, 10],
        "gamma": [0, 0.05, 0.1],
        "reg_lambda": [1, 5, 10],
    }

    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=12,
        scoring="roc_auc",
        cv=tscv,
        verbose=2,
        random_state=42,
        n_jobs=1, 
        error_score="raise", 
    )

    search.fit(X_train, y_train)

    print("\nBest params from search:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

  
    best_model = search.best_estimator_


    X_train_full = pd.concat([X_train, X_valid], axis=0)
    y_train_full = pd.concat([y_train, y_valid], axis=0)

    best_model.fit(X_train_full, y_train_full)

    # Evaluate
    proba = best_model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\nAccuracy:", accuracy_score(y_test, pred))
    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification report:\n", classification_report(y_test, pred))


if __name__ == "__main__":
    main()
