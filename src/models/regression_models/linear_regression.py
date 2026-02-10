import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import FEATURES_DIR



def main():
    path = FEATURES_DIR / "Toyota_features.csv"
    df = pd.read_csv(path)

    y_col = "target_next_close"
    leak_cols = {"date", "target_next_close", "target_next_return", "target_direction"}

    # Keep additional columns for later analysis
    X = df.drop(columns=[c for c in leak_cols if c in df.columns], errors="ignore")
    y = df[y_col]
    
    # Store direction and current close for directional accuracy calculation
    target_direction = df["target_direction"].copy()
    current_close = df["close"].copy()

    data = pd.concat([X, y, target_direction, current_close], axis=1).dropna()
    X = data.drop(columns=[y_col, "target_direction", "close"])
    y = data[y_col]
    target_direction = data["target_direction"]
    current_close = data["close"]

   
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
    
    # Also split direction and close for test set (for directional accuracy)
    target_direction_test = target_direction.iloc[test_split_idx:].copy()
    current_close_test = current_close.iloc[test_split_idx:].copy()

    print(f"Total rows: {len(X)}")
    print(f"Train rows: {len(X_train)}")
    print(f"Valid rows: {len(X_valid)}")
    print(f"Test rows : {len(X_test)}")

    
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(random_state=42, max_iter=10000))
    ])

    param_dist = {
        "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
        "model__fit_intercept": [True, False],
        "model__selection": ["cyclic", "random"],
    }

    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        verbose=2,
        random_state=42,
        n_jobs=-1,
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
    pred_train = best_model.predict(X_train_full)
    pred_test = best_model.predict(X_test)

    # Calculate metrics
    train_mae = mean_absolute_error(y_train_full, pred_train)
    train_mse = mean_squared_error(y_train_full, pred_train)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train_full, pred_train)

    test_mae = mean_absolute_error(y_test, pred_test)
    test_mse = mean_squared_error(y_test, pred_test)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, pred_test)

    print("\n" + "="*60)
    print("TRAINING SET METRICS")
    print("="*60)
    print(f"MAE (Mean Absolute Error): {train_mae:.4f}")
    print(f"MSE (Mean Squared Error):  {train_mse:.4f}")
    print(f"RMSE (Root MSE):           {train_rmse:.4f}")
    print(f"R² Score:                  {train_r2:.4f}")

    print("\n" + "="*60)
    print("TEST SET METRICS")
    print("="*60)
    print(f"MAE (Mean Absolute Error): {test_mae:.4f}")
    print(f"MSE (Mean Squared Error):  {test_mse:.4f}")
    print(f"RMSE (Root MSE):           {test_rmse:.4f}")
    print(f"R² Score:                  {test_r2:.4f}")

    # Calculate MAPE (Mean Absolute Percentage Error)
    train_mape = np.mean(np.abs((y_train_full - pred_train) / y_train_full)) * 100
    test_mape = np.mean(np.abs((y_test - pred_test) / y_test)) * 100
    
    # Calculate Directional Accuracy (how often we predict the direction correctly)
    # Get actual direction from original data
    actual_direction = (df['target_direction'].iloc[test_split_idx:]).values
    # Calculate predicted direction (1 if price predicted to go up, 0 if down)
    # Compare current close to predicted next close
    current_close = df['close'].iloc[test_split_idx:].values
    predicted_direction = (pred_test > current_close).astype(int)
    directional_accuracy = (actual_direction == predicted_direction).mean() * 100

    print("\n" + "="*60)
    print("ADDITIONAL METRICS")
    print("="*60)
    print(f"Train MAPE (Mean Absolute % Error): {train_mape:.2f}%")
    print(f"Test MAPE (Mean Absolute % Error):  {test_mape:.2f}%")
    print(f"Directional Accuracy (Test Set):    {directional_accuracy:.2f}%")
    print(f"  (How often we predict price direction correctly)")

    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (First 10 Test Set)")
    print("="*60)
    print(f"{'Actual':>12} {'Predicted':>12} {'Error':>12} {'% Error':>10}")
    print("-"*50)
    for i in range(min(10, len(y_test))):
        actual = y_test.iloc[i]
        predicted = pred_test[i]
        error = actual - predicted
        pct_error = (error / actual) * 100
        print(f"{actual:12.4f} {predicted:12.4f} {error:12.4f} {pct_error:9.2f}%")
if __name__ == "__main__":
    main()

