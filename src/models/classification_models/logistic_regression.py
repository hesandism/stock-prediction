from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

from src.models.baseline import time_split

from src.config import FEATURES_DIR

def main():
    path = FEATURES_DIR/"Toyota_features.csv"
    df = pd.read_csv(path)

    y_col = "target_direction"

    # removing target columns from X
    removing_cols = {"date", "target_next_close", "target_next_return", "target_direction"}

    X = df.drop(columns=[el for el in removing_cols if el in removing_cols ])
    y = df[y_col].astype(int)

    X = X.select_dtypes(include="number")
    
    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(columns=[y_col])
    y = data[y_col].astype(int)

    combined = pd.concat([X,y], axis=1)
    train_df, test_df = time_split(combined, train_ratio=0.8)

    X_train = train_df.drop(columns=[y_col])
    X_test = test_df.drop(columns=[y_col])
    
    y_train = train_df[y_col]
    y_test = test_df[y_col]


    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced"))
        ]
    )   

    clf.fit(X_train, y_train)

    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("===== Logistic Regression (Direction) =====")
    print("Train rows:", len(train_df), "Test rows:", len(test_df))
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

    
    if len(set(y_test)) == 2:
        print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

   


if __name__ == "__main__":
    main()


