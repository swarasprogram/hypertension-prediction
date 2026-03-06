import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_model(df: pd.DataFrame, model_out_path: str = "models/hypertension_model.joblib"):
    target = "hypertension_stage"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(exclude=["int64", "float64"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ]
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=3000))
    ])

    # Simplified parameter grid for faster training
    param_grid = [
        {"model": [RandomForestClassifier(random_state=42)], "model__n_estimators": [100], "model__max_depth": [10]},
    ]

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    preds = best.predict(X_test)
    report = classification_report(y_test, preds)

    print("Best Params:", grid.best_params_)
    print(report)

    joblib.dump(best, model_out_path)
    return report, grid.best_params_