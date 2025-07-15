"""Hyperparameter tuning for tree-based models using GridSearchCV."""
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier


__all__ = ["tune_models"]


def _param_grids() -> Dict[str, Dict[str, list[Any]]]:
    """Define parameter grids for each estimator."""
    grids: Dict[str, Dict[str, list[Any]]] = {
        "Decision Tree": {
            "estimator__max_depth": [None, 2, 3, 4, 5],
            "estimator__min_samples_split": [2, 3, 4, 5],
        },
       "Random Forest": {
            "estimator__n_estimators": [ 422, 423, 424, 425, 426, 430],
            "estimator__max_depth": [None, 2, 3, 4, 5],
        },
        "AdaBoost": {
            "estimator__n_estimators": [419, 420, 421, 438, 442],
            "estimator__learning_rate": [ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        },
         "XGBoost": {
            "estimator__n_estimators": [160, 165, 167, 168, 169],
            "estimator__max_depth": [2, 3, 4, 5, 6],
            "estimator__learning_rate": [0.07, 0.08, 0.09, 0.1, 0.2],
        }
    }
    return grids


def _estimators() -> Dict[str, Any]:
    ests: Dict[str, Any] = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
    }
    return ests


def tune_models(
    preprocessor: Pipeline,
    X_train,
    y_train,
    cv: int = 10,
    scoring: str = "accuracy",
) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    """Tune hyperparameters for each model and return best pipelines """

    grids = _param_grids()
    ests = _estimators()

    tuned: Dict[str, Pipeline] = {}
    records = []

    for name, est in ests.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("estimator", est),
        ])
        gs = GridSearchCV(
            pipe,
            param_grid=grids.get(name, {}),
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        tuned[name] = gs.best_estimator_
        records.append({"Model": name, "CV_Best": gs.best_score_, "Params": gs.best_params_})

    summary = pd.DataFrame(records).sort_values("CV_Best", ascending=False)
    return tuned, summary
