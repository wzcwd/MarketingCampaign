from typing import Dict
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier  # type: ignore

__all__ = ["train_models"]


def _build_estimators() -> Dict[str, object]:
    """Return a dict of model name -> estimator instance."""
    models: Dict[str, object] = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss")
    }
    return models


def train_models(preprocessor: Pipeline, X_train: pd.DataFrame,
                      y_train: pd.Series, ) -> Dict[str, Pipeline]:
    models = _build_estimators()
    trained_models: Dict[str, Pipeline] = {}

    for model_name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("estimator", estimator), ])
        pipe.fit(X_train, y_train)
        trained_models[model_name] = pipe

    return trained_models
