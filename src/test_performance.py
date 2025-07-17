"""Evaluate trained classification models on a test set"""
from typing import Dict, Tuple
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

__all__ = ["evaluate_models"]


def evaluate_models(trained_models: Dict[str, object],
                    X_test, y_test, ) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Compute evaluation metrics for each trained model"""

    records = []
    # confusion matrix
    cm_dict: Dict[str, object] = {}

    for model_name, pipe in trained_models.items():
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        cm_dict[model_name] = cm

        roc_auc = None
        # Some estimators may not implement predict_proba (e.g., some trees if not set).
        if hasattr(pipe.named_steps["estimator"], "predict_proba"):
            try:
                y_proba = pipe.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            except Exception:  # pragma: no cover
                roc_auc = None

        records.append(
            {
                "Model": model_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "ROC_AUC": roc_auc,
            }
        )

    results_df = pd.DataFrame(records).set_index("Model").sort_values("Accuracy", ascending=False)
    return results_df, cm_dict
