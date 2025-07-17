from typing import Mapping
import os
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from numpy.typing import ArrayLike

__all__ = ["plot_roc_curves"]

def plot_roc_curves(
    models: Mapping[str, object],
    X_test: ArrayLike,
    y_test: ArrayLike,
    out_path: str,
    title: str | None = None,
) -> None:
    """Plot ROC curves for a collection of fitted pipelines."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))

    for name, pipe in models.items():
        if hasattr(pipe, "predict_proba"):
            y_score = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe, "decision_function"):
            y_score = pipe.decision_function(X_test)
        else:
            raise AttributeError(
                f"Model '{name}' lacks both predict_proba and decision_function."
            )

        RocCurveDisplay.from_predictions(y_test, y_score, name=name, ax=ax)

    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(title or "ROC Curves")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig) 