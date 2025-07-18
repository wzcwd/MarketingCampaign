"""Print top-N feature importances for tree-based models to the console."""

from typing import Dict, Any, Sequence
import numpy as np

__all__: Sequence[str] = ["feature_importance"]


def feature_importance(
    trained_models: Dict[str, Any],
    preprocessor,
    top_n: int = 5,
) -> None:
    """Print top-N importances for each model"""

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:  # pragma: no cover
        raise ValueError(
            "preprocessor must implement get_feature_names_out(); upgrade sklearn >=1.0"
        )

    for name, pipe in trained_models.items():
        est = pipe.named_steps.get("estimator")
        if est is None or not hasattr(est, "feature_importances_"):
            continue  # skip models without native importances

        importance = est.feature_importances_
        if len(importance) != len(feature_names):
            continue  # safety check failed

        idx_sorted = np.argsort(importance)[-top_n:][::-1]
        top_features = np.array(feature_names)[idx_sorted]
        top_scores = importance[idx_sorted]

        print(f"\n=== {name} — Top {top_n} features ===")
        for feat, score in zip(top_features, top_scores):
            print(f"{feat:<30s} {score:.4f}")
