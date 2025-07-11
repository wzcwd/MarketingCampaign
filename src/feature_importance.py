"""Plot top-N feature importance for tree-based models"""
import os
from typing import Dict, Any, Sequence
import matplotlib.pyplot as plt
import numpy as np

__all__: Sequence[str] = ["feature_importance"]


def feature_importance(
        trained_models: Dict[str, Any],
        preprocessor,
        out_dir: str,
        top_n: int = 10,) -> None:
    """Generate and save feature-importance plots"""

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:  # pragma: no cover
        raise ValueError(
            "preprocessor must implement get_feature_names_out(); upgrade sklearn >=1.0"
        )

    for name, pipe in trained_models.items():
        est = pipe.named_steps.get("estimator", None)
        if est is None or not hasattr(est, "feature_importances_"):
            # Skip models without native feature importance
            continue

        importance = est.feature_importances_
        if len(importance) != len(feature_names):
            # Safety check: lengths must match
            continue

        idx_sorted = np.argsort(importance)[-top_n:][::-1]
        top_features = np.array(feature_names)[idx_sorted]
        top_scores = importance[idx_sorted]

        # Generate images
        plt.figure(figsize=(6, 4))
        plt.barh(range(len(top_features))[::-1], top_scores[::-1], color="skyblue")
        plt.yticks(range(len(top_features))[::-1], top_features[::-1])
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Features — {name}")
        plt.tight_layout()
        file_safe = name.lower().replace(" ", "_")
        plt.savefig(os.path.join(out_dir, f"fm_{file_safe}.png"))
        plt.close()
