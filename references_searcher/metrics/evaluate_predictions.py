import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from collections.abc import Callable, Iterable

from references_searcher import logger


def evaluate_predictions(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    metrics: list[Callable[[np.ndarray, np.ndarray], float]] | Callable[[np.ndarray, np.ndarray], float] | None = None,
    set_name: str = "Train",
) -> dict:
    if metrics is None:
        metrics = [accuracy_score, precision_score, recall_score, f1_score]
    if not isinstance(metrics, Iterable):
        metrics = [metrics]

    result = {}

    scores_string = f"{set_name} scores. "

    for metric in metrics:
        metric_value = metric(y_true, y_pred)
        result[metric.__name__] = metric_value

        scores_string += f"{metric.__name__}: {metric_value:.3f}; "

    logger.info(scores_string)
    return result
