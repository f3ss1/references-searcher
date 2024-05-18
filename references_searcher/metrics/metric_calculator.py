import numpy as np
import torch

from references_searcher import logger


class MetricCalculator:
    def __init__(self):
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0

    def update(
        self,
        predicted_classes: np.ndarray | torch.Tensor,
        actual_classes: np.ndarray | torch.Tensor,
    ) -> None:

        actual_positive = actual_classes == 1
        self.true_positive += int(((predicted_classes == actual_classes) & actual_positive).sum())
        self.true_negative += int(((predicted_classes == actual_classes) & ~actual_positive).sum())
        self.false_negative += int(((predicted_classes != actual_classes) & actual_positive).sum())
        self.false_positive += int(((predicted_classes != actual_classes) & ~actual_positive).sum())

    def calculate_metrics(self) -> dict:
        return {
            "accuracy": self._calculate_accuracy(),
            "precision": self._calculate_precision(),
            "recall": self._calculate_recall(),
            "f1": self._calculate_f1(),
        }

    def reset(self):
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0

    def _calculate_accuracy(self) -> float:
        denominator = self.true_positive + self.true_negative + self.false_negative + self.false_positive
        if denominator == 0:
            logger.warning("Detected division by zero when calculating accuracy score!")
            return 0
        numerator = self.true_positive + self.true_negative
        return numerator / denominator

    def _calculate_precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        if denominator == 0:
            logger.warning("Detected division by zero when calculating precision!")
            return 0
        return self.true_positive / denominator

    def _calculate_recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        if denominator == 0:
            logger.warning("Detected division by zero when calculating recall!")
            return 0
        return self.true_positive / denominator

    # TODO: ensure that the formula is correct just to be safe.
    def _calculate_f1(self) -> float:
        denominator = 2 * self.true_positive + self.false_negative + self.false_positive
        if denominator == 0:
            logger.warning("Detected division by zero when calculating f1 score!")
            return 0
        return 2 * self.true_positive / denominator
