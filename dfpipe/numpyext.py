import numpy as np
from typing import Tuple

class ProbabilityMatrixTopkPredictions:
    def __init__(self, prob_matrix, labels):
        """_summary_

        Args:
            prob_matrix (np.ndarray): (samples, classes)
            labels (np.ndarray): (classes)
        """
        assert prob_matrix.shape[-1] == len(labels), "Prob matrix and labels should have the same number of columns"
        self.prob_matrix = prob_matrix
        self.labels = labels

    def topk_predictions(self, topk=1) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            topk (int, optional): _description_. Defaults to 1.

        Returns:
            Tuple[np.ndarray, np.ndarray]: topk_labels (samples, k), topk_proba (samples, k)
        """
        assert self.prob_matrix.shape[-1] >= topk, f"Prob matrix should have at least {topk} columns"
        topk_indices = np.argsort(self.prob_matrix, axis=1)[:, -topk:][:, ::-1]
        topk_labels = np.take(self.labels, topk_indices)
        topk_proba = self.prob_matrix[np.arange(self.prob_matrix.shape[0])[:, None], topk_indices]
        return topk_labels, topk_proba
