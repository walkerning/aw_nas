from typing import List

from scipy.stats import stats
import numpy as np
from torch.utils.data import Dataset


class NasBenchDataset(Dataset):
    """
    Dataset for NAS-Bench-201 architecture-performance pairs.
    """
    def __init__(self, data: list) -> None:
        self.data = data
        self._len = len(self.data)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        data = self.data[idx]
        return data


def test_xk(
    true_scores: List[float],
    predict_scores: List[float],
    ratios: List[float] = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
) -> List:
    """
    Calculate p@topK.

    Args:
        true_scores (List[float]): Architectures' actual performances.
        predict_scores (List[float]): Predicted scores of the architectures.
        ratios (List[float]): Top ratios to calculate. Default: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    """
    true_inds = np.argsort(true_scores)[::-1]
    true_scores = np.array(true_scores)
    reorder_true_scores = true_scores[true_inds]
    predict_scores = np.array(predict_scores)
    reorder_predict_scores = predict_scores[true_inds]
    ranks = np.argsort(reorder_predict_scores)[::-1]
    num_archs = len(ranks)
    patks = []
    for ratio in ratios:
        k = int(num_archs * ratio)
        p = len(np.where(ranks[:k] < k)[0]) / float(k)
        arch_inds = ranks[:k][ranks[:k] < k]
        patks.append((k, ratio, len(arch_inds), p, stats.kendalltau(
            reorder_true_scores[arch_inds],
            reorder_predict_scores[arch_inds]).correlation))
    return patks
