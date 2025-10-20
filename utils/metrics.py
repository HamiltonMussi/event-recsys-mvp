import numpy as np
from typing import List, Dict


def average_precision_at_k(actual: List, predicted: List, k: int) -> float:
    if not actual:
        return 0.0

    predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mean_average_precision_at_k(actuals: Dict[str, List], predictions: Dict[str, List], k: int) -> float:
    ap_scores = []

    for user in actuals:
        if user in predictions:
            ap = average_precision_at_k(actuals[user], predictions[user], k)
            ap_scores.append(ap)

    return np.mean(ap_scores) if ap_scores else 0.0
