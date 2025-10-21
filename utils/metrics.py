import numpy as np
from typing import List, Dict, Tuple


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


def recall_at_k(actual: List, predicted: List, k: int) -> float:
    if not actual:
        return 0.0

    predicted = predicted[:k]
    hits = len(set(actual) & set(predicted))

    return hits / len(actual)


def mean_recall_at_k(actuals: Dict[str, List], predictions: Dict[str, List], k: int) -> float:
    recall_scores = []

    for user in actuals:
        if user in predictions:
            recall = recall_at_k(actuals[user], predictions[user], k)
            recall_scores.append(recall)

    return np.mean(recall_scores) if recall_scores else 0.0


def hit_rate_at_k(actuals: Dict[str, List], predictions: Dict[str, List], k: int) -> float:
    hits = 0
    total = 0

    for user in actuals:
        if user in predictions:
            predicted = predictions[user][:k]
            if len(set(actuals[user]) & set(predicted)) > 0:
                hits += 1
            total += 1

    return hits / total if total > 0 else 0.0


def contamination_rate_at_k(
    predictions: Dict[str, List],
    not_interested: Dict[str, List],
    k: int
) -> float:
    contaminated = 0
    total_recommendations = 0

    for user in predictions:
        predicted = predictions[user][:k]
        if user in not_interested:
            contaminated += len(set(predicted) & set(not_interested[user]))
        total_recommendations += len(predicted)

    return contaminated / total_recommendations if total_recommendations > 0 else 0.0


def evaluate_recommendations(
    actuals: Dict[str, List],
    predictions: Dict[str, List],
    not_interested: Dict[str, List],
    k: int
) -> Dict[str, float]:
    return {
        "Recall@K": mean_recall_at_k(actuals, predictions, k),
        "Hit_Rate@K": hit_rate_at_k(actuals, predictions, k),
        "Contamination@K": contamination_rate_at_k(predictions, not_interested, k)
    }
