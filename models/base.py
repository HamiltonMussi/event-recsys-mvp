from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def recommend(self, user_id: str, n: int = 200, exclude_seen: bool = True) -> List[str]:
        pass

    def recommend_batch(self, user_ids: List[str], n: int = 200) -> Dict[str, List[str]]:
        return {user_id: self.recommend(user_id, n) for user_id in user_ids}

    @staticmethod
    def min_max_normalize(scores: np.ndarray) -> np.ndarray:
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val == 0:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)
