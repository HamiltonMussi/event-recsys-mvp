import numpy as np
import pandas as pd
from typing import List
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from models.base import BaseRecommender


class CollaborativeFilteringRecommender(BaseRecommender):
    def __init__(
        self,
        n_factors: int,
        regularization: float,
        iterations: int,
        random_state: int
    ):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state

        self.model = None
        self.user_to_idx = None
        self.event_to_idx = None
        self.idx_to_user = None
        self.idx_to_event = None
        self.interaction_matrix = None
        self.train = None

    def fit(self, R: np.ndarray, W: np.ndarray, user_to_idx: dict, event_to_idx: dict, train: pd.DataFrame):
        self.user_to_idx = user_to_idx
        self.event_to_idx = event_to_idx
        self.idx_to_user = {i: u for u, i in user_to_idx.items()}
        self.idx_to_event = {i: e for e, i in event_to_idx.items()}
        self.train = train

        interaction_weighted = (R * W).T
        self.interaction_matrix = csr_matrix(interaction_weighted)

        self.model = AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state
        )

        self.model.fit(self.interaction_matrix)

    def recommend(self, user_id: str, n: int = 200, exclude_seen: bool = True) -> List[str]:
        user_idx = self.user_to_idx.get(user_id)
        if user_idx is None:
            return []

        if exclude_seen:
            seen_events = set(self.train[self.train["user"] == user_id]["event"])
            filter_items = [self.event_to_idx[e] for e in seen_events if e in self.event_to_idx]
        else:
            filter_items = []

        recommendations = self.model.recommend(
            user_idx,
            self.interaction_matrix[user_idx],
            N=n + len(filter_items),
            filter_already_liked_items=False
        )

        event_indices = [idx for idx, _ in recommendations]

        if exclude_seen:
            filter_set = set(filter_items)
            event_indices = [idx for idx in event_indices if idx not in filter_set]

        event_indices = event_indices[:n]

        return [self.idx_to_event[idx] for idx in event_indices]
