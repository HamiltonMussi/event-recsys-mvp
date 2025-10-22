import numpy as np
import pandas as pd
from typing import List
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from models.base import BaseRecommender
from utils.geo_filter import GeoFilter


class CollaborativeFilteringRecommender(BaseRecommender):
    def __init__(
        self,
        n_factors: int,
        regularization: float,
        iterations: int,
        random_state: int,
        geo_top_k: int
    ):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.geo_top_k = geo_top_k

        self.model = None
        self.user_to_idx = None
        self.event_to_idx = None
        self.idx_to_user = None
        self.idx_to_event = None
        self.interaction_matrix = None
        self.train = None
        self.events = None
        self.geo_filter = None

    def fit(self, R: np.ndarray, W: np.ndarray, user_to_idx: dict, event_to_idx: dict, train: pd.DataFrame, events: pd.DataFrame):
        self.user_to_idx = user_to_idx
        self.event_to_idx = event_to_idx
        self.idx_to_user = {i: u for u, i in user_to_idx.items()}
        self.idx_to_event = {i: e for e, i in event_to_idx.items()}
        self.train = train
        self.events = events

        self.geo_filter = GeoFilter(events, train)

        interaction_weighted = R * W
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

        candidate_events = self.geo_filter.get_nearby_events(
            user_id,
            top_k=self.geo_top_k,
            exclude_seen=exclude_seen
        )

        valid_candidates = []
        valid_indices = []
        for event in candidate_events:
            if event in self.event_to_idx:
                event_idx = self.event_to_idx[event]
                valid_candidates.append(event)
                valid_indices.append(event_idx)

        if not valid_indices:
            return []

        user_factor = self.model.user_factors[user_idx]
        event_factors = self.model.item_factors[valid_indices]

        scores = event_factors.dot(user_factor)

        top_positions = np.argsort(scores)[::-1][:n]

        return [valid_candidates[pos] for pos in top_positions]
