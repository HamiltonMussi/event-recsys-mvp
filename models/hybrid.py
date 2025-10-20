import numpy as np
from typing import List, Dict
from collections import defaultdict
from models.base import BaseRecommender


class HybridRecommender(BaseRecommender):
    def __init__(
        self,
        content_based_model: BaseRecommender,
        collaborative_model: BaseRecommender,
        social_model: BaseRecommender,
        weight_content: float,
        weight_collaborative: float,
        weight_social: float
    ):
        self.cb_model = content_based_model
        self.cf_model = collaborative_model
        self.social_model = social_model

        self.weight_content = weight_content
        self.weight_collaborative = weight_collaborative
        self.weight_social = weight_social

    def fit(self, *args, **kwargs):
        pass

    def recommend(self, user_id: str, n: int = 200, exclude_seen: bool = True) -> List[str]:
        cb_recs = self.cb_model.recommend(user_id, n=n * 2, exclude_seen=exclude_seen)
        cf_recs = self.cf_model.recommend(user_id, n=n * 2, exclude_seen=exclude_seen)
        social_recs = self.social_model.recommend(user_id, n=n * 2, exclude_seen=exclude_seen)

        event_scores = defaultdict(float)

        cb_scores = self._rank_to_scores(cb_recs)
        cf_scores = self._rank_to_scores(cf_recs)
        social_scores = self._rank_to_scores(social_recs)

        cb_scores_norm = self._normalize_scores(cb_scores)
        cf_scores_norm = self._normalize_scores(cf_scores)
        social_scores_norm = self._normalize_scores(social_scores)

        for event, score in cb_scores_norm.items():
            event_scores[event] += self.weight_content * score

        for event, score in cf_scores_norm.items():
            event_scores[event] += self.weight_collaborative * score

        for event, score in social_scores_norm.items():
            event_scores[event] += self.weight_social * score

        sorted_events = sorted(event_scores.items(), key=lambda x: x[1], reverse=True)

        return [event for event, _ in sorted_events[:n]]

    def _rank_to_scores(self, recommendations: List[str]) -> Dict[str, float]:
        scores = {}
        n = len(recommendations)
        for i, event in enumerate(recommendations):
            scores[event] = n - i
        return scores

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}

        values = np.array(list(scores.values()))
        min_val = values.min()
        max_val = values.max()

        if max_val - min_val == 0:
            return {k: 0.0 for k in scores}

        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}
