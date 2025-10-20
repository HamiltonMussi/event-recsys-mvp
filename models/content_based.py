import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from models.base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    def __init__(
        self,
        weight_purchase: float,
        weight_interested: float,
        temporal_decay: float
    ):
        self.weight_purchase = weight_purchase
        self.weight_interested = weight_interested
        self.temporal_decay = temporal_decay
        self.scaler = StandardScaler()

        self.events = None
        self.train = None
        self.event_attendees = None
        self.event_embeddings = None
        self.user_embeddings = None
        self.event_to_idx = None
        self.idx_to_event = None

    def fit(self, events: pd.DataFrame, train: pd.DataFrame, event_attendees: pd.DataFrame):
        self.events = events
        self.train = train
        self.event_attendees = event_attendees

        self.event_to_idx = {e: i for i, e in enumerate(events["event_id"])}
        self.idx_to_event = {i: e for e, i in self.event_to_idx.items()}

        self._build_event_embeddings()
        self._build_user_embeddings()

    def _build_event_embeddings(self):
        cat_features = pd.get_dummies(self.events["event_category"], prefix="cat")
        num_features = self.events[["hour", "weekday"]].fillna(0)
        num_features_scaled = self.scaler.fit_transform(num_features)

        self.event_embeddings = np.hstack([
            cat_features.values,
            num_features_scaled
        ])

    def _build_user_embeddings(self):
        users = self.train["user"].unique()
        n_users = len(users)
        n_features = self.event_embeddings.shape[1]

        self.user_embeddings = {}

        purchases = self.event_attendees[self.event_attendees["yes"].notna()].copy()
        purchases["yes"] = purchases["yes"].str.split()
        purchase_pairs = purchases.explode("yes")[["event", "yes"]].rename(columns={"yes": "user"})

        train_copy = self.train.copy()
        train_copy["timestamp"] = pd.to_datetime(train_copy["timestamp"], errors="coerce")
        reference_date = train_copy["timestamp"].max().timestamp()

        for user in users:
            user_interactions = train_copy[
                (train_copy["user"] == user) & (train_copy["interested"] == 1)
            ].copy()

            user_purchases = purchase_pairs[purchase_pairs["user"] == user].copy()

            weighted_embedding = np.zeros(n_features)
            total_weight = 0.0

            for _, row in user_interactions.iterrows():
                event_idx = self.event_to_idx.get(row["event"])
                if event_idx is None:
                    continue

                timestamp = row["timestamp"].timestamp() if pd.notna(row["timestamp"]) else reference_date
                days_since = (reference_date - timestamp) / 86400
                decay = np.exp(-self.temporal_decay * days_since)
                weight = self.weight_interested * decay

                weighted_embedding += weight * self.event_embeddings[event_idx]
                total_weight += weight

            for _, row in user_purchases.iterrows():
                event_idx = self.event_to_idx.get(row["event"])
                if event_idx is None:
                    continue

                weight = self.weight_purchase

                weighted_embedding += weight * self.event_embeddings[event_idx]
                total_weight += weight

            if total_weight > 0:
                self.user_embeddings[user] = weighted_embedding / total_weight
            else:
                self.user_embeddings[user] = np.zeros(n_features)

    def recommend(self, user_id: str, n: int = 200, exclude_seen: bool = True) -> List[str]:
        user_emb = self.user_embeddings.get(user_id)
        if user_emb is None:
            return []

        similarities = cosine_similarity([user_emb], self.event_embeddings)[0]

        if exclude_seen:
            seen_events = set(self.train[self.train["user"] == user_id]["event"])
            for event in seen_events:
                event_idx = self.event_to_idx.get(event)
                if event_idx is not None:
                    similarities[event_idx] = -np.inf

        top_indices = np.argsort(similarities)[::-1][:n]
        return [self.idx_to_event[idx] for idx in top_indices]
