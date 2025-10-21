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
        n_features = self.event_embeddings.shape[1]

        self.user_embeddings = {}

        purchases = self.event_attendees[self.event_attendees["yes"].notna()][["event", "yes"]]
        purchase_pairs = purchases.assign(yes=purchases["yes"].str.split()).explode("yes")
        purchase_pairs = purchase_pairs.rename(columns={"yes": "user"})

        purchases_grouped = purchase_pairs.groupby("user")["event"].apply(list).to_dict()

        train_with_ts = self.train[self.train["interested"] == 1].copy()
        train_with_ts["timestamp"] = pd.to_datetime(train_with_ts["timestamp"], errors="coerce")
        reference_date = train_with_ts["timestamp"].max().timestamp()

        train_with_ts["timestamp_unix"] = train_with_ts["timestamp"].apply(
            lambda x: x.timestamp() if pd.notna(x) else reference_date
        )
        train_with_ts["days_since"] = (reference_date - train_with_ts["timestamp_unix"]) / 86400
        train_with_ts["decay"] = np.exp(-self.temporal_decay * train_with_ts["days_since"])

        interactions_grouped = train_with_ts.groupby("user")

        for user in users:
            weighted_embedding = np.zeros(n_features)
            total_weight = 0.0

            if user in interactions_grouped.groups:
                user_data = interactions_grouped.get_group(user)

                event_ids = user_data["event"].values
                event_indices = np.array([self.event_to_idx.get(e, -1) for e in event_ids])
                valid_mask = event_indices >= 0

                if valid_mask.any():
                    valid_indices = event_indices[valid_mask]
                    decays = user_data["decay"].values[valid_mask]
                    weights = self.weight_interested * decays

                    weighted_embedding += np.sum(
                        self.event_embeddings[valid_indices] * weights[:, np.newaxis],
                        axis=0
                    )
                    total_weight += np.sum(weights)

            if user in purchases_grouped:
                user_purchase_events = purchases_grouped[user]
                event_indices = np.array([self.event_to_idx.get(e, -1) for e in user_purchase_events])
                valid_mask = event_indices >= 0

                if valid_mask.any():
                    valid_indices = event_indices[valid_mask]
                    weight = self.weight_purchase * len(valid_indices)

                    weighted_embedding += np.sum(
                        self.event_embeddings[valid_indices] * self.weight_purchase,
                        axis=0
                    )
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
