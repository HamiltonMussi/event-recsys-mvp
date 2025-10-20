import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans
from typing import Tuple


class DataLoader:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    def load_all(self) -> Tuple[pd.DataFrame, ...]:
        train = pd.read_csv(self.data_dir / "train.csv")
        events = pd.read_csv(self.data_dir / "events.csv")
        users = pd.read_csv(self.data_dir / "users.csv")
        user_friends = pd.read_csv(self.data_dir / "user_friends.csv")
        event_attendees = pd.read_csv(self.data_dir / "event_attendees.csv")

        return train, events, users, user_friends, event_attendees


class EventFeatureExtractor:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.kmeans = None

    def fit_transform(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()

        c_cols = [f"c_{i}" for i in range(1, 101)] + ["c_other"]
        c_features = events[c_cols].fillna(0)

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        events["event_category"] = self.kmeans.fit_predict(c_features)

        events["start_time"] = pd.to_datetime(events["start_time"], errors="coerce")
        events["hour"] = events["start_time"].dt.hour
        events["weekday"] = events["start_time"].dt.weekday

        return events

    def transform(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()

        c_cols = [f"c_{i}" for i in range(1, 101)] + ["c_other"]
        c_features = events[c_cols].fillna(0)

        events["event_category"] = self.kmeans.predict(c_features)
        events["start_time"] = pd.to_datetime(events["start_time"], errors="coerce")
        events["hour"] = events["start_time"].dt.hour
        events["weekday"] = events["start_time"].dt.weekday

        return events


class InteractionMatrix:
    def __init__(
        self,
        weight_purchase: float,
        weight_interested: float,
        weight_not_interested: float,
        weight_unseen: float
    ):
        self.weight_purchase = weight_purchase
        self.weight_interested = weight_interested
        self.weight_not_interested = weight_not_interested
        self.weight_unseen = weight_unseen

    def build_matrices(
        self,
        train: pd.DataFrame,
        event_attendees: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, dict, dict]:

        users = sorted(train["user"].unique())
        events = sorted(train["event"].unique())

        user_to_idx = {u: i for i, u in enumerate(users)}
        event_to_idx = {e: i for i, e in enumerate(events)}

        n_users = len(users)
        n_events = len(events)

        R = np.zeros((n_users, n_events))
        W = np.full((n_users, n_events), self.weight_unseen)

        purchases = event_attendees[event_attendees["yes"].notna()].copy()
        purchases["yes"] = purchases["yes"].str.split()
        purchase_pairs = purchases.explode("yes")[["event", "yes"]].rename(columns={"yes": "user"})

        for _, row in train.iterrows():
            u_idx = user_to_idx.get(row["user"])
            e_idx = event_to_idx.get(row["event"])

            if u_idx is None or e_idx is None:
                continue

            if row["interested"] == 1:
                R[u_idx, e_idx] = 1
                W[u_idx, e_idx] = self.weight_interested
            elif row["not_interested"] == 1:
                R[u_idx, e_idx] = 0
                W[u_idx, e_idx] = self.weight_not_interested

        for _, row in purchase_pairs.iterrows():
            u_idx = user_to_idx.get(row["user"])
            e_idx = event_to_idx.get(row["event"])

            if u_idx is not None and e_idx is not None:
                R[u_idx, e_idx] = 1
                W[u_idx, e_idx] = self.weight_purchase

        return R, W, user_to_idx, event_to_idx


class DataPreprocessor:
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(
        self,
        n_clusters: int,
        weight_purchase: float,
        weight_interested: float,
        weight_not_interested: float,
        weight_unseen: float
    ):
        loader = DataLoader(self.raw_dir)
        train, events, users, user_friends, event_attendees = loader.load_all()

        extractor = EventFeatureExtractor(n_clusters=n_clusters)
        events = extractor.fit_transform(events)

        matrix_builder = InteractionMatrix(
            weight_purchase=weight_purchase,
            weight_interested=weight_interested,
            weight_not_interested=weight_not_interested,
            weight_unseen=weight_unseen
        )
        R, W, user_to_idx, event_to_idx = matrix_builder.build_matrices(train, event_attendees)

        events.to_csv(self.processed_dir / "events_processed.csv", index=False)
        users.to_csv(self.processed_dir / "users_processed.csv", index=False)
        user_friends.to_csv(self.processed_dir / "user_friends.csv", index=False)
        train.to_csv(self.processed_dir / "train.csv", index=False)

        np.save(self.processed_dir / "R_matrix.npy", R)
        np.save(self.processed_dir / "W_matrix.npy", W)

        pd.DataFrame(list(user_to_idx.items()), columns=["user", "idx"]).to_csv(
            self.processed_dir / "user_to_idx.csv", index=False
        )
        pd.DataFrame(list(event_to_idx.items()), columns=["event", "idx"]).to_csv(
            self.processed_dir / "event_to_idx.csv", index=False
        )
