import numpy as np
import pandas as pd
from typing import List, Dict, Set
from collections import defaultdict
from models.base import BaseRecommender


class SocialRecommender(BaseRecommender):
    def __init__(
        self,
        weight_attending: float,
        weight_interested: float
    ):
        self.weight_attending = weight_attending
        self.weight_interested = weight_interested

        self.user_friends = None
        self.train = None
        self.event_attendees = None
        self.friend_graph = None

    def fit(self, user_friends: pd.DataFrame, train: pd.DataFrame, event_attendees: pd.DataFrame):
        self.user_friends = user_friends
        self.train = train
        self.event_attendees = event_attendees

        self._build_friend_graph()

    def _build_friend_graph(self) -> Dict[str, Set[str]]:
        self.friend_graph = defaultdict(set)

        for _, row in self.user_friends.iterrows():
            user = row["user"]
            friends = str(row["friends"]).split()
            self.friend_graph[user].update(friends)

    def recommend(self, user_id: str, n: int = 200, exclude_seen: bool = True) -> List[str]:
        friends = self.friend_graph.get(user_id, set())

        if not friends:
            return []

        event_scores = defaultdict(float)

        friend_interested = self.train[
            (self.train["user"].isin(friends)) & (self.train["interested"] == 1)
        ]

        for _, row in friend_interested.iterrows():
            event_scores[row["event"]] += self.weight_interested

        purchases = self.event_attendees[self.event_attendees["yes"].notna()].copy()
        purchases["yes"] = purchases["yes"].str.split()
        purchase_pairs = purchases.explode("yes")[["event", "yes"]].rename(columns={"yes": "user"})

        friend_purchases = purchase_pairs[purchase_pairs["user"].isin(friends)]

        for _, row in friend_purchases.iterrows():
            event_scores[row["event"]] += self.weight_attending

        if exclude_seen:
            seen_events = set(self.train[self.train["user"] == user_id]["event"])
            for event in seen_events:
                event_scores.pop(event, None)

        sorted_events = sorted(event_scores.items(), key=lambda x: x[1], reverse=True)

        return [event for event, _ in sorted_events[:n]]
