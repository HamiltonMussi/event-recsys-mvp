import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on earth (in kilometers).
    """
    R = 6371.0

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


class GeoFilter:
    def __init__(self, events: pd.DataFrame, train: pd.DataFrame):
        self.events = events
        self.train = train
        self.user_locations = {}

        self._compute_user_locations()

    def _compute_user_locations(self):
        users = self.train["user"].unique()

        for user in users:
            user_events = self.train[self.train["user"] == user]["event"].values

            event_coords = self.events[self.events["event_id"].isin(user_events)][["lat", "lng"]]
            event_coords = event_coords.dropna()

            if len(event_coords) > 0:
                median_lat = event_coords["lat"].median()
                median_lng = event_coords["lng"].median()
                self.user_locations[user] = (median_lat, median_lng)
            else:
                self.user_locations[user] = None

    def get_nearby_events(
        self,
        user_id: str,
        top_k: int = 1000,
        exclude_seen: bool = True
    ) -> List:
        user_location = self.user_locations.get(user_id)

        if user_location is None:
            return self.events["event_id"].tolist()

        user_lat, user_lng = user_location

        valid_events = self.events[self.events["lat"].notna() & self.events["lng"].notna()].copy()

        lat_array = valid_events["lat"].values
        lng_array = valid_events["lng"].values

        R = 6371.0
        user_lat_rad = np.radians(user_lat)
        user_lng_rad = np.radians(user_lng)
        lat_rad = np.radians(lat_array)
        lng_rad = np.radians(lng_array)

        dlat = lat_rad - user_lat_rad
        dlng = lng_rad - user_lng_rad

        a = np.sin(dlat / 2)**2 + np.cos(user_lat_rad) * np.cos(lat_rad) * np.sin(dlng / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distances = R * c

        valid_events["distance"] = distances

        if exclude_seen:
            seen_events = set(self.train[self.train["user"] == user_id]["event"])
            valid_events = valid_events[~valid_events["event_id"].isin(seen_events)]

        nearby = valid_events.nsmallest(top_k, "distance")

        return nearby["event_id"].tolist()

    def get_user_location(self, user_id: str) -> Tuple[float, float]:
        return self.user_locations.get(user_id)
