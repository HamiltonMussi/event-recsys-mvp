import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from models.content_based import ContentBasedRecommender
from utils.metrics import mean_average_precision_at_k


def main():
    processed_dir = Path("data/processed")

    print("Loading processed data...")
    events = pd.read_csv(processed_dir / "events_processed.csv")
    train = pd.read_csv(processed_dir / "train.csv")
    event_attendees_path = Path("data/raw/event_attendees.csv")
    event_attendees = pd.read_csv(event_attendees_path)

    print("Initializing Content-Based model...")
    model = ContentBasedRecommender(
        weight_purchase=3.0,
        weight_interested=1.0,
        temporal_decay=0.01
    )

    print("Training model...")
    model.fit(events, train, event_attendees)

    print("Generating recommendations...")
    test_users = train["user"].unique()[:100]

    predictions = {}
    for user in test_users:
        predictions[user] = model.recommend(user, n=200)

    actuals = {}
    for user in test_users:
        user_positives = train[(train["user"] == user) & (train["interested"] == 1)]["event"].tolist()
        actuals[user] = user_positives

    map_score = mean_average_precision_at_k(actuals, predictions, k=200)

    print(f"\nContent-Based Filtering Results:")
    print(f"MAP@200: {map_score:.5f}")

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "content_based_results.txt", "w") as f:
        f.write(f"MAP@200: {map_score:.5f}\n")


if __name__ == "__main__":
    main()
