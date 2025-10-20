import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeFilteringRecommender
from models.social import SocialRecommender
from models.hybrid import HybridRecommender
from utils.metrics import mean_average_precision_at_k


def main():
    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")

    print("Loading data...")
    events = pd.read_csv(processed_dir / "events_processed.csv")
    train = pd.read_csv(processed_dir / "train.csv")
    event_attendees = pd.read_csv(raw_dir / "event_attendees.csv")
    user_friends = pd.read_csv(processed_dir / "user_friends.csv")

    R = np.load(processed_dir / "R_matrix.npy")
    W = np.load(processed_dir / "W_matrix.npy")

    user_to_idx_df = pd.read_csv(processed_dir / "user_to_idx.csv")
    event_to_idx_df = pd.read_csv(processed_dir / "event_to_idx.csv")

    user_to_idx = dict(zip(user_to_idx_df["user"], user_to_idx_df["idx"]))
    event_to_idx = dict(zip(event_to_idx_df["event"], event_to_idx_df["idx"]))

    print("Initializing Content-Based model...")
    cb_model = ContentBasedRecommender(
        weight_purchase=3.0,
        weight_interested=1.0,
        temporal_decay=0.01
    )
    cb_model.fit(events, train, event_attendees)

    print("Initializing Collaborative Filtering model...")
    cf_model = CollaborativeFilteringRecommender(
        n_factors=20,
        regularization=0.01,
        iterations=15,
        random_state=42
    )
    cf_model.fit(R, W, user_to_idx, event_to_idx, train)

    print("Initializing Social Recommendation model...")
    social_model = SocialRecommender(
        weight_attending=2.0,
        weight_interested=1.0
    )
    social_model.fit(user_friends, train, event_attendees)

    print("Initializing Hybrid model...")
    hybrid_model = HybridRecommender(
        content_based_model=cb_model,
        collaborative_model=cf_model,
        social_model=social_model,
        weight_content=0.3,
        weight_collaborative=0.3,
        weight_social=0.4
    )

    print("Generating recommendations...")
    test_users = train["user"].unique()[:100]

    predictions = {}
    for user in test_users:
        predictions[user] = hybrid_model.recommend(user, n=200)

    actuals = {}
    for user in test_users:
        user_positives = train[(train["user"] == user) & (train["interested"] == 1)]["event"].tolist()
        actuals[user] = user_positives

    map_score = mean_average_precision_at_k(actuals, predictions, k=200)

    print(f"\nHybrid Model Results:")
    print(f"MAP@200: {map_score:.5f}")

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "hybrid_results.txt", "w") as f:
        f.write(f"MAP@200: {map_score:.5f}\n")


if __name__ == "__main__":
    main()
