import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from models.collaborative import CollaborativeFilteringRecommender
from utils.metrics import mean_average_precision_at_k


def main():
    processed_dir = Path("data/processed")

    print("Loading processed data...")
    train = pd.read_csv(processed_dir / "train.csv")
    R = np.load(processed_dir / "R_matrix.npy")
    W = np.load(processed_dir / "W_matrix.npy")

    user_to_idx_df = pd.read_csv(processed_dir / "user_to_idx.csv")
    event_to_idx_df = pd.read_csv(processed_dir / "event_to_idx.csv")

    user_to_idx = dict(zip(user_to_idx_df["user"], user_to_idx_df["idx"]))
    event_to_idx = dict(zip(event_to_idx_df["event"], event_to_idx_df["idx"]))

    print("Initializing Collaborative Filtering model...")
    model = CollaborativeFilteringRecommender(
        n_factors=20,
        regularization=0.01,
        iterations=15,
        random_state=42
    )

    print("Training model...")
    model.fit(R, W, user_to_idx, event_to_idx, train)

    print("Generating recommendations...")
    test_users = list(user_to_idx.keys())[:100]

    predictions = {}
    for user in test_users:
        predictions[user] = model.recommend(user, n=200)

    actuals = {}
    for user in test_users:
        user_positives = train[(train["user"] == user) & (train["interested"] == 1)]["event"].tolist()
        actuals[user] = user_positives

    map_score = mean_average_precision_at_k(actuals, predictions, k=200)

    print(f"\nCollaborative Filtering Results:")
    print(f"MAP@200: {map_score:.5f}")

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "collaborative_results.txt", "w") as f:
        f.write(f"MAP@200: {map_score:.5f}\n")


if __name__ == "__main__":
    main()
