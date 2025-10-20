import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from models.social import SocialRecommender
from utils.metrics import mean_average_precision_at_k


def main():
    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")

    print("Loading data...")
    train = pd.read_csv(processed_dir / "train.csv")
    user_friends = pd.read_csv(processed_dir / "user_friends.csv")
    event_attendees = pd.read_csv(raw_dir / "event_attendees.csv")

    print("Initializing Social Recommendation model...")
    model = SocialRecommender(
        weight_attending=2.0,
        weight_interested=1.0
    )

    print("Training model...")
    model.fit(user_friends, train, event_attendees)

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

    print(f"\nSocial Recommendation Results:")
    print(f"MAP@200: {map_score:.5f}")

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "social_results.txt", "w") as f:
        f.write(f"MAP@200: {map_score:.5f}\n")


if __name__ == "__main__":
    main()
