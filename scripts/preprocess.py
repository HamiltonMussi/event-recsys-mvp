import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.preprocessing import DataPreprocessor


def main():
    preprocessor = DataPreprocessor(
        raw_dir="data/raw",
        processed_dir="data/processed"
    )

    print("Starting data preprocessing...")
    preprocessor.preprocess(
        n_clusters=30,
        weight_purchase=100.0,
        weight_interested=10.0,
        weight_not_interested=1.0,
        weight_unseen=0.1
    )
    print("Preprocessing complete. Files saved to data/processed/")


if __name__ == "__main__":
    main()
