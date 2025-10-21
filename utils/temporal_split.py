import pandas as pd
import numpy as np
from typing import Tuple


def temporal_split_per_user(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    min_interactions: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally for each user individually.

    For each user:
    - Sort interactions by timestamp
    - Use first train_ratio% as train
    - Use remaining as validation

    Only includes users with at least min_interactions total.

    Args:
        df: DataFrame with columns [user, event, timestamp, interested, ...]
        train_ratio: Ratio of data to use for training (default 0.8)
        min_interactions: Minimum interactions per user (default 3)

    Returns:
        train_df, val_df
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')

    user_counts = df['user'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index

    df_filtered = df[df['user'].isin(valid_users)].copy()
    df_filtered = df_filtered.sort_values(['user', 'timestamp'])

    train_rows = []
    val_rows = []

    for user, user_data in df_filtered.groupby('user'):
        n = len(user_data)
        split_idx = int(n * train_ratio)

        if split_idx >= n:
            split_idx = n - 1

        train_rows.append(user_data.iloc[:split_idx])
        val_rows.append(user_data.iloc[split_idx:])

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = pd.concat(val_rows, ignore_index=True)

    return train_df, val_df


def print_split_stats(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Print statistics about the train/val split."""
    print("="*60)
    print("TEMPORAL SPLIT STATISTICS")
    print("="*60)

    print(f"\nTRAIN SET:")
    print(f"  Total interactions: {len(train_df)}")
    print(f"  Unique users: {train_df['user'].nunique()}")
    print(f"  Unique events: {train_df['event'].nunique()}")
    if 'interested' in train_df.columns:
        print(f"  Interested=1: {(train_df['interested']==1).sum()}")

    print(f"\nVALIDATION SET:")
    print(f"  Total interactions: {len(val_df)}")
    print(f"  Unique users: {val_df['user'].nunique()}")
    print(f"  Unique events: {val_df['event'].nunique()}")
    if 'interested' in val_df.columns:
        print(f"  Interested=1: {(val_df['interested']==1).sum()}")

    overlap_users = set(train_df['user']) & set(val_df['user'])
    print(f"\nOVERLAP:")
    print(f"  Users in both: {len(overlap_users)}")
    print(f"  Events in both: {len(set(train_df['event']) & set(val_df['event']))}")

    print("="*60)


if __name__ == "__main__":
    train_raw = pd.read_csv("data/raw/train.csv")
    train_df, val_df = temporal_split_per_user(train_raw, train_ratio=0.7, min_interactions=3)
    print_split_stats(train_df, val_df)
