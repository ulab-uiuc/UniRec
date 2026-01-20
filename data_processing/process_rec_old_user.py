"""
Movie Recommendation Dataset Processing Script (Amazon Only)

This script processes Amazon movie recommendation data to create training and evaluation samples
for sequential recommendation tasks. It handles data loading, formatting, and generates
various types of prompts for different evaluation scenarios.
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple, Any
import json
from sklearn.model_selection import train_test_split
import copy

def load_data(data_source: str) -> pd.DataFrame:
    """Load interaction data from Amazon movie dataset files.
    
    Args:
        data_source: e.g., 'Amazon_All_Beauty'
    
    Returns:
        DataFrame containing user interactions with items
    """
    # Load interaction data
    interactions = []
    with open(f'data_rec/{data_source}.inter', 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                user_id = str(parts[0])
                item_id = parts[1]
                rating = float(parts[2])
                timestamp = int(parts[3])
                interactions.append((user_id, item_id, rating, timestamp))
    # Convert to DataFrame and sort by timestamp
    df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'rating', 'timestamp'])
    df = df.sort_values(['user_id', 'timestamp'])
    return df

def create_train_test_samples(df: pd.DataFrame,
                               num_candidates: int = 100) -> Tuple[List[Dict], List[Dict]]:
    """Create train and test samples for sequential recommendation."""
    # Get users with more than 22 interactions
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts > 12].index.tolist()

    train_samples = []
    test_samples = []
    all_items = set(df['item_id'].unique())

    for user_id in valid_users:
        # Get user's interactions
        user_interactions = df[df['user_id'] == user_id].sort_values('timestamp')
        item_sequence = user_interactions['item_id'].tolist()

        # Create training sample: history is [-22:-2], ground truth is the second to last item
        train_history = item_sequence[-22:-2]
        train_ground_truth = item_sequence[-2]

        # Sample negative candidates for training
        excluded_items = set(train_history + [train_ground_truth])
        available_items = list(all_items - excluded_items)
        if len(available_items) < num_candidates - 1:
            continue
        negative_candidates = random.sample(available_items, num_candidates - 1)

        # Combine ground truth and negative candidates for training
        train_candidates = [train_ground_truth] + negative_candidates
        random.shuffle(train_candidates)

        train_sample = {
            'user_id': user_id,
            'history': train_history,
            'candidate': train_candidates,
            'ground_truth': train_ground_truth
        }
        train_samples.append(train_sample)

        # Create test sample: history is [-21:-1], ground truth is the last item
        test_history = item_sequence[-21:-1]
        test_ground_truth = item_sequence[-1]

        # Sample negative candidates for testing
        excluded_items = set(test_history + [test_ground_truth])
        available_items = list(all_items - excluded_items)
        if len(available_items) < num_candidates - 1:
            continue
        negative_candidates = random.sample(available_items, num_candidates - 1)

        # Combine ground truth and negative candidates for testing
        test_candidates = [test_ground_truth] + negative_candidates
        random.shuffle(test_candidates)

        test_sample = {
            'user_id': user_id,
            'history': test_history,
            'candidate': test_candidates,
            'ground_truth': test_ground_truth
        }
        test_samples.append(test_sample)

    return train_samples, test_samples


def process_samples(samples: List[Dict]) -> List[Dict]:
    """Process samples to create output with item IDs only and required field names."""
    data_all = []
    
    for sample in samples:
        all_sample = {
            'user_id': str(sample['user_id']),
            'history': sample['history'],
            'candidate': sample['candidate'],
            'ground_truth': sample['ground_truth']
        }
        data_all.append(all_sample)
    
    return data_all

def main():
    """Main execution function."""
    # Only process Amazon_All_Beauty (or other specified dataset)
    data_sources = ['Amazon_All_Beauty']
    
    for data_source in data_sources:
        try:
            # Load data
            df = load_data(data_source)
            
            # Create samples
            train_samples, test_samples = create_train_test_samples(df)
            
            # Process train samples (IDs only)
            train_all = process_samples(train_samples)
            
            # Process test samples (IDs only)
            test_all = process_samples(test_samples)
            
            # Save the processed data (IDs only)
            with open(f'data_rec/data/{data_source}_20_train.json', 'w', encoding='utf-8') as f:
                json.dump(train_all, f, ensure_ascii=False, indent=4)
            
            with open(f'data_rec/data/{data_source}_20_test.json', 'w', encoding='utf-8') as f:
                json.dump(test_all, f, ensure_ascii=False, indent=4)
            
            print(f"Created {len(train_all)} train all samples for {data_source}")
            print(f"Created {len(test_all)} test all samples for {data_source}")
            if train_samples:
                print(f"Each train sample contains {len(train_samples[0]['history'])} historical items and {len(train_samples[0]['candidate'])} candidate items")
            if test_samples:
                print(f"Each test sample contains {len(test_samples[0]['history'])} historical items and {len(test_samples[0]['candidate'])} candidate items")
            
        except Exception as e:
            print(f"Error processing {data_source}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 