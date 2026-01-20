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

def create_sequential_samples(df: pd.DataFrame, 
                           num_samples: int = 290,
                           hist_len: int = 10,
                           num_candidates: int = 100) -> List[Dict]:
    """Create sequential recommendation samples."""
    # Get users with at least hist_len + 1 interactions
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= hist_len + 1].index.tolist()
    
    if len(valid_users) < num_samples:
        raise ValueError(f"Not enough users with sufficient interactions. Found {len(valid_users)} users, need {num_samples}")
    
    # Randomly select users
    selected_users = random.sample(valid_users, num_samples)
    samples = []
    all_items = set(df['item_id'].unique())
    
    for user_id in selected_users:
        # Get user's interactions
        user_interactions = df[df['user_id'] == user_id].sort_values('timestamp')
        item_sequence = user_interactions['item_id'].tolist()
        
        # Get history and ground truth
        history = item_sequence[:hist_len]
        ground_truth = item_sequence[hist_len]
        
        # Sample negative candidates
        excluded_items = set(history + [ground_truth])
        available_items = list(all_items - excluded_items)
        negative_candidates = random.sample(available_items, num_candidates - 1)
        
        # Combine ground truth and negative candidates
        candidates = [ground_truth] + negative_candidates
        random.shuffle(candidates)  # Shuffle to avoid position bias
        
        # Create sample (IDs only)
        sample = {
            'user_id': user_interactions['user_id'].iloc[0],
            'history': history,
            'candidate': candidates,
            'ground_truth': ground_truth
        }
        samples.append(sample)
    
    return samples

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
            samples = create_sequential_samples(df)
            
            # Split into train and test
            train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)
            
            # Process train samples (IDs only)
            train_all = process_samples(train_samples)
            
            # Process test samples (IDs only)
            test_all = process_samples(test_samples)
            
            # Save the processed data (IDs only)
            with open(f'data_rec/data/{data_source}_all_train_LRanker.json', 'w', encoding='utf-8') as f:
                json.dump(train_all, f, ensure_ascii=False, indent=4)
            
            with open(f'data_rec/data/{data_source}_all_test_LRanker.json', 'w', encoding='utf-8') as f:
                json.dump(test_all, f, ensure_ascii=False, indent=4)
            
            print(f"Created {len(train_all)} train all samples for {data_source}")
            print(f"Created {len(test_all)} test all samples for {data_source}")
            print(f"Each sample contains {len(samples[0]['history'])} historical items and {len(samples[0]['candidate'])} candidate items")
            
        except Exception as e:
            print(f"Error processing {data_source}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 