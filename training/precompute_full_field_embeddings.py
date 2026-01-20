
import os
from models.item_encoder_pure_value import ItemEncoder
from models.qformer_utils import QFormerDataset, load_real_data

def precompute_full_dataset_embeddings(data_path, cache_dir, precompute_batch_size=8192, max_samples=None):
    """
    Pre-computes and caches field embeddings for the entire dataset.
    """
    print("🚀 Starting pre-computation for the full dataset...")
    
    # Load all item samples
    all_samples = load_real_data(data_path, max_samples=max_samples)
    print(f"Loaded {len(all_samples)} samples from {data_path}")
    
    # Initialize the item encoder
    item_encoder = ItemEncoder()
    
    # The QFormerDataset will handle the pre-computation and caching
    print(f"Using cache directory: {cache_dir}")
    _ = QFormerDataset(
        samples=all_samples,
        item_encoder=item_encoder,
        cache_dir=cache_dir,
        precompute_batch_size=precompute_batch_size
    )
    
    print("\n✅ Successfully pre-computed and cached field embeddings for the full dataset.")
    print(f"Cache saved in: {cache_dir}")

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "data_rec/dict/All_Beauty_item_triplet_dict.json"
    CACHE_DIR = "embedding_cache_contrastive/full"
    # For RTX A6000 (49GB VRAM), you can use a large batch size
    PRECOMPUTE_BATCH_SIZE = 8192 
    MAX_SAMPLES = None # Set to None to process all samples

    # Create the cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    precompute_full_dataset_embeddings(
        data_path=DATA_PATH,
        cache_dir=CACHE_DIR,
        precompute_batch_size=PRECOMPUTE_BATCH_SIZE,
        max_samples=MAX_SAMPLES
    )