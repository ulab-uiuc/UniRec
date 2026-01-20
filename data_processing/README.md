## Data processing

These scripts build the data and embeddings needed before training any models.

- **`create_item_dict.py`**
  - Builds a JSON dictionary mapping `item_id` → item metadata (title, image URLs, etc.).
  - Run once per dataset; other scripts read this file to know the fields available for each item.

- **`create_review_dict.py`**
  - Builds a dictionary mapping reviews/review IDs to their text (and possibly metadata).
  - Used by review embedding and item encoders that incorporate review text.

- **`create_triplet_dict.py`**
  - Builds an item-level triplet dictionary (anchor, positive, negative semantics).
  - Used by item encoders and Item Q-Former training scripts that rely on triplet-based objectives.

- **`process_rec_new_user.py`** / **`process_rec_old_user.py`**
  - Generate train/test JSON files for recommendation:
    - New users vs. existing users.
    - Include user history, ground-truth target items, and candidate sets.
  - Outputs files under `data_rec/data/...` (for example, `Amazon_All_Beauty_all_train_LRanker.json` and `..._20_train.json`).

- **`item_embedding_clip.py`**
  - Computes CLIP embeddings for each item using its text and/or image fields.
  - Inputs: item dict + train/test data (to know which `item_id`s appear).
  - Outputs: a JSON file under `data_rec/embeddings/...` mapping `item_id` → CLIP embedding.

- **`review_embedding_clip.py`**
  - Computes CLIP embeddings over review text (if you incorporate reviews into item representation).
  - Inputs: review dict + item dict + train data.
  - Outputs: JSON file of review embeddings and/or aggregated item-level review features.

- **`qformer_inference.py`**
  - Loads a trained Item Q-Former checkpoint and runs fast inference over cached item field embeddings.
  - Outputs item-level query tokens (for example, `num_query_tokens × hidden_dim`) for each `item_id`.

- **`generate_all_item_embeddings.py`**
  - Orchestrates batch generation of item query tokens for all items:
    - Loads the item dict.
    - Uses `qformer_inference.py` to run the Item Q-Former in batches.
    - Saves a JSON / pickle of `item_id` → query tokens.
  - Run this after training the Item Q-Former; the resulting query-token cache is consumed by user/Qwen training.

