## Evaluation

This folder contains scripts for evaluating trained models.

- **`evaluate_item_qformer.py`**
  - **What it evaluates**: the Item Q-Former.
  - **Metrics**:
    - Masked reconstruction MSE between original and reconstructed field embeddings.
    - Cosine similarity between original vs. reconstructed embeddings on valid (non-padded) fields.
  - **Flow**:
    1. Load a saved Item Q-Former checkpoint.
    2. Load cached validation embeddings and masks (from `embedding_cache_contrastive/val`).
    3. Run the model to reconstruct fields.
    4. Compute average reconstruction loss and cosine similarity and print a summary.
  - **Use this** to sanity‑check Item Q-Former training and to compare different checkpoints.

