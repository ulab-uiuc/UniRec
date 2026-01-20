## Training scripts

These scripts train each stage of the nested Q-Former + Qwen3 LoRA architecture.

- **`item_qformer_training.py`**
  - **Goal**: train the Item Q-Former so that it:
    - Reconstructs item field embeddings (reconstruction loss).
    - Respects sequence relationships between items (triplet/contrastive loss).
  - **Flow**:
    1. Load item samples with `load_real_data`.
    2. Build a `QFormerTripletDataset` that creates (anchor, positive, negative) triplets from item sequences.
    3. Train `QFormerForItemRepresentation` with a combined reconstruction + contrastive objective.
    4. Periodically evaluate reconstruction quality and save the best checkpoint.

- **`precompute_full_field_embeddings.py`**
  - **Goal**: precompute and cache all item field embeddings (before Q-Former).
  - **Flow**:
    1. Use `load_real_data` to read all items.
    2. Use `ItemEncoder` to compute per-field embeddings.
    3. Let `QFormerDataset` cache these to disk (for example, under `embedding_cache_contrastive/...`).
  - **Why**: dramatically speeds up repeated Item Q-Former training/evaluation and inference.

- **`user_qformer_training.py`**
  - **Goal**: train a User Q-Former that consumes sequences of item query tokens and outputs user query tokens.
  - **Flow**:
    1. Use `UserSequenceEncoder` to build sequences for each user (history of items).
    2. Feed sequences into a `UserQFormer` (a Q-Former configured for user-level cross-attention).
    3. Train the model to predict next-item query tokens or a user embedding suitable for downstream ranking.

- **`train_item_individual_token_joint.py`**
  - **Goal**: joint training of Qwen3 + LoRA with the nested Q-Former stack.
  - **High-level idea**:
    1. Precompute item query tokens (via Item Q-Former) and possibly user-side query tokens.
    2. Reserve special tokens in Qwen3’s tokenizer to mark where these query tokens should appear.
    3. Inject the learned query-token embeddings directly into Qwen3’s input embedding matrix at those positions.
    4. Train Qwen3+LoRA so its final pooled embedding matches the ground-truth item embeddings, using an InfoNCE-style ranking loss over positive vs. negative candidates.
  - **When to run**:
    - After you have:
      - An item query-token cache from `generate_all_item_embeddings.py`.
      - A trained Item Q-Former checkpoint.
      - Candidate item embedding JSON (for example, `all_beauty_item_embedding_qwen3_0.6B.json`).

