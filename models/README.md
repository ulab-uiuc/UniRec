## Shared models

These modules are shared across data processing, training, evaluation, and inference.

- **`qformer.py`**
  - BERT-like Q-Former backbone (multi-layer Transformer with cross‑attention).
  - Used by both Item Q-Former and User Q-Former wrappers.

- **`qformer_model.py`**
  - Item Q-Former model wrapper built on top of `qformer.py`.
  - Provides a `QFormerForItemRepresentation` implementation used in some training/inference flows.

- **`qformer_utils.py`**
  - Provides:
    - `QFormerForItemRepresentation` (commonly used version).
    - `QFormerDataset` for building field-embedding datasets with caching.
    - `load_real_data(...)` and helper utilities for reading item samples.
  - Used by item Q-Former training, precompute scripts, inference scripts, and joint training.

- **`item_encoder_triplet.py`**
  - Defines an item encoder that can use CLIP and other modalities to produce field embeddings, optimized for triplet-style objectives.
  - Used by pipelines that operate on richer item representations.

- **`item_encoder_pure_value.py`**
  - Defines the general-purpose `ItemEncoder` class:
    - Loads a YAML config (`triplet_config.yaml`) to know which fields and encoders to use.
    - Uses CLIP/other encoders plus `mwne` utilities for numeric/time/geo features.
  - Used by Item Q-Former training, precomputation, inference, and sometimes user modeling.

- **`user_sequence_encoder.py`**
  - Turns user histories into dense sequences for user modeling:
    - Combines item-level features, timestamp features, and geolocation features.
    - Uses `ItemEncoder`, `TimestampEncoder`, `GeoCoordinateEncoder`, and positional encodings.
  - Used by `user_qformer_training.py` for building user sequence inputs.

- **`mwne.py`**
  - Contains encoders for non-text fields (for example, timestamp, coordinates).
  - Used by `item_encoder_pure_value.py` and `user_sequence_encoder.py` to augment items/users with temporal and spatial signals.

