## Repo overview

This repo demonstrates a **nested Q-Former + Qwen3 LoRA** recommendation stack:

1. **Item encoder + Item Q-Former**  
   Raw item fields (text, CLIP features, etc.) → dense field embeddings → **item query tokens**.
2. **User Q-Former**  
   User history as a sequence of item query tokens → **user query tokens**.
3. **Qwen3 + LoRA joint model**  
   Injects item/user query tokens as *special tokens* in Qwen3, then uses the final embedding as a **predicted next-item embedding** for ranking a candidate pool.

All scripts here are copies from the original project, reorganized into a GitHub‑friendly layout (no hardcoded API keys or absolute cluster paths).

---

## Folder structure

- **`data_processing/`** – build dicts, process recommendation data, generate CLIP embeddings, run Item Q-Former inference, and batch-generate item query tokens.  
  See `data_processing/README.md` for details and example flows.

- **`models/`** – core model components (Q-Former backbone + wrappers, item/user encoders, MWNE utilities).  
  See `models/README.md` for a breakdown of each module.

- **`training/`** – training scripts for:
  - Item Q-Former,
  - User Q-Former,
  - Joint Qwen3+LoRA with injected query tokens.  
  See `training/README.md` for per-script goals and rough pipelines.

- **`evaluation/`** – evaluation scripts (currently: Item Q-Former reconstruction quality).  
  See `evaluation/README.md` for usage and metrics.

---

## Quickstart: typical pipeline

1. **Prepare data**
   - Put your raw dataset under `data_rec/data/...`.
   - Run the dict builders and rec processors in `data_processing/`:
     - `create_item_dict.py`, `create_review_dict.py`, `create_triplet_dict.py`.
     - `process_rec_new_user.py` / `process_rec_old_user.py`.

2. **Generate base embeddings**
   - Run `item_embedding_clip.py` (and `review_embedding_clip.py` if you use reviews) to generate CLIP embeddings under `data_rec/embeddings/...`.

3. **Train Item Q-Former**
   - Optionally run `precompute_full_field_embeddings.py` to cache field embeddings.
   - Run `item_qformer_training.py` to train the Item Q-Former and save a checkpoint.

4. **Generate item query tokens**
   - Run `generate_all_item_embeddings.py` to create a cache of item query tokens for all items.

5. **Train User Q-Former and Qwen3+LoRA**
   - Run `user_qformer_training.py` to learn user query tokens from history.
   - Run `train_item_individual_token_joint.py` to jointly train Qwen3+LoRA with injected query tokens.

6. **Evaluate**
   - Run `evaluate_item_qformer.py` to measure Item Q-Former reconstruction quality.

All paths and hyperparameters are **meant to be edited** for your dataset; everything now uses relative paths so the project can be safely pushed to GitHub.
