# UniRec: Nested Q-Former + Qwen3 LoRA Recommendation Stack

<p align="center">
    <a href="https://github.com/ulab-uiuc/UniRec">
        <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Repository-blue?logo=github">
    </a>
    <!-- <a href="http://arxiv.org/abs/2507.10540">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2507.10540-red?logo=arxiv">
    </a> -->
    <a href="https://github.com/ulab-uiuc/UniRec/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <br>
    <a href="https://github.com/ulab-uiuc/UniRec">
        <img alt="Stars" src="https://img.shields.io/github/stars/ulab-uiuc/UniRec">
    </a>
    <a href="https://github.com/ulab-uiuc/UniRec">
        <img alt="Forks" src="https://img.shields.io/github/forks/ulab-uiuc/UniRec">
    </a>
    <a href="https://github.com/ulab-uiuc/UniRec">
        <img alt="Issues" src="https://img.shields.io/github/issues/ulab-uiuc/UniRec">
    </a>
</p>

<p align="center">
    <a href="https://github.com/ulab-uiuc/UniRec">📦 Repository</a> |
    <!-- <a href="http://arxiv.org/abs/2507.10540">📜 arXiv</a> | -->
    <a href="#-folder-structure">📂 Structure</a> |
    <a href="#-quickstart">🚀 Quickstart</a>
</p>



## Overview

This repository demonstrates a **nested Q-Former + Qwen3 LoRA** recommendation stack:

1. **Item encoder + Item Q-Former**  
   Raw item fields (text, CLIP features, etc.) → dense field embeddings → **item query tokens**.
2. **User Q-Former**  
   User history as a sequence of item query tokens → **user query tokens**.
3. **Qwen3 + LoRA joint model**  
   Injects item/user query tokens as *special tokens* in Qwen3, then uses the final embedding as a **predicted next-item embedding** for ranking a candidate pool.

All scripts here are copies from the original project, reorganized into a GitHub‑friendly layout (no hardcoded API keys or absolute cluster paths).



## 🛠️ Environment Setup

```bash
conda create -n unirec python=3.9
conda activate unirec

# Core deep learning libraries
pip install torch
pip install transformers
pip install sentence-transformers
pip install peft

# Data processing and utilities
pip install numpy
pip install pandas
pip install scikit-learn
pip install pyyaml
pip install tqdm

# Image processing
pip install Pillow

# HTTP requests (for downloading images)
pip install requests
```

**Note**: This project uses:
- **Qwen3-Embedding-0.6B** for text embeddings (via `sentence-transformers`)
- **CLIP ViT-Large** for image embeddings (via `transformers`)
- **Qwen3-Embedding-0.6B** as the base model for joint training (via `transformers`)
- **PEFT/LoRA** for parameter-efficient fine-tuning

Make sure you have CUDA-compatible PyTorch installed if you plan to use GPU acceleration.



## 📂 Folder Structure

- **`data_processing/`** – build dicts, process recommendation data, generate CLIP embeddings, run Item Q-Former inference, and batch-generate item query tokens.  
  See [`data_processing/README.md`](data_processing/README.md) for details and example flows.

- **`models/`** – core model components (Q-Former backbone + wrappers, item/user encoders, MWNE utilities).  
  See [`models/README.md`](models/README.md) for a breakdown of each module.

- **`training/`** – training scripts for:
  - Item Q-Former,
  - User Q-Former,
  - Joint Qwen3+LoRA with injected query tokens.  
  See [`training/README.md`](training/README.md) for per-script goals and rough pipelines.

- **`evaluation/`** – evaluation scripts (currently: Item Q-Former reconstruction quality).  
  See [`evaluation/README.md`](evaluation/README.md) for usage and metrics.



## 🎯 Data Processing

Run the following commands to prepare your dataset:

### 1. Prepare Data

Put your raw dataset under `data_rec/data/...` and run the dict builders and rec processors:

```bash
# Build item dictionary
python data_processing/create_item_dict.py

# Build review dictionary (if using reviews)
python data_processing/create_review_dict.py

# Build triplet dictionary
python data_processing/create_triplet_dict.py

# Process recommendation data
python data_processing/process_rec_new_user.py
python data_processing/process_rec_old_user.py
```

You may refer to the specific README in the [`data_processing`](data_processing/README.md) directory for detailed argument descriptions.

### 2. Generate Base Embeddings

Run CLIP embedding generation scripts:

```bash
# Generate CLIP embeddings for items
python data_processing/item_embedding_clip.py

# Generate CLIP embeddings for reviews (if using reviews)
python data_processing/review_embedding_clip.py
```

This will generate CLIP embeddings under `data_rec/embeddings/...`.



## 📊 Training

### Item Q-Former Training

First, optionally precompute field embeddings to speed up training:

```bash
# Precompute and cache all item field embeddings
python training/precompute_full_field_embeddings.py
```

Then train the Item Q-Former:

```bash
# Train Item Q-Former
python training/item_qformer_training.py
```

For more detailed information about the training process, please refer to the specific README in the [`training`](training/README.md) directory.

### Generate Item Query Tokens

After training the Item Q-Former, generate item query tokens for all items:

```bash
# Generate item query tokens cache
python data_processing/generate_all_item_embeddings.py
```

### User Q-Former and Joint Training

Train the User Q-Former and jointly train Qwen3+LoRA:

```bash
# Train User Q-Former
python training/user_qformer_training.py

# Jointly train Qwen3+LoRA with injected query tokens
python training/train_item_individual_token_joint.py
```

You may refer to the specific README in the [`training`](training/README.md) directory for detailed instructions and hyperparameter configurations.



## 📈 Evaluation

UniRec provides evaluation scripts to assess model performance. Currently supported:

- **Item Q-Former reconstruction quality** – measures how well the Item Q-Former reconstructs item field embeddings.

To evaluate your model's performance:

```bash
# Evaluate Item Q-Former reconstruction quality
python evaluation/evaluate_item_qformer.py
```

For detailed information about the evaluation framework, supported metrics, and usage instructions, please refer to the [`evaluation/README.md`](evaluation/README.md).



## 🚀 Quickstart: Typical Pipeline

For a complete end-to-end workflow:

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



## Citation

If you find this repository useful, please consider citing:

```bibtex
@misc{unirec2024,
  title={UniRec: Nested Q-Former + Qwen3 LoRA Recommendation Stack},
  author={UIUC U-Lab},
  year={2024},
  howpublished={\url{https://github.com/ulab-uiuc/UniRec}}
}
```
