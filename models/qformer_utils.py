
import torch
import torch.nn as nn
import numpy as np
import json
import os
from tqdm import tqdm
import sys
import pickle
from torch.utils.data import Dataset

# Refactored layout imports
from .item_encoder_pure_value import ItemEncoder
from .qformer import BertModel, BertConfig

class QFormerForItemRepresentation(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_hidden_layers: int = 12, num_attention_heads: int = 16,
                 intermediate_size: int = 4096, num_query_tokens: int = 32, field_embedding_dim: int = 1024,
                 num_fields: int = None, dropout: float = 0.2):
        super().__init__()
        if num_fields is None: raise ValueError("num_fields must be provided")
        
        self.config = BertConfig(
            hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
            add_cross_attention=True, query_length=num_query_tokens, encoder_width=field_embedding_dim,
            cross_attention_freq=2
        )
        self.num_query_tokens = num_query_tokens
        self.query_embeddings = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        self.qformer = BertModel(self.config, add_pooling_layer=False)
        self.item_representation_head = nn.Linear(hidden_size, field_embedding_dim)
        # Reconstruction head and a projection to match number of fields
        self.reconstruction_head = nn.Linear(hidden_size, field_embedding_dim)
        self.field_projection = nn.Linear(num_query_tokens, num_fields)

    def forward(self, field_embeddings: torch.Tensor, attention_mask: torch.Tensor = None):
        batch_size = field_embeddings.shape[0]
        query_embeds = self.query_embeddings.expand(batch_size, -1, -1)
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, field_embeddings.shape[1], device=field_embeddings.device)
        
        query_attention_mask = torch.ones(batch_size, self.num_query_tokens, device=field_embeddings.device)
        
        outputs = self.qformer(
            query_embeds=query_embeds, encoder_hidden_states=field_embeddings,
            encoder_attention_mask=attention_mask, attention_mask=query_attention_mask, return_dict=True
        )
        query_outputs = outputs.last_hidden_state
        item_representation = self.item_representation_head(query_outputs.mean(dim=1))
        
        # Reconstruct each field embedding from the query outputs
        reconstructed_fields_from_queries = self.reconstruction_head(query_outputs)
        reconstructed_fields = self.field_projection(reconstructed_fields_from_queries.transpose(1, 2)).transpose(1, 2)
        
        return {
            'query_outputs': query_outputs,
            'item_representation': item_representation,
            'reconstructed_fields': reconstructed_fields
        }

class QFormerDataset(Dataset):
    def __init__(self, samples, item_encoder, cache_dir=None, precompute_batch_size=8192):
        self.samples = samples
        self.item_encoder = item_encoder
        self.cache_dir = cache_dir
        self.precompute_batch_size = precompute_batch_size
        self.available_fields = self._analyze_fields()
        self.embedding_cache = {}
        self.mask_cache = {}

        if cache_dir and self._load_cache():
            print("Loaded from cache.")
        else:
            self._precompute()
            if cache_dir: self._save_cache()
    
    def _analyze_fields(self):
        """
        Analyzes all unique fields present in the item samples.
        """
        print("Analyzing unique fields from all samples...")
        all_fields = set()
        for sample in self.samples:
            all_fields.update(sample.keys())
        
        # Define a consistent order for fields, excluding item_id
        sorted_fields = sorted([f for f in all_fields if f != 'item_id'])
        print(f"Found {len(sorted_fields)} unique fields.")
        return sorted_fields

    def _precompute(self):
        """
        Pre-computes embeddings for all samples in the dataset using efficient batching.
        """
        batch_size = self.precompute_batch_size
        print(f"🚀 Starting batched embedding pre-computation with batch size {batch_size}...")
        print(f"💻 Optimized for RTX A6000 (49GB VRAM) - using large batches for maximum GPU utilization")
        
        num_samples = len(self.samples)
        for i in tqdm(range(0, num_samples, batch_size), desc="Preprocessing Batches"):
            batch_samples = self.samples[i:i+batch_size]
            
            # Use the new batched encoder
            encoded_batches = self.item_encoder.encode_batch_by_field(batch_samples, self.available_fields)
            
            # Re-assemble the embeddings and masks for each sample in the batch
            for j in range(len(batch_samples)):
                sample_idx = i + j
                embeddings = []
                mask = []
                for field in self.available_fields:
                    embedding = encoded_batches[field][j]
                    embeddings.append(embedding)
                    # A field is considered valid if its embedding is not a zero vector
                    mask.append(1 if np.any(embedding) else 0)
                
                self.embedding_cache[sample_idx] = torch.tensor(np.array(embeddings), dtype=torch.float32)
                self.mask_cache[sample_idx] = torch.tensor(mask, dtype=torch.long)

    def _load_cache(self):
        emb_path = os.path.join(self.cache_dir, "embeddings.pt")
        mask_path = os.path.join(self.cache_dir, "masks.pt")
        fields_path = os.path.join(self.cache_dir, "fields.json")
        
        if os.path.exists(emb_path) and os.path.exists(mask_path) and os.path.exists(fields_path):
            with open(fields_path, 'r') as f:
                cached_fields = json.load(f)
            
            # Validate that the cached fields match the current fields
            if cached_fields == self.available_fields:
                print("Cache is valid. Loading embeddings and masks.")
                self.embedding_cache = torch.load(emb_path)
                self.mask_cache = torch.load(mask_path)
                return True
            else:
                print("Cache is outdated because fields have changed. Re-computing...")
                return False
        return False
    
    def _save_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save(self.embedding_cache, os.path.join(self.cache_dir, "embeddings.pt"))
        torch.save(self.mask_cache, os.path.join(self.cache_dir, "masks.pt"))
        # Save the fields list to validate cache later
        with open(os.path.join(self.cache_dir, "fields.json"), 'w') as f:
            json.dump(self.available_fields, f)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return {
            'field_embeddings': self.embedding_cache[idx],
            'attention_mask': self.mask_cache[idx],
            'item_id': self.samples[idx].get('item_id', str(idx))
        }

def load_real_data(data_path, max_samples=None):
    with open(data_path, 'r') as f: data = json.load(f)
    items = list(data.values())
    for i, s in enumerate(items): s['item_id'] = list(data.keys())[i]
    if max_samples: items = items[:max_samples]
    return items