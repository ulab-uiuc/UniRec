
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from tqdm import tqdm
import sys
import torch.nn.functional as F
import time
import math
import pickle
from typing import Optional
import random

# Import from refactored package layout
from models.item_encoder_pure_value import ItemEncoder
from models.qformer_utils import QFormerForItemRepresentation, QFormerDataset, load_real_data


class QFormerTripletDataset(Dataset):
    def __init__(self, samples, item_encoder, item_sequences, cache_dir, precompute_batch_size=8192):
        self.base_dataset = QFormerDataset(samples, item_encoder, cache_dir, precompute_batch_size)
        self.item_id_to_idx = {s['item_id']: i for i, s in enumerate(samples)}
        self.triplet_pairs = [(seq[i], seq[i+1]) for seq in item_sequences for i in range(len(seq)-1)
                              if seq[i] in self.item_id_to_idx and seq[i+1] in self.item_id_to_idx]
    def __len__(self): return len(self.triplet_pairs)
    def __getitem__(self, idx):
        anchor_id, pos_id = self.triplet_pairs[idx]
        neg_id = random.choice(list(self.item_id_to_idx.keys()))
        while neg_id == anchor_id or neg_id == pos_id:
            neg_id = random.choice(list(self.item_id_to_idx.keys()))
        return {
            "anchor": self.base_dataset[self.item_id_to_idx[anchor_id]],
            "positive": self.base_dataset[self.item_id_to_idx[pos_id]],
            "negative": self.base_dataset[self.item_id_to_idx[neg_id]],
        }

class QFormerLoss(nn.Module):
    def __init__(self, reconstruction_weight=1.0, contrastive_weight=0.5, margin=0.5):
        super().__init__()
        self.recon_w = reconstruction_weight
        self.cont_w = contrastive_weight
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        self.mse_loss = nn.MSELoss(reduction='none') # Use 'none' to handle masking

    def forward(self, model_output, input_embeddings, pos_rep, neg_rep, attention_mask):
        # Masked reconstruction loss
        recon_loss_unreduced = self.mse_loss(model_output['reconstructed_fields'], input_embeddings['field_embeddings'])
        # Mask out the loss for padding fields
        masked_recon_loss = (recon_loss_unreduced * attention_mask.unsqueeze(-1)).sum() / attention_mask.sum()
        
        cont_loss = self.triplet_loss(model_output['item_representation'], pos_rep, neg_rep)
        return self.recon_w * masked_recon_loss + self.cont_w * cont_loss, masked_recon_loss, cont_loss

def load_item_sequences(data_path, max_sequences=None):
    with open(data_path, 'r') as f: data = json.load(f)
    sequences = [item['history'] for item in data if 'history' in item and len(item['history']) > 1]
    if max_sequences: sequences = sequences[:max_sequences]
    return sequences

def split_data_for_validation(samples, train_ratio=0.9, random_seed=42):
    np.random.seed(random_seed)
    perm = np.random.permutation(len(samples))
    split = int(train_ratio * len(samples))
    return [samples[i] for i in perm[:split]], [samples[i] for i in perm[split:]]

def train_qformer(data_path, sequences_path, output_dir, **kwargs):
    all_samples = load_real_data(data_path, kwargs.get("max_samples"))
    item_sequences = load_item_sequences(sequences_path, kwargs.get("max_sequences"))
    
    train_samples, val_samples = split_data_for_validation(all_samples)
    
    item_encoder = ItemEncoder()
    cache_dir = kwargs.get("cache_dir")
    precompute_batch_size = kwargs.get("precompute_batch_size", 8192)
    train_dataset = QFormerTripletDataset(train_samples, item_encoder, item_sequences, 
                                        cache_dir=os.path.join(cache_dir, "train"), 
                                        precompute_batch_size=precompute_batch_size)
    val_dataset = QFormerDataset(val_samples, item_encoder, 
                               cache_dir=os.path.join(cache_dir, "val"), 
                               precompute_batch_size=precompute_batch_size)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=kwargs.get("batch_size", 32), 
        shuffle=True,
        num_workers=kwargs.get("num_workers", 8),
        pin_memory=True,
        prefetch_factor=kwargs.get("prefetch_factor", 4),
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=kwargs.get("batch_size", 32),
        num_workers=kwargs.get("num_workers", 4),  # Fewer workers for validation
        pin_memory=True,
        prefetch_factor=kwargs.get("prefetch_factor", 2),
        persistent_workers=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QFormerForItemRepresentation(num_fields=len(train_dataset.base_dataset.available_fields)).to(device)
    criterion = QFormerLoss(contrastive_weight=kwargs.get("contrastive_weight", 0.1))
    optimizer = optim.AdamW(model.parameters(), lr=kwargs.get("learning_rate", 5e-5))
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(kwargs.get("num_epochs", 10)):
        model.train()
        total_train_loss, total_recon_loss, total_cont_loss = 0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            anc_in, pos_in, neg_in = batch['anchor'], batch['positive'], batch['negative']
            anc_emb = anc_in['field_embeddings'].to(device)
            anc_mask = anc_in['attention_mask'].to(device)
            
            anc_out = model(anc_emb, anc_mask)
            with torch.no_grad():
                pos_out = model(pos_in['field_embeddings'].to(device), pos_in['attention_mask'].to(device))
                neg_out = model(neg_in['field_embeddings'].to(device), neg_in['attention_mask'].to(device))
                
            loss, recon, cont = criterion(anc_out, {k: v.to(device) for k, v in anc_in.items() if isinstance(v, torch.Tensor)}, pos_out['item_representation'], neg_out['item_representation'], anc_mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_recon_loss += recon.item()
            total_cont_loss += cont.item()

        print(f"Avg Train Loss: {total_train_loss/len(train_loader):.4f}, Recon: {total_recon_loss/len(train_loader):.4f}, Contrast: {total_cont_loss/len(train_loader):.4f}")

        # Evaluate on validation set every 10 epochs
        if (epoch + 1) % 50 == 0:
            model.eval()
            total_val_loss = 0
            total_cosine_sim = 0
            num_valid_fields = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    field_embeddings = batch['field_embeddings'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    out = model(field_embeddings, attention_mask)
                    
                    # Compute masked MSE loss for validation
                    recon_loss_unreduced = F.mse_loss(out['reconstructed_fields'], field_embeddings, reduction='none')
                    masked_val_loss = (recon_loss_unreduced * attention_mask.unsqueeze(-1)).sum() / attention_mask.sum()
                    total_val_loss += masked_val_loss.item()

                    # Compute cosine similarity for valid fields
                    for i in range(field_embeddings.size(0)):
                        for j in range(field_embeddings.size(1)):
                            if attention_mask[i, j].item() == 1:
                                original = F.normalize(field_embeddings[i, j], p=2, dim=0)
                                reconstructed = F.normalize(out['reconstructed_fields'][i, j], p=2, dim=0)
                                total_cosine_sim += F.cosine_similarity(original, reconstructed, dim=0).item()
                                num_valid_fields += 1
                                
            avg_val_loss = total_val_loss / len(val_loader)
            avg_cosine_sim = total_cosine_sim / num_valid_fields if num_valid_fields > 0 else 0
            
            print(f"Epoch {epoch+1}, Val Recon Loss: {avg_val_loss:.4f}, Avg Cosine Sim: {avg_cosine_sim:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                
                # Ensure the output directory exists before saving
                os.makedirs(output_dir, exist_ok=True)

                # Prepare checkpoint with model state, config, and field names
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'config': model.config,
                    'field_names': train_dataset.base_dataset.available_fields
                }
                
                # Save the checkpoint
                save_path = os.path.join(output_dir, "best_qformer_model.pth")
                torch.save(checkpoint, save_path)
                print(f"Saved new best model to {save_path}")

if __name__ == "__main__":
    # For RTX A6000 (49GB VRAM) - feel free to increase precompute_batch_size further if needed
    # Recommended values: 8192 (safe), 12288 (aggressive), 16384 (max for most scenarios)
    train_qformer(
        data_path="data_rec/dict/All_Beauty_item_triplet_dict.json",
        sequences_path="data_rec/data/Amazon_All_Beauty_all_train_LRanker.json",
        output_dir="qformer_checkpoints_contrastive_32_query_tokens",
        num_epochs=500,
        batch_size=4096,
        learning_rate=1e-4,
        cache_dir="embedding_cache_contrastive",
        contrastive_weight=0.25,
        max_samples=112590,
        max_sequences=1000000,
        num_workers=8,  # Enable DataLoader parallelization
        prefetch_factor=4,  # Prefetch batches for better GPU utilization
        precompute_batch_size=4096  # Maximum batch size for RTX A6000 - adjust down if OOM occurs
    )
    