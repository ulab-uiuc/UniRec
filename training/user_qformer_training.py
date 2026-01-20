import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from tqdm import tqdm
import random
import torch.nn.functional as F

# Refactored layout imports
from models.user_sequence_encoder import UserSequenceEncoder
from models.qformer import BertModel, BertConfig

# --- 1. User-Level Q-Former Model ---
class UserQFormer(nn.Module):
    """
    A Q-Former model to create a fixed-length representation of a variable-length user sequence.
    """
    def __init__(self, hidden_size: int = 1024, num_hidden_layers: int = 4, num_attention_heads: int = 16,
                 intermediate_size: int = 4096, num_query_tokens: int = 64, input_embedding_dim: int = 1024,
                 num_item_tokens_to_predict: int = 32, dropout: float = 0.1):
        super().__init__()
        
        self.config = BertConfig(
            hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
            add_cross_attention=True, query_length=num_query_tokens, encoder_width=input_embedding_dim,
            cross_attention_freq=1 # Cross-attend at every layer for user modeling
        )
        self.num_query_tokens = num_query_tokens
        self.query_embeddings = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        
        self.qformer = BertModel(self.config, add_pooling_layer=False)
        
        # Prediction head to map the user representation to the shape of the next item's query tokens
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_item_tokens_to_predict * input_embedding_dim) # Project to the flattened size of item tokens
        )
        self.num_item_tokens_to_predict = num_item_tokens_to_predict
        self.input_embedding_dim = input_embedding_dim

    def forward(self, user_sequence_tokens: torch.Tensor, attention_mask: torch.Tensor):
        batch_size = user_sequence_tokens.shape[0]
        query_embeds = self.query_embeddings.expand(batch_size, -1, -1)
        
        query_attention_mask = torch.ones(batch_size, self.num_query_tokens, device=user_sequence_tokens.device)
        
        outputs = self.qformer(
            query_embeds=query_embeds, 
            encoder_hidden_states=user_sequence_tokens,
            encoder_attention_mask=attention_mask, 
            attention_mask=query_attention_mask, 
            return_dict=True
        )
        user_representation = outputs.last_hidden_state.mean(dim=1) # Average query outputs to get user vector
        
        # Predict the next item's query tokens
        predicted_item_flat = self.prediction_head(user_representation)
        predicted_item_tokens = predicted_item_flat.view(
            batch_size, self.num_item_tokens_to_predict, self.input_embedding_dim
        )
        
        return predicted_item_tokens

# --- 2. Dataset and Dataloader ---
class UserHistoryDataset(Dataset):
    def __init__(self, history_path, review_path, item_data_path, min_seq_len=3, max_seq_len=50):
        print("🔄 Loading and processing user history data...")
        
        with open(history_path, 'r') as f:
            user_histories = json.load(f)
        with open(review_path, 'r') as f:
            review_data = json.load(f)
        with open(item_data_path, 'r') as f:
            self.item_data_map = json.load(f)
            
        self.timestamp_map = self._build_timestamp_map(review_data)
        self.training_samples = self._create_training_samples(user_histories, min_seq_len, max_seq_len)
        
        print(f"✅ Created {len(self.training_samples)} training samples.")

    def _build_timestamp_map(self, review_data):
        """Build a map from item_id to its earliest review timestamp."""
        ts_map = {}
        for item_id, reviews in review_data.items():
            if reviews:
                # Use the timestamp of the first review as a proxy
                ts_map[item_id] = reviews[0].get("unixReviewTime", 0)
        return ts_map

    def _create_training_samples(self, user_histories, min_seq_len, max_seq_len):
        """Create (input_sequence, target_item) samples from user histories."""
        samples = []
        for user_data in tqdm(user_histories, desc="Creating Samples"):
            history = user_data.get("history", [])
            if len(history) < min_seq_len:
                continue
            
            # Truncate long histories
            history = history[-max_seq_len:]
            
            # Create sliding window samples
            for i in range(1, len(history) - 1):
                input_hist = history[:i]
                target_item_id = history[i]
                samples.append((input_hist, target_item_id))
        return samples

    def __len__(self):
        return len(self.training_samples)

    def __getitem__(self, idx):
        history_ids, target_item_id = self.training_samples[idx]
        
        # Prepare input sequence data
        input_sequence = []
        for item_id in history_ids:
            input_sequence.append({
                "item_data": self.item_data_map.get(item_id, {}),
                "timestamp": self.timestamp_map.get(item_id, 0),
                "coordinates": [0.0, 0.0] # Placeholder as data is not available
            })
            
        # Prepare target item data
        target_item = {
            "item_data": self.item_data_map.get(target_item_id, {}),
            "timestamp": self.timestamp_map.get(target_item_id, 0),
            "coordinates": [0.0, 0.0]
        }
        
        return {"input_sequence": input_sequence, "target_item": target_item}

def collate_fn(batch, sequence_encoder):
    """
    Custom collate function to process batches of sequences with the UserSequenceEncoder.
    """
    input_sequences = [item['input_sequence'] for item in batch]
    target_items = [item['target_item'] for item in batch]
    
    # 1. Encode target items to get their query tokens (our prediction target)
    target_item_data = [t['item_data'] for t in target_items]
    # Use the internal method of the encoder to get item tokens
    target_tokens = sequence_encoder._get_item_query_tokens_batch(target_item_data)

    # 2. Encode input sequences
    encoded_inputs = [sequence_encoder.encode_user_sequence(seq) for seq in input_sequences]
    
    # 3. Pad the input sequences to the same length
    max_len = max(seq.shape[0] for seq in encoded_inputs)
    padded_inputs = torch.zeros(len(batch), max_len, sequence_encoder.embedding_dim)
    attention_masks = torch.zeros(len(batch), max_len)
    
    for i, seq in enumerate(encoded_inputs):
        seq_len = seq.shape[0]
        padded_inputs[i, :seq_len] = seq
        attention_masks[i, :seq_len] = 1
        
    return padded_inputs, attention_masks, target_tokens

# --- 3. Training Loop ---
def train_user_qformer(output_dir, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the master encoder for all data preparation
    sequence_encoder = UserSequenceEncoder(
        item_qformer_checkpoint_path=kwargs["item_qformer_checkpoint"],
        item_encoder_config_path=kwargs["item_encoder_config"]
    )
    
    # Create Dataset
    dataset = UserHistoryDataset(
        history_path=kwargs["history_path"],
        review_path=kwargs["review_path"],
        item_data_path=kwargs["item_data_path"]
    )
    
    # Create DataLoader with the custom collate function
    data_loader = DataLoader(
        dataset,
        batch_size=kwargs.get("batch_size", 32),
        shuffle=True,
        num_workers=kwargs.get("num_workers", 0), # Multi-processing can be tricky with large models in memory
        collate_fn=lambda b: collate_fn(b, sequence_encoder)
    )

    # Initialize Model, Loss, and Optimizer
    model = UserQFormer(num_item_tokens_to_predict=sequence_encoder.item_qformer.num_query_tokens).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=kwargs.get("learning_rate", 1e-4))
    
    os.makedirs(output_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(kwargs.get("num_epochs", 10)):
        model.train()
        total_loss = 0
        
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
            inputs, masks, targets = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            
            predictions = model(inputs, masks)
            loss = criterion(predictions, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}, Average Training Loss: {avg_loss:.4f}")
        
        # Simple saving logic (can be expanded with a validation set)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(output_dir, "best_user_qformer_model.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model.config,
                'epoch': epoch,
                'loss': best_loss
            }, save_path)
            print(f"✅ Saved new best model to {save_path}")

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    train_user_qformer(
        output_dir="user_qformer_checkpoints",
        item_qformer_checkpoint="qformer_checkpoints_contrastive_32_query_tokens/best_qformer_model.pth",
        item_encoder_config="triplet_config.yaml",
        history_path="data_rec/data/Amazon_All_Beauty_all_train_LRanker.json",
        review_path="data_rec/dict/All_Beauty_review_dict.json",
        item_data_path="data_rec/dict/All_Beauty_item_triplet_dict.json",
        num_epochs=50,
        batch_size=64,
        learning_rate=5e-5,
        num_workers=4,
    )
