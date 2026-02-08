import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Any

from .mwne import TimestampEncoder, GeoCoordinateEncoder
from .item_encoder_pure_value import ItemEncoder
from .qformer_utils import QFormerForItemRepresentation


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class UserSequenceEncoder:
    """
    Encodes user interaction sequences by combining context-aware item representations.
    This class is designed to be used in a lightweight training script for a User-QFormer.
    """
    def __init__(self, item_qformer_checkpoint_path: str, item_encoder_config_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Using device: {self.device}")

        self.item_encoder = ItemEncoder(config_path=item_encoder_config_path)
        self._load_item_qformer(item_qformer_checkpoint_path)

        # Initialize metadata encoders
        self.embedding_dim = self.item_qformer.config.hidden_size
        self.timestamp_encoder = TimestampEncoder(embedding_dim=self.embedding_dim).to(self.device)
        self.geo_encoder = GeoCoordinateEncoder(embedding_dim=self.embedding_dim).to(self.device)
        self.positional_encoder = PositionalEncoding(d_model=self.embedding_dim).to(self.device)

        print("✅ UserSequenceEncoder initialized successfully.")

    def _load_item_qformer(self, checkpoint_path: str):
        """Loads the pre-trained item-level Q-Former model."""
        print("🔄 Loading pre-trained Item Q-Former...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Determine num_fields from the checkpoint
        num_fields = len(checkpoint['field_names'])
        
        # Initialize the model with the same architecture
        self.item_qformer = QFormerForItemRepresentation(num_fields=num_fields).to(self.device)
        self.item_qformer.load_state_dict(checkpoint['model_state_dict'])
        self.item_qformer.eval() # Set to evaluation mode
        self.item_qformer_fields = checkpoint['field_names'] # Store field names
        print(f"✅ Item Q-Former loaded with {num_fields} fields.")

    @torch.no_grad()
    def _get_item_query_tokens_batch(self, item_samples: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Computes item query tokens for a batch of item samples on-the-fly.
        """
        # 1. Get field embeddings using the ItemEncoder
        encoded_fields = self.item_encoder.encode_batch_by_field(item_samples, self.item_qformer_fields)

        # 2. Assemble embeddings and create attention masks
        batch_embeddings = []
        batch_masks = []
        for i in range(len(item_samples)):
            embeddings = []
            mask = []
            for field in self.item_qformer_fields:
                embedding = encoded_fields[field][i]
                embeddings.append(embedding)
                mask.append(1 if np.any(embedding) else 0)
            batch_embeddings.append(np.array(embeddings))
            batch_masks.append(mask)
        
        field_embeddings_tensor = torch.tensor(np.array(batch_embeddings), dtype=torch.float32).to(self.device)
        attention_mask_tensor = torch.tensor(batch_masks, dtype=torch.long).to(self.device)

        # 3. Pass through the Item Q-Former to get query tokens
        qformer_output = self.item_qformer(field_embeddings_tensor, attention_mask_tensor)
        
        # Return the raw query outputs, which are the richest representation
        return qformer_output['query_outputs'] # Shape: [batch_size, num_queries, hidden_dim]

    def encode_user_sequence(self, user_history: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Encodes a single user's history into a sequence of contextual event tokens.
        
        Args:
            user_history: A list of interaction events, where each event is a dictionary
                          containing 'item_data', 'timestamp', and 'coordinates'.
        
        Returns:
            A tensor representing the user's encoded sequence.
        """
        if not user_history:
            return torch.empty(0, self.item_qformer.num_query_tokens, self.embedding_dim)

        # Extract data for batch processing
        item_samples = [event['item_data'] for event in user_history]
        timestamps = torch.tensor([event['timestamp'] for event in user_history], device=self.device)
        coords = torch.tensor([event['coordinates'] for event in user_history], device=self.device)

        # Get all item representations in one batch
        item_query_tokens = self._get_item_query_tokens_batch(item_samples)

        # Get all metadata embeddings in one batch
        with torch.no_grad():
            time_embs = self.timestamp_encoder(timestamps) # [seq_len, dim]
            geo_embs = self.geo_encoder(coords)           # [seq_len, dim]
        
        # Fuse context into item tokens
        # Reshape context embeddings to be broadcastable: [seq_len, 1, dim]
        context_embs = time_embs.unsqueeze(1) + geo_embs.unsqueeze(1)
        contextual_event_tokens = item_query_tokens + context_embs # Broadcasting applies context to all query tokens

        # Reshape from [seq_len, num_queries, dim] to [seq_len * num_queries, dim]
        num_queries = self.item_qformer.num_query_tokens
        sequence_len = len(user_history)
        flat_sequence = contextual_event_tokens.view(sequence_len * num_queries, self.embedding_dim)

        # Add positional encoding
        # The positional encoder expects [seq_len, batch_size, dim], so we adapt
        flat_sequence_with_pos = self.positional_encoder(flat_sequence.unsqueeze(1)).squeeze(1)

        return flat_sequence_with_pos

def main():
    """Example usage of the UserSequenceEncoder."""
    # This is a placeholder for a real user history dataset
    # In a real scenario, you would load this data from a file.
    dummy_user_history = [
        {
            "item_data": {
                "title": "Natural Language Processing with Transformers",
                "description": "A book about modern NLP techniques.",
                "price": 79.99,
                "main_image": "http://images.amazon.com/images/P/B00005N5PF.01.LZZZZZZZ.jpg" # Example URL
            },
            "timestamp": 1672531200,  # 2023-01-01 00:00:00
            "coordinates": [40.7128, -74.0060]  # New York City
        },
        {
            "item_data": {
                "title": "Designing Data-Intensive Applications",
                "description": "The Big Ideas Behind Reliable, Scalable, and Maintainable Systems.",
                "price": 59.99,
                "main_image": "http://images.amazon.com/images/P/B00005N5PI.01.LZZZZZZZ.jpg" # Example URL
            },
            "timestamp": 1672617600,  # 2023-01-02 00:00:00
            "coordinates": [34.0522, -118.2437] # Los Angeles
        }
    ]
    
    print("🚀 Initializing User Sequence Encoder...")
    sequence_encoder = UserSequenceEncoder(
        item_qformer_checkpoint_path="qformer_checkpoints_contrastive_32_query_tokens/best_qformer_model.pth",
        item_encoder_config_path="config/triplet_config.yaml"
    )
    
    print("\n🔍 Encoding user sequence...")
    encoded_sequence = sequence_encoder.encode_user_sequence(dummy_user_history)
    
    print("\n📊 Encoding complete.")
    print(f"  - Shape of final sequence tensor: {encoded_sequence.shape}")
    
    num_events = len(dummy_user_history)
    tokens_per_item = sequence_encoder.item_qformer.num_query_tokens
    embedding_dim = sequence_encoder.embedding_dim
    
    print(f"  - Expected shape: [{num_events * tokens_per_item}, {embedding_dim}]")
    assert encoded_sequence.shape == (num_events * tokens_per_item, embedding_dim)
    print("✅ Shape validation successful!")

if __name__ == "__main__":
    main()