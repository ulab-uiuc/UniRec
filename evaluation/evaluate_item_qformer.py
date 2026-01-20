#!/usr/bin/env python3
"""
Evaluate trained Q-Former model on recommendation tasks.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import json
from models.item_encoder_pure_value import ItemEncoder
from models.qformer_utils import QFormerForItemRepresentation
from torch.utils.data import DataLoader, TensorDataset


def load_trained_model(checkpoint_path: str):
    """Load trained Q-Former model."""
    # Load with weights_only=False for compatibility
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    field_names = checkpoint.get('field_names')
    if field_names is None:
        raise ValueError("Checkpoint must contain 'field_names'")

    # Instantiate the model from the config
    model = QFormerForItemRepresentation(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        num_query_tokens=config.query_length,
        field_embedding_dim=config.encoder_width,
        num_fields=len(field_names),
        dropout=config.hidden_dropout_prob
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, field_names

def evaluate_reconstruction_quality(model, cache_path, device='cuda', batch_size=2048):
    """Evaluate reconstruction quality using cosine similarity on cached validation embeddings."""
    model.to(device)
    print("🔍 Evaluating reconstruction quality...")

    # Load cached validation data
    try:
        val_embeddings_dict = torch.load(os.path.join(cache_path, "embeddings.pt"))
        val_masks_dict = torch.load(os.path.join(cache_path, "masks.pt"))
        
        # The cached data is a dictionary of tensors; stack them into a single tensor.
        val_embeddings = torch.stack(list(val_embeddings_dict.values()))
        val_masks = torch.stack(list(val_masks_dict.values()))
            
    except FileNotFoundError:
        print(f"❌ Cache not found at {cache_path}. Please run training first to generate cache.")
        return None

    val_dataset = TensorDataset(val_embeddings, val_masks)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    total_cosine_sim = 0
    num_valid_fields = 0
    total_val_loss = 0

    with torch.no_grad():
        for field_embeddings, attention_mask in val_loader:
            field_embeddings = field_embeddings.to(device)
            attention_mask = attention_mask.to(device)

            out = model(field_embeddings, attention_mask)
            reconstructed_fields = out['reconstructed_fields']

            # Compute masked MSE loss for validation
            recon_loss_unreduced = F.mse_loss(reconstructed_fields, field_embeddings, reduction='none')
            masked_val_loss = (recon_loss_unreduced * attention_mask.unsqueeze(-1)).sum() / attention_mask.sum()
            total_val_loss += masked_val_loss.item()
            
            # Vectorized cosine similarity calculation
            valid_mask = attention_mask.bool()
            
            original_valid = field_embeddings[valid_mask]
            reconstructed_valid = reconstructed_fields[valid_mask]

            if original_valid.shape[0] > 0:
                original_norm = F.normalize(original_valid, p=2, dim=-1)
                reconstructed_norm = F.normalize(reconstructed_valid, p=2, dim=-1)

                cosine_sims = torch.sum(original_norm * reconstructed_norm, dim=-1)
                
                total_cosine_sim += cosine_sims.sum().item()
                num_valid_fields += original_valid.shape[0]

    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
    avg_cosine_sim = total_cosine_sim / num_valid_fields if num_valid_fields > 0 else 0

    print(f"\n📊 Reconstruction Quality Results:")
    print(f"  Validation Recon Loss (MSE): {avg_val_loss:.4f}")
    print(f"  Avg Cosine Similarity: {avg_cosine_sim:.4f}")

    return {
        'val_recon_loss': avg_val_loss,
        'avg_cosine_similarity': avg_cosine_sim,
    }


def main():
    """Main evaluation function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model
    checkpoint_path = "qformer_checkpoints_contrastive_16_query_tokens/best_qformer_model.pth"
    print(f"🔄 Loading model from {checkpoint_path}...")
    model, _ = load_trained_model(checkpoint_path)
    
    # Path to the cached validation data
    cache_path = "embedding_cache_contrastive/val"
    
    # Evaluate
    results = evaluate_reconstruction_quality(model, cache_path, device)
    
    if results:
        print(f"\n✅ Evaluation completed!")
        print(f"📈 Avg Cosine Similarity: {results['avg_cosine_similarity']:.4f}")

if __name__ == "__main__":
    main() 