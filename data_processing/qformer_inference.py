import torch
import torch.nn as nn
import numpy as np
import json
import os
from tqdm import tqdm
import pickle

from models.item_encoder_pure_value import ItemEncoder
from models.qformer import BertConfig
from models.qformer_utils import QFormerForItemRepresentation

def load_qformer_model(checkpoint_path, device):
    """
    Load a trained Q-Former model from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint (.pth file).
        device (torch.device): The device to load the model onto ('cuda' or 'cpu').
        
    Returns:
        tuple: (model, field_names)
            - model (QFormerForItemRepresentation): The loaded model.
            - field_names (list): The list of field names used during training.
    """
    print(f"🔄 Loading Q-Former model from: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration and field names from the checkpoint
    config = checkpoint['config']
    field_names = checkpoint['field_names']
    
    # Re-initialize the model with the saved configuration
    model = QFormerForItemRepresentation(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        num_query_tokens=config.query_length,
        field_embedding_dim=config.encoder_width,
        num_fields=len(field_names),  # Set num_fields based on the saved field names
        dropout=config.hidden_dropout_prob
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"✅ Model loaded successfully!")
    print(f"🔧 Model configured with {len(field_names)} fields: {field_names}")
    
    return model, field_names

def process_item_for_inference(item_data, item_encoder, field_names):
    """
    Process a single JSON item to create field embeddings and an attention mask.
    
    Args:
        item_data (dict): A dictionary representing a single item.
        item_encoder (ItemEncoder): An instance of the item encoder.
        field_names (list): The ordered list of field names from training.
        
    Returns:
        tuple: (field_embeddings, attention_mask)
            - field_embeddings (torch.Tensor): Tensor of shape (1, num_fields, field_embedding_dim).
            - attention_mask (torch.Tensor): Tensor of shape (1, num_fields).
    """
    embeddings = {}
    
    # Helper to check for null values
    def _is_null_value(value):
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() in ["", "null", "NULL", "Null", "none", "NONE", "None", "nan", "NaN", "NAN"]
        return False

    # Encode each field value
    for field in field_names:
        value = item_data.get(field)
        
        if not _is_null_value(value):
            try:
                if field in ['price', 'average_rating', 'rating_number']:
                    embeddings[field] = item_encoder._encode_number(float(value))
                elif field == 'main_image':
                    embeddings[field] = item_encoder._encode_image(str(value))
                else:
                    embeddings[field] = item_encoder._encode_text(str(value))
            except Exception as e:
                # print(f"⚠️ Warning: Failed to encode {field}='{value}': {e}")
                embeddings[field] = np.zeros(1024)
        else:
            # Missing field - use zero embedding
            embeddings[field] = np.zeros(1024)
            
    # Create tensor in the correct order
    field_embeddings_list = [embeddings[field] for field in field_names]
    field_embeddings = torch.tensor(np.array(field_embeddings_list), dtype=torch.float32).unsqueeze(0)
    
    # Create attention mask
    attention_mask = torch.ones(1, len(field_names), dtype=torch.long)
    for i, field in enumerate(field_names):
        if _is_null_value(item_data.get(field)):
            attention_mask[0, i] = 0
            
    return field_embeddings, attention_mask

def run_inference(model, field_names, item_encoder, data_path, output_path, batch_size=64):
    """
    Run inference on a JSON dataset to extract query tokens for each item.
    
    Args:
        model (QFormerForItemRepresentation): The trained Q-Former model.
        field_names (list): List of field names used during training.
        item_encoder (ItemEncoder): An instance of the item encoder.
        data_path (str): Path to the input JSON file.
        output_path (str): Path to save the output query tokens (as a .pkl file).
        batch_size (int): Batch size for inference.
    """
    print(f"🚀 Starting inference on: {data_path}")
    
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    items = []
    for item_id, item_data in data.items():
        item_data['item_id'] = item_id
        items.append(item_data)
        
    print(f"📊 Found {len(items)} items to process.")
    
    # Set up device
    device = next(model.parameters()).device
    
    # Dictionary to store results
    all_query_tokens = {}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(items), batch_size), desc="Running Inference"):
            batch_items = items[i:i+batch_size]
            
            batch_field_embeddings = []
            batch_attention_masks = []
            
            for item in batch_items:
                field_embeddings, attention_mask = process_item_for_inference(item, item_encoder, field_names)
                batch_field_embeddings.append(field_embeddings)
                batch_attention_masks.append(attention_mask)
            
            # Concatenate batch tensors
            batch_field_embeddings = torch.cat(batch_field_embeddings, dim=0).to(device)
            batch_attention_masks = torch.cat(batch_attention_masks, dim=0).to(device)
            
            # Forward pass
            outputs = model(batch_field_embeddings, batch_attention_masks)
            
            # Get query outputs (batch_size, num_query_tokens, hidden_size)
            query_outputs = outputs['query_outputs'].cpu().numpy()
            
            # Store results
            for j, item in enumerate(batch_items):
                item_id = item['item_id']
                all_query_tokens[item_id] = query_outputs[j]

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(all_query_tokens, f)
        
    print(f"✅ Inference complete!")
    print(f"💾 Query tokens for {len(all_query_tokens)} items saved to: {output_path}")

if __name__ == "__main__":
    # --- Configuration ---
    CHECKPOINT_PATH = "qformer_checkpoints_contrastive_8_query_tokens/best_qformer_model.pth"
    INPUT_DATA_PATH = "data_rec/dict/All_Beauty_item_triplet_dict.json"
    OUTPUT_TOKENS_PATH = "inference_results/qformer_item_query_tokens.pkl"
    INFERENCE_BATCH_SIZE = 128
    
    # Set device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # --- Main Execution ---
    
    # 1. Load the trained model
    model, field_names = load_qformer_model(CHECKPOINT_PATH, DEVICE)
    
    # 2. Initialize the item encoder
    item_encoder = ItemEncoder()
    
    # 3. Run inference to generate query tokens
    run_inference(
        model=model,
        field_names=field_names,
        item_encoder=item_encoder,
        data_path=INPUT_DATA_PATH,
        output_path=OUTPUT_TOKENS_PATH,
        batch_size=INFERENCE_BATCH_SIZE
    )

    # --- Verification (Optional) ---
    print("\n🔍 Verifying the output file...")
    try:
        with open(OUTPUT_TOKENS_PATH, 'rb') as f:
            saved_tokens = pickle.load(f)
        
        num_items = len(saved_tokens)
        print(f"✅ Successfully loaded {num_items} items from the output file.")
        
        if num_items > 0:
            first_item_id = list(saved_tokens.keys())[0]
            first_item_tokens = saved_tokens[first_item_id]
            print(f"  - First item ID: {first_item_id}")
            print(f"  - Shape of query tokens for first item: {first_item_tokens.shape}")
            
    except Exception as e:
        print(f"⚠️ Error verifying output file: {e}")