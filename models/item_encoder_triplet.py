import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import json

class ItemEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        num_queries: int = 8,
        field_vocab_size: int = 16,  # Based on your FIELD_MAPPING
        modality_vocab_size: int = 4,
        text_encoder_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        image_encoder_name: str = 'openai/clip-vit-base-patch32',
        category_vocab: Optional[Dict[str, int]] = None,
        precomputed_embeddings: Optional[Dict[str, torch.Tensor]] = None
    ):
        super().__init__()
        
        # Meta encoders
        self.field_encoder = nn.Embedding(field_vocab_size, hidden_dim)
        self.modality_encoder = nn.Embedding(modality_vocab_size, hidden_dim)
        
        # Modality-specific encoders
        self.text_encoder = TextEncoder(text_encoder_name, hidden_dim)
        self.category_encoder = CategoryEncoder(category_vocab, hidden_dim)
        self.image_encoder = ImageEncoder(image_encoder_name, hidden_dim)
        self.number_encoder = NumberEncoder(hidden_dim)
        
        # Q-Former
        self.q_former = QFormer(hidden_dim, num_queries)
        
        # Store precomputed embeddings if provided
        self.precomputed_embeddings = precomputed_embeddings or {}
        
        # Field and modality mappings
        self.field_mapping = FIELD_MAPPING
        self.modality_ids = MODALITY_IDS
        
    def extract_fields_from_item(self, item_id: str, item_data: Dict) -> List[Dict]:
        """Extract and prepare fields from raw item data."""
        fields = []
        
        # Text fields
        if item_data.get('title'):
            fields.append({
                'field_name': 'title',
                'field_id': self.field_mapping['title'][0],
                'modality_id': self.field_mapping['title'][1],
                'modality_type': 'text',
                'value': item_data['title'],
                'item_id': item_id
            })
        
        if item_data.get('description') and item_data['description'].strip():
            fields.append({
                'field_name': 'description',
                'field_id': self.field_mapping['description'][0],
                'modality_id': self.field_mapping['description'][1],
                'modality_type': 'text',
                'value': item_data['description'],
                'item_id': item_id
            })
        
        if item_data.get('features') and item_data['features'].strip():
            # Handle features as single text or list
            features_text = item_data['features']
            if isinstance(features_text, list):
                features_text = ' '.join(features_text)
            fields.append({
                'field_name': 'features',
                'field_id': self.field_mapping['features'][0],
                'modality_id': self.field_mapping['features'][1],
                'modality_type': 'text',
                'value': features_text,
                'item_id': item_id
            })
        
        # Category fields
        if item_data.get('main_category'):
            fields.append({
                'field_name': 'main_category',
                'field_id': self.field_mapping['main_category'][0],
                'modality_id': self.field_mapping['main_category'][1],
                'modality_type': 'category',
                'value': item_data['main_category'],
                'item_id': item_id
            })
        
        if item_data.get('store'):
            fields.append({
                'field_name': 'store',
                'field_id': self.field_mapping['store'][0],
                'modality_id': self.field_mapping['store'][1],
                'modality_type': 'category',
                'value': item_data['store'],
                'item_id': item_id
            })
        
        # Handle details fields if they exist
        if 'details' in item_data:
            details = item_data['details']
            for detail_field in ['Brand', 'Style', 'Color', 'Size', 'Material']:
                if detail_field in details:
                    field_name = detail_field.lower()
                    if field_name in self.field_mapping:
                        fields.append({
                            'field_name': field_name,
                            'field_id': self.field_mapping[field_name][0],
                            'modality_id': self.field_mapping[field_name][1],
                            'modality_type': 'category',
                            'value': details[detail_field],
                            'item_id': item_id
                        })
        
        # Image field
        if item_data.get('main_image'):
            fields.append({
                'field_name': 'main_image',
                'field_id': self.field_mapping['main_image'][0],
                'modality_id': self.field_mapping['main_image'][1],
                'modality_type': 'image',
                'value': item_data['main_image'],
                'item_id': item_id
            })
        
        # Number fields
        if item_data.get('price') is not None:
            fields.append({
                'field_name': 'price',
                'field_id': self.field_mapping['price'][0],
                'modality_id': self.field_mapping['price'][1],
                'modality_type': 'number',
                'value': float(item_data['price']),
                'item_id': item_id
            })
        
        if item_data.get('average_rating') is not None:
            fields.append({
                'field_name': 'average_rating',
                'field_id': self.field_mapping['average_rating'][0],
                'modality_id': self.field_mapping['average_rating'][1],
                'modality_type': 'number',
                'value': float(item_data['average_rating']),
                'item_id': item_id
            })
        
        if item_data.get('rating_number') is not None:
            fields.append({
                'field_name': 'rating_number',
                'field_id': self.field_mapping['rating_number'][0],
                'modality_id': self.field_mapping['rating_number'][1],
                'modality_type': 'number',
                'value': float(item_data['rating_number']),
                'item_id': item_id
            })
        
        return fields
    
    def encode_field(self, field_dict: Dict) -> torch.Tensor:
        """Encode a single field using the triplet approach."""
        # Get meta embeddings
        field_emb = self.field_encoder(torch.tensor(field_dict['field_id']))
        modality_emb = self.modality_encoder(torch.tensor(field_dict['modality_id']))
        
        # Check for precomputed embeddings first
        cache_key = f"{field_dict['item_id']}_{field_dict['field_name']}"
        if cache_key in self.precomputed_embeddings:
            value_emb = self.precomputed_embeddings[cache_key]
        else:
            # Encode value based on modality
            if field_dict['modality_type'] == 'text':
                value_emb = self.text_encoder(field_dict['value'])
            elif field_dict['modality_type'] == 'category':
                value_emb = self.category_encoder(field_dict['value'])
            elif field_dict['modality_type'] == 'image':
                value_emb = self.image_encoder(field_dict['value'])
            elif field_dict['modality_type'] == 'number':
                value_emb = self.number_encoder(field_dict['value'], field_dict['field_name'])
            
        # Combine triplet
        field_representation = field_emb + modality_emb + value_emb
        return field_representation
    
    def forward(self, item_ids: List[str], items_data: Dict[str, Dict]) -> torch.Tensor:
        """
        Args:
            item_ids: List of item IDs (parent_asins)
            items_data: Dictionary mapping item_id to item data
        
        Returns:
            Tensor of shape [batch_size, num_queries, hidden_dim]
        """
        batch_field_embeddings = []
        batch_field_masks = []
        max_fields = 0
        
        # Process each item
        for item_id in item_ids:
            item_data = items_data[item_id]
            fields = self.extract_fields_from_item(item_id, item_data)
            
            # Encode each field
            field_embeddings = []
            for field in fields:
                field_emb = self.encode_field(field)
                field_embeddings.append(field_emb)
            
            batch_field_embeddings.append(field_embeddings)
            max_fields = max(max_fields, len(field_embeddings))
        
        # Pad to max_fields in batch
        padded_embeddings = []
        field_masks = []
        
        for field_embeddings in batch_field_embeddings:
            num_fields = len(field_embeddings)
            
            # Create mask
            mask = torch.zeros(max_fields, dtype=torch.bool)
            mask[:num_fields] = True
            field_masks.append(mask)
            
            # Pad embeddings if necessary
            if num_fields < max_fields:
                padding = [torch.zeros_like(field_embeddings[0]) 
                          for _ in range(max_fields - num_fields)]
                field_embeddings.extend(padding)
            
            padded_embeddings.append(torch.stack(field_embeddings))
        
        # Stack for batch
        batch_embeddings = torch.stack(padded_embeddings)  # [batch_size, max_fields, hidden_dim]
        batch_masks = torch.stack(field_masks)  # [batch_size, max_fields]
        
        # Pass through Q-Former
        item_representations = self.q_former(batch_embeddings, batch_masks)
        
        return item_representations


# Supporting encoder classes
class TextEncoder(nn.Module):
    def __init__(self, model_name: str, output_dim: int):
        super().__init__()
        # Use a pretrained sentence transformer
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name)
        self.projection = nn.Linear(self.encoder.get_sentence_embedding_dimension(), output_dim)
    
    def forward(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            embedding = self.encoder.encode(text, convert_to_tensor=True)
        return self.projection(embedding)


class CategoryEncoder(nn.Module):
    def __init__(self, category_vocab: Optional[Dict[str, int]], output_dim: int):
        super().__init__()
        self.category_vocab = category_vocab or {}
        vocab_size = max(self.category_vocab.values()) + 2 if category_vocab else 10000
        self.embedding = nn.Embedding(vocab_size, output_dim)
        self.unk_idx = vocab_size - 1
    
    def forward(self, category: str) -> torch.Tensor:
        idx = self.category_vocab.get(category, self.unk_idx)
        return self.embedding(torch.tensor(idx))


class NumberEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        # Different projections for different number types
        self.projections = nn.ModuleDict({
            'price': nn.Linear(1, output_dim),
            'average_rating': nn.Linear(1, output_dim),
            'rating_number': nn.Linear(1, output_dim)
        })
        
        # Normalization parameters (can be learned or fixed)
        self.normalizers = {
            'price': lambda x: torch.log1p(x) / 10.0,  # Log scale for price
            'average_rating': lambda x: x / 5.0,  # Normalize to [0, 1]
            'rating_number': lambda x: torch.log1p(x) / 15.0  # Log scale for counts
        }
    
    def forward(self, value: float, field_name: str) -> torch.Tensor:
        normalized = self.normalizers[field_name](torch.tensor([value], dtype=torch.float32))
        return self.projections[field_name](normalized.unsqueeze(-1)).squeeze(0)


class ImageEncoder(nn.Module):
    def __init__(self, model_name: str, output_dim: int):
        super().__init__()
        # Use CLIP or similar vision model
        from transformers import CLIPVisionModel
        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, output_dim)
    
    def forward(self, image_url: str) -> torch.Tensor:
        # In practice, you'd load and preprocess the image
        # This is a placeholder
        image_tensor = load_and_preprocess_image(image_url)
        features = self.encoder(image_tensor).pooler_output
        return self.projection(features)