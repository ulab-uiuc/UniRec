import torch
import yaml
import json
import numpy as np
from typing import Dict, List, Any, Union
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor
from .mwne import load_trained_encoder
from PIL import Image
import io
import base64
import requests
import torch.nn.functional as F

class ItemEncoder:
    def __init__(self, config_path: str = "config/triplet_config.yaml"):
        """
        Initialize the ItemEncoder with all necessary encoders for different modalities.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.field_mapping = None
        self.modality_ids = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = 1024
        
        # Load configuration
        self._load_config()
        
        # Initialize encoders
        self._initialize_encoders()
    
    def _load_config(self):
        """Load the field mapping configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.field_mapping = config['FIELD_MAPPING']
        self.modality_ids = config['MODALITY_IDS']
        print(f"✓ Loaded configuration with {len(self.field_mapping)} fields")
    
    def _initialize_encoders(self):
        """Initialize encoders for each modality."""
        print("🔄 Initializing encoders...")
        
        # Text encoder (also used for categorical)
        print("  Loading text encoder...")
        self.text_encoder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        self.text_encoder.to(self.device)
        
        # CLIP encoder for image modality (using largest model with projection to 1024d)
        print("  Loading CLIP ViT-Large encoder with 1024d projection...")
        self.clip_model = AutoModel.from_pretrained(
            "openai/clip-vit-large-patch14", 
            torch_dtype=torch.float32,  # Use float32 for compatibility
            attn_implementation="sdpa"
        )
        self.clip_model.to(self.device)
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # The projection layer is removed, as we will use zero-padding instead.
        
        # Numerical encoder
        print("  Loading numerical encoder...")
        try:
            self.number_encoder = load_trained_encoder(
                "number_encoders/mathematical_encoder_1024d_normalized.pth"
            )
            self.number_encoder.to(self.device)
        except Exception as e:
            print(f"⚠️  Warning: Could not load numerical encoder: {e}")
            self.number_encoder = None
        
        print("✓ All encoders initialized")
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text using the sentence transformer."""
        if not text or text.strip() == "":
            # Return zero embedding for empty text
            return np.zeros(1024)  # Full Qwen3 dimensions
        
        embeddings = self.text_encoder.encode([text], convert_to_numpy=True)
        # Use full 1024d embeddings
        return embeddings[0]
    
    def _encode_text_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts efficiently."""
        # Find indices of non-empty texts to encode
        valid_indices = [i for i, text in enumerate(texts) if text and str(text).strip()]
        texts_to_encode = [texts[i] for i in valid_indices]
        
        result_embeddings = np.zeros((len(texts), 1024), dtype=np.float32)
        
        if not texts_to_encode:
            return result_embeddings
            
        # Encode all non-empty texts in batches for better memory management
        batch_size = 256  # Reduced to prevent CUDA OOM errors
        all_embeddings = []
        
        for batch_start in range(0, len(texts_to_encode), batch_size):
            batch_texts = texts_to_encode[batch_start:batch_start + batch_size]
            batch_embeddings = self.text_encoder.encode(
                batch_texts, 
                convert_to_numpy=True,
                batch_size=min(batch_size, 32),  # Reduced internal batch size to prevent OOM
                show_progress_bar=False
            )
            all_embeddings.extend(batch_embeddings)
        
        embeddings = np.array(all_embeddings)
        
        # Place the computed embeddings back into the correct positions
        for i, emb in zip(valid_indices, embeddings):
            result_embeddings[i] = emb
            
        return result_embeddings

    def _encode_image(self, image_data: str) -> np.ndarray:
        """
        Encode image using CLIP image encoder only.
        Returns pure image embeddings without text combination.
        
        Args:
            image_data: Image URL, file path, or base64 string
        """
        try:
            # Load image
            if image_data.startswith('http'):
                # URL
                import requests
                image = Image.open(requests.get(image_data, stream=True).raw).convert('RGB')
            elif image_data.startswith('data:image') or len(image_data) > 100:
                # Base64 encoded
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            else:
                # File path
                image = Image.open(image_data).convert('RGB')
            
            # Process with CLIP image encoder only
            # Use a dummy text to satisfy the processor requirements
            dummy_text = ["product image"]
            inputs = self.clip_processor(
                text=dummy_text,
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Get image embeddings only
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                image_emb = outputs.image_embeds[0]  # Shape: [768]
                # Pad the 768d embedding to 1024d with zeros
                padded_emb = F.pad(image_emb, (0, 1024 - 768), "constant", 0)
            
            return padded_emb.cpu().numpy()
            
        except Exception as e:
            print(f"⚠️  Warning: Could not encode image: {e}")
            # Return zero embedding for failed image encoding
            return np.zeros(1024)  # Projected CLIP image embeddings are 1024d

    def _encode_image_batch(self, image_data_list: List[str]) -> np.ndarray:
        """Encode a batch of images efficiently using CLIP with parallel URL downloading."""
        import concurrent.futures
        from functools import partial
        
        result_embeddings = np.zeros((len(image_data_list), 1024), dtype=np.float32)
        
        if not image_data_list:
            return result_embeddings
        
        # Separate URLs from other image data types
        url_indices = []
        url_list = []
        non_url_data = []
        non_url_indices = []
        
        for i, image_data in enumerate(image_data_list):
            if not image_data or not str(image_data).strip():
                continue
            
            if str(image_data).startswith('http'):
                url_indices.append(i)
                url_list.append(str(image_data))
            else:
                non_url_indices.append(i)
                non_url_data.append((i, str(image_data)))
        
        valid_indices = []
        images_to_process = []
        
        # Parallel download for URLs
        if url_list:
            def download_image(url_data):
                idx, url = url_data
                try:
                    response = requests.get(url, stream=True, timeout=10)
                    response.raise_for_status()
                    image = Image.open(response.raw).convert('RGB')
                    return (idx, image)
                except Exception as e:
                    print(f"⚠️  Failed to download image from {url}: {e}")
                    return (idx, None)
            
            # Use ThreadPoolExecutor for parallel downloading (increased workers for better throughput)
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                url_results = list(executor.map(download_image, zip(url_indices, url_list)))
            
            # Add successfully downloaded images
            for idx, image in url_results:
                if image is not None:
                    valid_indices.append(idx)
                    images_to_process.append(image)
        
        # Process non-URL data (base64, file paths) sequentially (usually fast)
        for i, image_data in non_url_data:
            try:
                if image_data.startswith('data:image') or len(image_data) > 100:
                    # Base64 encoded
                    if image_data.startswith('data:image'):
                        image_data = image_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                else:
                    # File path
                    image = Image.open(image_data).convert('RGB')
                
                valid_indices.append(i)
                images_to_process.append(image)
            except Exception as e:
                print(f"⚠️  Failed to process image data: {e}")
                continue
        
        if not images_to_process:
            return result_embeddings
        
        # Batch encode all successfully loaded images (process in chunks for memory efficiency)
        chunk_size = 32  # Reduced to prevent CUDA OOM errors with CLIP model
        all_embeddings = []
        
        for chunk_start in range(0, len(images_to_process), chunk_size):
            chunk_images = images_to_process[chunk_start:chunk_start + chunk_size]
            inputs = self.clip_processor(images=chunk_images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                image_embeds_768d = self.clip_model.get_image_features(**inputs)
                # Pad the batch of 768d embeddings to 1024d
                image_embeds_1024d = F.pad(image_embeds_768d, (0, 1024 - 768), "constant", 0)
                all_embeddings.append(image_embeds_1024d.cpu())
        
        # Concatenate all chunks
        if all_embeddings:
            image_embeds_1024d = torch.cat(all_embeddings, dim=0)

        # Place embeddings back in correct positions
        if all_embeddings:
            for i, emb in zip(valid_indices, image_embeds_1024d.numpy()):
                result_embeddings[i] = emb
            
        return result_embeddings

    def _encode_number(self, number: Union[float, int]) -> np.ndarray:
        """Encode numerical value using the mathematical encoder."""
        if self.number_encoder is None:
            print(f"❌ Error: Number encoder not available. Cannot encode number {number}")
            raise RuntimeError("Number encoder not available. Please ensure the MWNE encoder is properly loaded.")
        
        try:
            # Convert to tensor and encode
            number_tensor = torch.tensor([float(number)], device=self.device)
            with torch.no_grad():
                embedding = self.number_encoder(number_tensor)  # Shape: [1024]
                # Apply L2 normalization to ensure unit norm
                embedding_norm = torch.norm(embedding, p=2, dim=-1, keepdim=True)
                embedding_normalized = embedding / (embedding_norm + 1e-8)
                return embedding_normalized.cpu().numpy()[0]
        except Exception as e:
            print(f"❌ Error: Could not encode number {number}: {e}")
            raise RuntimeError(f"Failed to encode number {number}: {e}")

    def _encode_number_batch(self, numbers: List[Union[float, int, str]]) -> np.ndarray:
        """Encode a batch of numerical values, handling bad data."""
        if self.number_encoder is None:
            raise RuntimeError("Number encoder not available.")
        
        # Sanitize input: convert to float, default to 0.0 for invalid/empty values
        sanitized_numbers = []
        for n in numbers:
            try:
                sanitized_numbers.append(float(n))
            except (ValueError, TypeError):
                sanitized_numbers.append(0.0)

        number_tensor = torch.tensor(sanitized_numbers, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            embeddings = self.number_encoder(number_tensor)
            return F.normalize(embeddings, p=2, dim=1).cpu().numpy()

    def encode_sample(self, sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Encode a single sample (JSON object) into embeddings for each field.
        
        Args:
            sample: Dictionary containing the item data
            
        Returns:
            Dictionary mapping field names to their embeddings
        """
        embeddings = {}
        
        for field_name, field_info in self.field_mapping.items():
            field_id, modality_id, modality_type = field_info
            
            # Get the field value
            field_value = sample.get(field_name, "")
            
            # Encode based on modality type
            if modality_type == "text":
                embeddings[field_name] = self._encode_text(str(field_value))
            
            elif modality_type == "category":
                # Treat categorical as text
                embeddings[field_name] = self._encode_text(str(field_value))
            
            elif modality_type == "image":
                # Skip empty image fields
                if field_value and str(field_value).strip():
                    embeddings[field_name] = self._encode_image(str(field_value))
                else:
                    embeddings[field_name] = np.zeros(1024)  # Zero embedding for missing images
            
            elif modality_type == "number":
                # Convert to float, handle None/empty values
                try:
                    num_value = float(field_value) if field_value not in [None, "", "null"] else 0.0
                    embeddings[field_name] = self._encode_number(num_value)
                except (ValueError, TypeError) as e:
                    print(f"❌ Error: Invalid number value '{field_value}' for field '{field_name}': {e}")
                    raise RuntimeError(f"Invalid number value '{field_value}' for field '{field_name}': {e}")
                except RuntimeError as e:
                    # Re-raise RuntimeError from _encode_number
                    raise e
            
            else:
                print(f"❌ Error: Unknown modality type: {modality_type} for field {field_name}")
                raise ValueError(f"Unknown modality type: {modality_type} for field {field_name}")
        
        return embeddings
    
    def encode_batch_by_field(self, samples: List[Dict[str, Any]], fields_to_encode: List[str]) -> Dict[str, np.ndarray]:
        """
        Encodes a batch of samples and returns a dictionary where keys are field names
        and values are numpy arrays of embeddings for that field across the batch.
        """
        if not samples:
            return {field: np.array([]) for field in fields_to_encode}

        # 1. Collect data for each field
        field_batches = {field: [s.get(field, "") for s in samples] for field in fields_to_encode}

        # 2. Encode each field's batch based on its modality
        encoded_field_batches = {}
        for field_name in fields_to_encode:
            field_info = self.field_mapping.get(field_name)
            if not field_info:
                print(f"Warning: Field '{field_name}' not in field_mapping. Skipping.")
                encoded_field_batches[field_name] = np.zeros((len(samples), 1024))
                continue

            modality_type = field_info[2]
            data_batch = field_batches[field_name]
            
            if modality_type in ["text", "category"]:
                encoded_field_batches[field_name] = self._encode_text_batch(data_batch)
            elif modality_type == "image":
                encoded_field_batches[field_name] = self._encode_image_batch(data_batch)
            elif modality_type == "number":
                encoded_field_batches[field_name] = self._encode_number_batch(data_batch)
            else:
                raise ValueError(f"Unknown modality type: {modality_type}")
                
        return encoded_field_batches

    def encode_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, np.ndarray]]:
        """
        Encode a batch of samples. Wrapper around encode_batch_by_field.
        """
        if not samples:
            return []
            
        fields_to_encode = list(self.field_mapping.keys())
        encoded_by_field = self.encode_batch_by_field(samples, fields_to_encode)
        
        # Re-assemble results into a list of dictionaries
        results = [{} for _ in range(len(samples))]
        for field_name, embeddings in encoded_by_field.items():
            for i, embedding in enumerate(embeddings):
                results[i][field_name] = embedding
                
        return results
    
    def get_embedding_dimensions(self) -> Dict[str, int]:
        """Get the embedding dimensions for each field."""
        dimensions = {}
        for field_name, field_info in self.field_mapping.items():
            modality_type = field_info[2]
            
            if modality_type in ["text", "category"]:
                dimensions[field_name] = 1024  # Full Qwen3 embeddings
            elif modality_type == "number":
                dimensions[field_name] = 1024  # MWNE 1024d encoder
            elif modality_type == "image":
                dimensions[field_name] = 1024  # Projected CLIP image embeddings dimensions
            else:
                dimensions[field_name] = 1024  # Default
        
        return dimensions


def main():
    """Example usage of the ItemEncoder."""
    # Initialize encoder
    encoder = ItemEncoder()
    
    # Example sample (you would load this from your JSON file)
    example_sample = {
        "title": "Nike Air Max Running Shoes",
        "description": "Comfortable running shoes with excellent cushioning",
        "features": "Lightweight, breathable, durable",
        "main_category": "Footwear",
        "store": "Nike Store",
        "brand": "Nike",
        "style": "Athletic",
        "color": "Black",
        "size": "10",
        "material": "Mesh",
        "main_image": "path/to/image.jpg",  # or base64 encoded image
        "price": 129.99,
        "average_rating": 4.5,
        "rating_number": 1250
    }
    
    print("\n🔍 Encoding sample...")
    embeddings = encoder.encode_sample(example_sample)
    
    print("\n📊 Encoded embeddings:")
    for field_name, embedding in embeddings.items():
        print(f"  {field_name}: {embedding.shape} (norm: {np.linalg.norm(embedding):.4f})")
    
    # Get embedding dimensions
    dimensions = encoder.get_embedding_dimensions()
    print(f"\n📏 Embedding dimensions: {dimensions}")


if __name__ == "__main__":
    main()
