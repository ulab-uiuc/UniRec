import os

import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import json

# === Load all data files at the beginning ===
# File paths
train_path = 'data_rec/data/Amazon_All_Beauty_all_train_LRanker.json'
review_dict_path = 'data_rec/dict/All_Beauty_review_dict.json'
item_embedding_path = 'data_rec/embeddings/all_beauty_item_embedding_clip.json'
item_dict_path = 'data_rec/dict/All_Beauty_item_dict.json'

# Load train file and get the first element
with open(train_path, 'r') as f:
    train_data = json.load(f)
first_obj = train_data[0]
user_id = first_obj['user_id']
history = first_obj['history']
if not history:
    raise ValueError('No history for first user')

# Load review dictionary
with open(review_dict_path, 'r') as f:
    review_dict = json.load(f)

# Load item embeddings
with open(item_embedding_path, 'r') as f:
    item_embeddings = json.load(f)

# Load item metadata dict
with open(item_dict_path, 'r') as f:
    item_dict = json.load(f)

def get_main_images(item):
    return [img['large'] for img in item.get('images', []) if 'large' in img]

# === CLIP Embedding Extraction for all history reviews ===

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model = model.to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

batch_texts = []
batch_images = []
valid_item_ids = []
item_embedding_tensors = []

for item_id in history:
    key = f'{user_id}|{item_id}'
    review = review_dict.get(key)
    item_embedding = item_embeddings.get(item_id)
    if review is None or item_embedding is None:
        continue
    # Prepare text
    text = (review.get("title", "") + " " + review.get("text", "")).strip()
    # Prepare image (if any)
    img = None
    # Try review image first
    if review.get("images"):
        images = review["images"]
        if images:
            url = images[0]
            try:
                img = Image.open(requests.get(url, stream=True).raw).convert('RGB')
            except Exception:
                img = None
    # If no review image, fallback to item image
    if img is None:
        item = item_dict.get(item_id)
        if item is not None:
            image_urls = get_main_images(item)
            if image_urls:
                try:
                    img = Image.open(requests.get(image_urls[0], stream=True).raw).convert('RGB')
                except Exception:
                    img = None
    batch_texts.append(text)
    batch_images.append(img)
    valid_item_ids.append(item_id)
    item_embedding_tensors.append(torch.tensor(item_embedding, device=device).unsqueeze(0))

if not batch_texts:
    print('No valid reviews with item embeddings found in history.')
    exit(1)

# Remove None images and corresponding texts/ids/embeddings
filtered_texts = []
filtered_images = []
filtered_item_ids = []
filtered_item_embedding_tensors = []
for i, img in enumerate(batch_images):
    if img is not None:
        filtered_texts.append(batch_texts[i])
        filtered_images.append(img)
        filtered_item_ids.append(valid_item_ids[i])
        filtered_item_embedding_tensors.append(item_embedding_tensors[i])

if not filtered_texts:
    print('No valid reviews with images found in history.')
    exit(1)

inputs = processor(
    text=filtered_texts,
    images=filtered_images,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=77,
)
inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)

# Average text and image embeddings for each review
review_embeds = []
for idx in range(len(filtered_texts)):
    text_emb = outputs.text_embeds[idx]
    image_emb = outputs.image_embeds[idx]
    combined_emb = torch.stack([text_emb, image_emb], dim=0).mean(dim=0)
    review_embeds.append(combined_emb.unsqueeze(0))
review_embeds = torch.cat(review_embeds, dim=0)  # [batch, hidden]

# Stack item embeddings to match batch
item_embedding_tensor = torch.cat(filtered_item_embedding_tensors, dim=0)  # [batch, hidden]

# Concatenate review and item embeddings
combined_embedding = torch.cat([review_embeds, item_embedding_tensor], dim=1)  # [batch, 2*hidden]

print(f"Review embedding batch shape: {review_embeds.shape}")
print(f"Item embedding batch shape: {item_embedding_tensor.shape}")
print(f"Combined embedding batch shape: {combined_embedding.shape}") 