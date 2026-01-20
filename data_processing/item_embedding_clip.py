import os

import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import json
from tqdm import tqdm
from math import ceil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Helper functions for extracting text and images from item dict
def extract_text(item):
    text_parts = [item.get('title', '')]
    if item.get('features'):
        text_parts.append(' '.join(item['features']))
    if item.get('description'):
        text_parts.append(' '.join(item['description']))
    if item.get('details'):
        details_str = ', '.join([f"{k}: {v}" for k, v in item['details'].items()])
        text_parts.append(details_str)
    return ' '.join(text_parts)

def get_main_images(item):
    return [img['large'] for img in item.get('images', []) if 'large' in img]

def load_image(url):
    try:
        return Image.open(requests.get(url, stream=True).raw).convert('RGB')
    except Exception:
        return None

def load_images_parallel(urls, texts, ids, max_workers=8):
    images = [None] * len(urls)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(load_image, url): idx for idx, url in enumerate(urls)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            img = future.result()
            images[idx] = img
    # Filter out items where image loading failed
    valid_images = []
    valid_texts = []
    valid_ids = []
    for i, img in enumerate(images):
        if img is not None:
            valid_images.append(img)
            valid_texts.append(texts[i])
            valid_ids.append(ids[i])
    return valid_images, valid_texts, valid_ids

model = AutoModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model = model.to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load item dict
item_dict_path = "data_rec/dict/All_Beauty_item_dict.json"
item_dict = json.load(open(item_dict_path, "r"))

# Load train and test files and collect all unique item IDs
train_json_path = "data_rec/data/Amazon_All_Beauty_all_train_LRanker.json"
test_json_path = "data_rec/data/Amazon_All_Beauty_all_test_LRanker.json"

unique_item_ids = set()
for json_path in [train_json_path, test_json_path]:
    with open(json_path, "r") as f:
        data = json.load(f)
    for sample in data:
        unique_item_ids.update(sample['history'])
        unique_item_ids.update(sample['candidate'])
        if 'ground_truth' in sample:
            unique_item_ids.add(sample['ground_truth'])

print(f"Total unique items to embed: {len(unique_item_ids)}")

# Embedding pipeline
embeddings = {}

def batch_iterable(iterable, batch_size):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]

item_ids = list(unique_item_ids)
batch_size = 128  # Adjust based on your GPU memory

for batch_ids in tqdm(batch_iterable(item_ids, batch_size), total=ceil(len(item_ids)/batch_size)):
    batch_texts = []
    batch_image_urls = []
    batch_valid_ids = []
    for item_id in batch_ids:
        item = item_dict.get(item_id)
        if item is None:
            continue
        text = extract_text(item)
        image_urls = get_main_images(item)
        if not image_urls:
            continue
        batch_texts.append(text)
        batch_image_urls.append(image_urls[0])  # Only use the first image per item
        batch_valid_ids.append(item_id)
    batch_images, batch_texts, batch_valid_ids = load_images_parallel(batch_image_urls, batch_texts, batch_valid_ids, max_workers=32)
    if not batch_images:
        continue
    inputs = processor(
        text=batch_texts,
        images=batch_images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    for idx, item_id in enumerate(batch_valid_ids):
        text_emb = outputs.text_embeds[idx]
        image_emb = outputs.image_embeds[idx]
        combined_emb = torch.stack([text_emb, image_emb], dim=0).mean(dim=0)
        embeddings[item_id] = combined_emb.cpu().tolist()

# Save embeddings to file
output_path = "data_rec/embeddings/all_beauty_item_embedding_clip.json"
with open(output_path, "w") as f:
    json.dump(embeddings, f)

print(f"Saved {len(embeddings)} item embeddings to {output_path}")