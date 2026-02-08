#!/usr/bin/env python3
"""
Script to create All_Beauty_item_triplet_dict.json by filtering
All_Beauty_item_dict.json based on fields specified in config/triplet_config.yaml
"""

import json
import yaml
import os

def load_config(config_path):
    """Load the triplet configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def extract_fields_from_details(details, field_mapping):
    """Extract specific fields from the details dictionary"""
    extracted = {}
    
    # Map details fields to config fields
    field_mapping_details = {
        'Brand': 'brand',
        'Style': 'style', 
        'Color': 'color',
        'Size': 'size',
        'Material': 'material'
    }
    
    for detail_key, detail_value in details.items():
        if detail_key in field_mapping_details:
            config_field = field_mapping_details[detail_key]
            if config_field in field_mapping:
                extracted[config_field] = detail_value
    
    return extracted

def process_images(images):
    """Process images to extract main_image only"""
    if not images:
        return None
    
    # Find main image (variant = "MAIN")
    main_img = None
    
    for img in images:
        if img.get('variant') == 'MAIN':
            main_img = img.get('large') or img.get('hi_res')
            break
    
    return main_img

def filter_item_data(item_data, field_mapping):
    """Filter item data to keep only fields specified in config"""
    filtered_data = {}
    
    # Direct field mappings
    direct_fields = [
        'title', 'main_category', 'store', 'price', 'average_rating', 'rating_number'
    ]
    
    for field in direct_fields:
        if field in field_mapping and field in item_data:
            filtered_data[field] = item_data[field]
    
    # Handle description and features - convert lists to strings
    if 'description' in field_mapping and 'description' in item_data:
        desc = item_data['description']
        if isinstance(desc, list):
            filtered_data['description'] = ' '.join(desc) if desc else ''
        else:
            filtered_data['description'] = str(desc) if desc else ''
    
    if 'features' in field_mapping and 'features' in item_data:
        features = item_data['features']
        if isinstance(features, list):
            filtered_data['features'] = ' '.join(features) if features else ''
        else:
            filtered_data['features'] = str(features) if features else ''
    
    # Extract fields from details
    if 'details' in item_data:
        details_fields = extract_fields_from_details(item_data['details'], field_mapping)
        filtered_data.update(details_fields)
    
    # Process images
    if 'images' in item_data:
        main_img = process_images(item_data['images'])
        if 'main_image' in field_mapping and main_img:
            filtered_data['main_image'] = main_img
    
    return filtered_data

def main():
    # File paths
    config_path = 'config/triplet_config.yaml'
    input_path = 'data_rec/dict/All_Beauty_item_dict.json'
    output_path = 'data_rec/dict/All_Beauty_item_triplet_dict.json'
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found!")
        return
    
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found!")
        return
    
    print("Loading configuration...")
    config = load_config(config_path)
    field_mapping = config['FIELD_MAPPING']
    
    print("Loading input dictionary...")
    with open(input_path, 'r', encoding='utf-8') as f:
        item_dict = json.load(f)
    
    print(f"Processing {len(item_dict)} items...")
    
    # Filter each item
    filtered_dict = {}
    for asin, item_data in item_dict.items():
        filtered_data = filter_item_data(item_data, field_mapping)
        if filtered_data:  # Only keep items with some data
            filtered_dict[asin] = filtered_data
    
    print(f"Writing filtered dictionary with {len(filtered_dict)} items...")
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully created {output_path}")
    print(f"Original items: {len(item_dict)}")
    print(f"Filtered items: {len(filtered_dict)}")

if __name__ == "__main__":
    main() 