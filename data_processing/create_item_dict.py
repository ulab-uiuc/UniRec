import json
import os

# Dataset name; override with `UNIREC_DATASET=<name>` to use a different Amazon category.
DATASET_NAME = os.environ.get("UNIREC_DATASET", "Beauty_and_Personal_Care")

input_path = f'data_rec/temp/meta_{DATASET_NAME}.jsonl'
output_path = f'data_rec/dict/{DATASET_NAME}_item_dict.json'

item_dict = {}

with open(input_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        if not line.strip():
            continue
        obj = json.loads(line)
        parent_asin = obj.pop('parent_asin', None)
        if parent_asin is not None:
            item_dict[parent_asin] = obj

with open(output_path, 'w', encoding='utf-8') as outfile:
    json.dump(item_dict, outfile)
