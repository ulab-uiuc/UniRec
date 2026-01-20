import json

input_path = 'data_rec/temp/meta_All_Beauty.jsonl'
output_path = 'data_rec/dict/All_Beauty_item_dict.json'

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
