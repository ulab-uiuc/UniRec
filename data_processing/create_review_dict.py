import json

input_path = 'data_rec/temp/All_Beauty.jsonl'
output_path = 'data_rec/dict/All_Beauty_review_dict.json'

review_dict = {}

with open(input_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        if not line.strip():
            continue
        obj = json.loads(line)
        user_id = obj.pop('user_id', None)
        parent_asin = obj.pop('parent_asin', None)
        if user_id is not None and parent_asin is not None:
            key = f"{user_id}|{parent_asin}"
            review_dict[key] = obj

with open(output_path, 'w', encoding='utf-8') as outfile:
    json.dump(review_dict, outfile)
