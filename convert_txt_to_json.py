# convert_txt_to_json.py

import json

with open("doc_map.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

doc_map = {}
for line in lines:
    parts = line.strip().split(":")
    if len(parts) == 2:
        key = parts[0].strip()
        value = parts[1].strip()
        doc_map[key] = value

with open("doc_map.json", "w", encoding="utf-8") as f:
    json.dump(doc_map, f, ensure_ascii=False, indent=2)

print("✅ doc_map.json に変換完了！")
