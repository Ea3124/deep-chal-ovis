import json, sys

with open("test.json", encoding="utf-8") as f:
    data = json.load(f)

for i, item in enumerate(data):
    item["id"] = i

with open("test_reindexed.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
