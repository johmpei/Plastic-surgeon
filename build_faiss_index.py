
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = []
doc_names = []

for filename in sorted(os.listdir("converted_text")):
    with open(f"converted_text/{filename}", "r", encoding="utf-8") as f:
        text = f.read().strip()
        if text:
            docs.append(text)
            doc_names.append(filename)

embeddings = model.encode(docs, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "faiss_index.idx")

with open("doc_map.txt", "w", encoding="utf-8") as f:
    for i, name in enumerate(doc_names):
        f.write(f"{i}\t{name}\n")

print("✅ FAISSインデックスを作成＆保存したよ！")
