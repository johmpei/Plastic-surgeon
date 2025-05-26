
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.idx")

with open("doc_map.txt", "r", encoding="utf-8") as f:
    doc_map = {int(line.split("\t")[0]): line.strip().split("\t")[1] for line in f}

query = input("🔍 検索したいキーワードや質問を入力してね：\n> ")
q_vector = model.encode([query], convert_to_numpy=True)
D, I = index.search(q_vector, k=3)

print("\n🔎 関連するページ：")
for rank, idx in enumerate(I[0]):
    print(f"{rank+1}. {doc_map[idx]}")
