from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 検索用の準備
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.idx")

with open("doc_map.txt", "r", encoding="utf-8") as f:
    doc_map = {int(line.split("\t")[0]): line.strip().split("\t")[1] for line in f}

# Flaskアプリ起動
app = Flask(__name__)

@app.route("/search")
def search():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "検索キーワードを指定してください！"}), 400

    q_vector = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vector, k=3)

    results = []
    for idx in I[0]:
        results.append({
            "page": doc_map.get(idx),
            "score": float(D[0][list(I[0]).index(idx)])
        })

    return jsonify({
        "query": query,
        "results": results
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
