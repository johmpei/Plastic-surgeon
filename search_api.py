from flask import Flask, request, jsonify
import faiss
import numpy as np
import os

app = Flask(__name__)

# 🔹 起動時に重たい SentenceTransformer を読み込まない
# 🔹 代わりに、インデックスとページマップだけ読み込む
index = faiss.read_index("faiss_index.idx")

with open("doc_map.txt", "r", encoding="utf-8") as f:
    doc_map = {int(line.split("\t")[0]): line.strip().split("\t")[1] for line in f}

@app.route("/search")
def search():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "検索キーワードを指定してください！"}), 400

    # 🔥 モデルはリクエストのたびに読み込む（ここがポイント！）
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
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

# 🔧 Renderで必要なポート設定
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
