from flask import Flask, request, jsonify
import faiss
import numpy as np
import openai
import os
from dotenv import load_dotenv
import json

app = Flask(__name__)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-3-small"):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']

@app.route("/search")
def search():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "クエリが必要です"}), 400

    try:
        # 🔽 クエリごとにインデックスとマップを読み込む（メモリ節約）
        index = faiss.read_index("faiss_index.idx")
        with open("doc_map.txt", "r", encoding="utf-8") as f:
            doc_map = json.load(f)

        q_vector = get_embedding(query)
        q_vector = np.array([q_vector], dtype=np.float32)

        D, I = index.search(q_vector, k=3)
        results = []
        for idx in I[0]:
            results.append({
                "page": doc_map.get(str(idx), "不明"),
                "score": float(D[0][list(I[0]).index(idx)])
            })

        return jsonify({
            "query": query,
            "results": results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "内部エラー", "detail": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
