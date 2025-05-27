from flask import Flask, request, jsonify
import faiss
import numpy as np
import openai
import os
from dotenv import load_dotenv
import json

# --- FlaskとOpenAI初期化 ---
app = Flask(__name__)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- OpenAI埋め込み取得関数 ---
def get_embedding(text, model="text-embedding-3-small"):
    try:
        response = openai.Embedding.create(
            input=[text],  # 🔑 リスト形式が確実
            model=model
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        print("❌ 埋め込み取得エラー:", e)
        raise

# --- 検索APIエンドポイント ---
@app.route("/search")
def search():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "検索クエリ（query）を指定してください"}), 400

    try:
        # 🔁 クエリのたびに読み込むことでRender Freeでも耐える
        index = faiss.read_index("faiss_index.idx")
        with open("doc_map.txt", "r", encoding="utf-8") as f:
            doc_map = json.load(f)

        # 🔍 クエリをベクトル化
        q_vector = get_embedding(query)
        q_vector = np.array([q_vector], dtype=np.float32)

        # 🔎 FAISS検索
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
        return jsonify({
            "error": "内部エラーが発生しました",
            "detail": str(e)
        }), 500

# --- Render対応：PORT環境変数で起動 ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
