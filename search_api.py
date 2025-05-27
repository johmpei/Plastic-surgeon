from flask import Flask, request, jsonify
import faiss
import numpy as np
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# --- 初期化 ---
app = Flask(__name__)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 埋め込み取得関数（OpenAI最新版対応） ---
def get_embedding(text, model="text-embedding-3-small"):
    try:
        response = client.embeddings.create(
            model=model,
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

# --- /search エンドポイント ---
@app.route("/search")
def search():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "検索クエリ（query）を指定してください"}), 400

    try:
        index = faiss.read_index("faiss_index.idx")
        with open("doc_map.txt", "r", encoding="utf-8") as f:
            doc_map = json.load(f)

        q_vector = get_embedding(query)
        if q_vector is None:
            return jsonify({"error": "OpenAI埋め込みに失敗しました"}), 500

        q_vector = np.array([q_vector], dtype=np.float32)

        D, I = index.search(q_vector, k=3)
        results = []
        for idx in I[0]:
            results.append({
                "page": doc_map.get(str(idx), "不明"),
                "score": float(D[0][list(I[0]).index(idx)])
            })

        return jsonify({"query": query, "results": results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "内部エラー", "detail": str(e)}), 500

# --- トップページ：検索フォーム表示 ---
@app.route("/")
def home():
    return '''
    <html>
      <head><meta charset="utf-8"><title>チャッピー検索</title></head>
      <body>
        <h1>🔍 チャッピー ベクトル検索</h1>
        <form action="/search" method="get">
          <label>検索ワード：</label>
          <input type="text" name="query" placeholder="例：巻き爪" required>
          <button type="submit">検索</button>
        </form>
      </body>
    </html>
    ''', 200

# --- Render対応ポート設定 ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
