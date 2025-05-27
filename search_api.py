from flask import Flask, request, jsonify
import faiss
import numpy as np
import openai
import os
from dotenv import load_dotenv
import json

# --- 初期化 ---
app = Flask(__name__)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- 安定版埋め込み取得関数 ---
def get_embedding(text, model="text-embedding-3-small"):
    try:
        response = openai.Embedding.create(
            input=[text],  # ← かならずリスト形式にする！
            model=model
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        print("❌ OpenAI埋め込みエラー:", e)
        return None

# --- /search エンドポイント ---
@app.route("/search")
def search():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "検索クエリ（query）を指定してください"}), 400

    try:
        # 🔁 インデックスとマップを都度読み込み（Render Free対策）
        index = faiss.read_index("faiss_index.idx")
        with open("doc_map.txt", "r", encoding="utf-8") as f:
            doc_map = json.load(f)

        # 🔍 クエリをベクトル化
        q_vector = get_embedding(query)
        if q_vector is None:
            return jsonify({
                "error": "OpenAI埋め込みに失敗しました（キー・ネット・モデル確認）",
                "detail": "get_embedding() returned None"
            }), 500

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

# --- トップページ用（アクセス確認用） ---
@app.route("/")
def home():
    return "✅ チャッピーベクトルAIへようこそ！ → /search?query=巻き爪", 200

# --- Render用ポート設定 ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
