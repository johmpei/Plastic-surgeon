from sentence_transformers import SentenceTransformer

# ✅ グローバルに読み込んでおく（起動時に一度だけ）
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

@app.route("/search")
def search():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "検索キーワードを指定してください！"}), 400

    # ❌ もうここで読み込まない → modelは上で定義済
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
