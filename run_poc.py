import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# -------------------------------------------------------------
# 1. 模擬風管處內部作業風險準則規範 (脫敏合成資料)
# -------------------------------------------------------------
MOCK_GUIDELINES = [
    {
        "doc_id": "OP-RISK-101",
        "title": "授信業務作業風險控管細則",
        "content": "依據 OP-RISK-101，分行辦理法人無擔保授信案件，單一集團暴險金額超過新台幣 3,000 萬元時，除應取得雙簽核定外，必須呈報風管處專案小組審查，不得以分期批覆規避權限。"
    },
    {
        "doc_id": "OP-RISK-302",
        "title": "金融交易與衍生性商品作業風險規範",
        "content": "依據 OP-RISK-302，交易室前台進行複雜型外匯衍生性商品交易時，若單日未平倉名目本金超過 1,000 萬美元，應即刻將部位確認單（Confirmation）同步抄送風險管理處市場風管組進行跨日監控。"
    },
    {
        "doc_id": "OP-RISK-505",
        "title": "洗錢防制與高風險態樣排查準則",
        "content": "依據 OP-RISK-505，若法人帳戶開立未滿三個月且連續三日內自境外匯入款項頻率異常增加 5 倍以上，審查人員應暫停其網銀大額交易功能，並於 24 小時內通報風管處防制洗錢專案團隊。"
    }
]

CORPUS = [doc["content"] for doc in MOCK_GUIDELINES]

# -------------------------------------------------------------
# 2. 檢索器初始化 (Dense + Sparse + Cross-Encoder)
# -------------------------------------------------------------
print("[Init] 正在載入檢索與重排模型...")
# 輕量語意向量模型
dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = dense_model.encode(CORPUS, convert_to_numpy=True)

# 關鍵字 BM25 模型 (以簡易空格/分詞模擬)
tokenized_corpus = [doc.lower().split() for doc in CORPUS]
bm25_model = BM25Okapi(tokenized_corpus)

# 交叉編碼精排模型
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -------------------------------------------------------------
# 3. 核心檢索與融合演算法
# -------------------------------------------------------------
def search_dense(query: str, top_k=2):
    query_emb = dense_model.encode([query], convert_to_numpy=True)
    scores = np.dot(doc_embeddings, query_emb.T).flatten()
    return np.argsort(scores)[::-1][:top_k].tolist()

def search_bm25(query: str, top_k=2):
    tokenized_query = query.lower().split()
    scores = bm25_model.get_scores(tokenized_query)
    return np.argsort(scores)[::-1][:top_k].tolist()

def rrf_fusion(sparse_ranks, dense_ranks, k=60):
    rrf_scores = {}
    for rank, idx in enumerate(sparse_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank + 1))
    for rank, idx in enumerate(dense_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank + 1))
    return sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

# -------------------------------------------------------------
# 4. 測試情境執行
# -------------------------------------------------------------
def run_evaluation(query: str, threshold: float = -2.0):
    print("\n" + "="*80)
    print(f"【提問測試】: {query}")
    print("="*80)

    # 階段 1：純 Dense 向量檢索 (Baseline)
    dense_hits = search_dense(query, top_k=1)
    print(f"[Phase 1: 純向量檢索 Top-1] -> {MOCK_GUIDELINES[dense_hits[0]]['doc_id']}")

    # 階段 2：純 BM25 關鍵字檢索
    sparse_hits = search_bm25(query, top_k=2)
    print(f"[Phase 2: 純 BM25 關鍵字 Top-1] -> {MOCK_GUIDELINES[sparse_hits[0]]['doc_id']}")

    # 階段 3：雙路混合 (RRF) + Cross-Encoder 重排
    fused_candidates = rrf_fusion(sparse_hits, search_dense(query, top_k=2))
    candidate_docs = [MOCK_GUIDELINES[i] for i in fused_candidates]

    pairs = [[query, doc["content"]] for doc in candidate_docs]
    rerank_scores = rerank_model.predict(pairs)

    best_idx = int(np.argmax(rerank_scores))
    best_score = float(rerank_scores[best_idx])
    best_doc = candidate_docs[best_idx]

    print(f"[Phase 3: 混合檢索 + 精排 Top-1] -> {best_doc['doc_id']} (精排評分: {best_score:.4f})")

    # 階段 4：抗幻覺門檻防禦檢查 (Guardrails)
    if best_score < threshold:
        print(">> [防禦觸發]：精排分數低於安全門檻，系統判定：查無對應作業風險規範，拒絕生成回答以防幻覺。")
    else:
        print(f">> [檢索命中]：引用依據【{best_doc['doc_id']} {best_doc['title']}】")
        print(f">> [準則摘錄]：{best_doc['content']}")

if __name__ == "__main__":
    # 測試 1：精確條號與業務條件混合提問 (驗證條號精準定位)
    run_evaluation("請依據 OP-RISK-302，衍生性商品單日名目本金超過多少需通報風管處？")

    # 測試 2：法規庫外提問 (驗證抗幻覺截斷防護)
    run_evaluation("資訊處同仁申請更換公務筆記型電腦之作業程序為何？")
