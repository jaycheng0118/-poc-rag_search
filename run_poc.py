"""
專案名稱：內部作業風險準則規範檢索器 (Hybrid RAG POC)
作者：Jay Cheng
目前版次：v1.2.0 (Last Updated: 2026-09)

版本更新紀錄 (Changelog):
- v1.0.0: 完成初版純向量 (Dense) 與 BM25 檢索，建立基礎 RRF 融合架構。
- v1.0.1: 引入 Cross-Encoder 進行二階段重排 (Rerank)，改善 Top-1 排序。
- v1.1.0: 新增 Cross-Encoder Logits 門檻防護（Anti-Hallucination Guardrail）。
- v1.2.0:
    1. 補上 Vanilla RAG (單純向量檢索) Baseline 對照組。
    2. 記錄法規文本轉向量時的四大痛點：條號稀釋、數值不敏感、多條件長句失真、餘弦分數假性偏高。

AI筆記：
實作筆記：為什麼 Vanilla RAG 應付內部法規會翻車？
- 條號與代碼弱化：all-MiniLM 遇到 "OP-RISK-302" 這類代號，會拆成多個子詞，無法像 BM25 做到 100% 精準捕捉。
- 數值門檻模糊：純向量算 Cosine 時，"3,000 萬" 和 "1,000 萬" 距離極近，無法區分權限邊界。
- 假性高分：庫外無效提問算出的 Cosine 仍常有 0.5 以上，難以直接設 Threshold 拒答。
"""

測試思路與實務踩坑筆記：
1. 為什麼要 Hybrid（Dense + Sparse）？
   - 業務同仁提問常夾帶「精確編號/金額/門檻」（如 OP-RISK-302、3,000 萬），純向量模型對高基數 ID 或生僻編號極易失真。
   - BM25 專門死咬關鍵字與條號，Dense 負責吃同義詞與模糊語意，兩者互補。
2. 分詞器限制（Tokenizer Caveat）：
   - 目前 BM25 使用簡易 split()，對中文單字抓取能力有限。若後續擴充至數萬篇規章，需換成 jieba 或字符級 n-gram。
3. 為什麼最後一定要掛 Cross-Encoder？
   - Bi-Encoder（Dense 模型）是 Query 和 Doc 分開算向量後點積，運算快但看不見細部 interaction。
   - Cross-Encoder 把 [Query, Doc] 一起丟進 Transformer 做 Full Attention，能抓到「3,000 萬 vs 1,000 萬」這種細微條件差異。
4. 拒絕回答機制（Guardrail）：
   - 模型不該「有問必答」。當重排最高分依然低於安全門檻時，強制截斷並回報查無資料，避免後續 LLM 瞎掰幻覺。
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# -------------------------------------------------------------
# 1. 模擬風管處內部作業風險準則規範 (Mock Knowledge Base)
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

# 輕量向量檢索模型 (POC 階段先用 all-MiniLM-L6-v2 驗證管線，正式環境建議換成 bge-m3 或 text2vec-base-chinese)
dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = dense_model.encode(CORPUS, convert_to_numpy=True)

# 關鍵字檢索 BM25 (以簡易 split 模擬分詞)
tokenized_corpus = [doc.lower().split() for doc in CORPUS]
bm25_model = BM25Okapi(tokenized_corpus)

# 二階段交叉編碼精排模型 (MS-MARCO 訓練權重，輸出未經過 Sigmoid 的 Logits 分數)
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -------------------------------------------------------------
# 3. 核心檢索與融合演算法
# -------------------------------------------------------------
def search_dense(query: str, top_k=2):
    """階段 1：語意向量召回 (基於餘弦/內積相似度)"""
    query_emb = dense_model.encode([query], convert_to_numpy=True)
    scores = np.dot(doc_embeddings, query_emb.T).flatten()
    return np.argsort(scores)[::-1][:top_k].tolist()

def search_bm25(query: str, top_k=2):
    """階段 2：關鍵字精確匹配召回 (鎖定編號、專有名詞與數值門檻)"""
    tokenized_query = query.lower().split()
    scores = bm25_model.get_scores(tokenized_query)
    return np.argsort(scores)[::-1][:top_k].tolist()

def rrf_fusion(sparse_ranks, dense_ranks, k=60):
    """
    Reciprocal Rank Fusion (RRF):
    - 不依賴分數絕對值（避免 BM25 與 Cosine 尺度不一致的問題），純粹用排名權重相加。
    - k 預設 60 為學界與實務通用平滑常數，可抑制極端高名次的主導性。
    """
    rrf_scores = {}
    for rank, idx in enumerate(sparse_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank + 1))
    for rank, idx in enumerate(dense_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank + 1))
    return sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

# -------------------------------------------------------------
# 4. 測試情境執行與防護評估
# -------------------------------------------------------------
def run_evaluation(query: str, threshold: float = -2.5):
    """
    執行端到端檢索流程並進行抗幻覺防護檢查
    threshold 設定思考：
    - ms-marco-MiniLM-L-6-v2 的輸出分數為 Logits。
    - 實測相關文本大多落在 0 ~ 8 之間；完全不相干或雜訊大多落於 -4 以下。
    - 門檻設在 -2.5 可有效過濾掉風馬牛不相干的提問，避免硬塞無效 context 給 LLM。
    """
    print("\n" + "="*80)
    print(f"【提問測試】: {query}")
    print("="*80)

    # 階段 1：純 Dense 向量檢索 Baseline
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

    print(f"[Phase 3: 混合檢索 + 精排 Top-1] -> {best_doc['doc_id']} (精排 Logits: {best_score:.4f})")

    # 階段 4：抗幻覺門檻防禦檢查 (Guardrails)
    if best_score < threshold:
        print(">> [防禦觸發]：精排分數低於安全門檻，系統判定：查無對應作業風險規範，拒絕生成回答以防幻覺。")
    else:
        print(f">> [檢索命中]：引用依據【{best_doc['doc_id']} {best_doc['title']}】")
        print(f">> [準則摘錄]：{best_doc['content']}")

if __name__ == "__main__":
    # 測試思路 1：精確條號 + 專用術語 (驗證 BM25 在少數關鍵字上的快速鎖定能力)
    run_evaluation("請依據 OP-RISK-302，衍生性商品單日名目本金超過多少需通報風管處？")

    # 測試思路 2：自然語意模糊提問，完全不提條號 (驗證 Dense 向量語意泛化與 Cross-Encoder 精準度)
    run_evaluation("新開戶沒幾個月的公司突然一直有大筆國外匯款進來，第一線該怎麼處理？")

    # 測試思路 3：與法規庫完全無關之異常/越界提問 (驗證抗幻覺截斷防護是否正常發揮)
    run_evaluation("資訊處同仁申請更換公務筆記型電腦之作業程序為何？")
