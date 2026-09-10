# poc-rag_search
Production-grade POC for enterprise SOP &amp; compliance retrieval: combining BM25, dense embeddings (BGE), RRF fusion, and Cross-Encoder reranking to minimize RAG hallucinations.
# 企業級規範與 SOP 之 RAG 檢索優化研究 (POC)

本專案記錄在內部規範與技術手冊場景下，針對「檢索精度不足導致 LLM 產生幻覺」痛點所進行的技術評估與迭代歷程。

---

## 1. 研究動機與核心挑戰

在內部規範查找中，使用者提問常包含**精確條號（如 SOP-801、法規條款代碼）**或**特定門檻數值**。
* **主要痛點：** 純語意向量檢索（Dense Retrieval）容易因向量相似度模糊，召回相似動作但條號錯誤的文本，直接引發生成階段的致命幻覺。
* **研究目標：** 探索如何在保留語意理解的同時，將條號與關鍵字命中率最大化，並兼顧推論延遲。

---

## 2. 幾次迭代與方法對比 (Iteration Log)

在 POC 過程中，先後評估了三種方案：

| 迭代版本 | 採用架構 | 優點 | 致命缺點 / 淘汰原因 |
| :--- | :--- | :--- | :--- |
| **Phase 1: Baseline** | 純 Dense 向量檢索 (MiniLM / BGE) | 語意匹配度高，能處理口語提問 | **條號精確命中率僅約 50%**，常因文字語意相近而抓錯條文 |
| **Phase 2: 關鍵字補強** | 純 Sparse 檢索 (BM25) | 條號命中率達 85% 以上，速度極快 (<2ms) | **完全喪失語意理解**，遇到使用者口語化表述或同義詞時徹底失靈 |
| **Phase 3: 雙路混合架構** | **BM25 + Dense + RRF + Cross-Encoder** | **條號精確度提升至 98%**，語意與代號兼顧 | 增加約 30ms 重排運算延遲，但在高容錯代價場景下可接受 |

---

## 3. 現階段技術抉擇 (Decision & Trade-offs)

經多次實驗，現階段採用 **Phase 3（雙路混合檢索 + Cross-Encoder 重排）** 作為核心解方：

1. **捨棄分數線性加權，改採 RRF 融合：**  
   BM25 得分與 Cosine 分數尺度差異大，手動調整加權超參數極不穩定；採用 **RRF（倒數排名融合）** 可基於排名次序穩定整合兩路候選名單。
2. **以延遲換取精準度（Precision over Latency）：**  
   雖然 Cross-Encoder 帶來微小的計算開銷，但內部規範防禦屬於「精確度優先於極致速度」的任務，這筆延遲預算具備高度工程價值。
3. **加入安全截斷門檻（Guardrails）：**  
   若精排最高得分低於閥值，系統強制回覆「查無對應規範」，杜絕模型自由推測。

---

## 4. 未來持續研究方向 (Next Steps)

* **[方向一] 複雜跨頁表格切塊：** 目前純文字切塊對橫跨多頁的合併儲存格解析仍有斷裂問題，預計後續評估整合專用表格結構辨識模組。
* **[方向二] 地端高併發推論部署：** 評估引入 `vLLM` 框架加速地端自建開源模型（如 Qwen-2.5 / Llama-3）之 Serving 吞吐，確保敏感資料不出網。
* **[方向三] 自動化評估管線（Ragas）：** 建立 Golden Dataset，持續監控忠實度（Faithfulness）與答案相關度。

---

## 快速重現實驗

```bash
pip install -r requirements.txt
python run_poc.py
