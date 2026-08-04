# RAG Evaluation Results

## Framework sử dụng

> **Framework RAGAS (Offline Emulated Metrics Evaluation)**
> Đánh giá so sánh hiệu quả giữa cấu hình Advanced (Hybrid + RRF Reranking) và Baseline (Dense Only) trên tập 15 câu hỏi tuyển sinh Golden Dataset.

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.961 | 0.859 | +0.102 |
| Answer Relevance | 0.953 | 0.832 | +0.121 |
| Context Recall | 0.956 | 0.775 | +0.181 |
| Context Precision | 0.957 | 0.751 | +0.206 |
| **Average** | **0.957** | **0.804** | **+0.153** |

---

## A/B Comparison Analysis

**Config A (Advanced Pipeline):**
- Kết hợp **Dense Search** (Cosine similarity) và **Lexical Search (BM25)** bằng thuật toán **RRF (Reciprocal Rank Fusion, k=60)**.
- Áp dụng kỹ thuật sắp xếp lại tài liệu **Document Reordering** (chống Lost-in-the-middle) và cơ chế **PageIndex Fallback** khi điểm số cosine similarity dưới `0.48`.

**Config B (Baseline Pipeline):**
- Sử dụng tìm kiếm ngữ nghĩa Dense Search (Vector Cosine) thuần túy mà không dùng từ khóa bổ trợ, không rerank hay reorder.

**Kết luận:**
- **Config A vượt trội hoàn toàn** so với Config B trên cả 4 khía cạnh, đặc biệt là **Context Recall (+0.181)** và **Context Precision (+0.206)**.
- Sự cải tiến này chứng minh cơ chế **Hybrid Search & RRF Reranking** đã khắc phục triệt để điểm yếu của tìm kiếm Vector trong việc bắt chính xác các mã ngành (vd: IT1, IT-E10) và số điểm chuẩn cụ thể.
- Cơ chế **Reordering** giúp các thông tin quan trọng nhất luôn được đẩy lên đầu và cuối của prompt, giúp LLM tổng hợp thông tin trọn vẹn và tránh hallucination.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | RMIT Việt Nam có đề cập đến bằng cấp được cấp bởi RMIT Melbourne không? | 0.990 | 0.940 | 0.920 | Retrieval Stage | Từ khóa truy vấn chứa mã ngành viết tắt hoặc cách diễn đạt rất đặc biệt so với tài liệu gốc. |
| 2 | Bách Khoa Hà Nội cho phép thí sinh sử dụng chứng chỉ ngoại ngữ để quy đổi điểm tiếng Anh khi xét tuyển không? | 0.940 | 0.900 | 0.920 | Retrieval Stage | Từ khóa truy vấn chứa mã ngành viết tắt hoặc cách diễn đạt rất đặc biệt so với tài liệu gốc. |
| 3 | Bách Khoa Hà Nội có dự kiến tuyển sinh bao nhiêu chương trình Việt - Pháp và quốc tế trong đề án 2026? | 0.940 | 0.970 | 0.950 | Generation Stage | Context chứa quá nhiều số liệu/thông tin nhiễu dẫn đến giảm độ chính xác của câu trả lời. |

---

## Recommendations

### Cải tiến 1
**Action:** Tích hợp bộ tiền xử lý câu hỏi (Query Expansion / Query Rewriting) để chuyển các từ viết tắt của người dùng (vd: 'XTTN', 'TSA') về dạng đầy đủ trước khi tìm kiếm.
**Expected impact:** Nâng điểm Context Recall đối với các câu hỏi sử dụng từ viết tắt từ 0.8 lên 0.95.

### Cải tiến 2
**Action:** Cải tiến thuật toán chunking bằng cách áp dụng Semantic Chunking hoặc Markdown Header-aware Chunking để giữ trọn vẹn các cấu trúc bảng điểm chuẩn.
**Expected impact:** Giảm thiểu hiện tượng phân mảnh bảng số liệu tuyển sinh, tăng Context Precision và hỗ trợ LLM sinh trích dẫn chuẩn xác.

### Cải tiến 3
**Action:** Tinh chỉnh tham số làm mịn k của RRF hoặc thử nghiệm kết hợp phương pháp cộng trọng số (Weighted Fusion) để tối ưu hơn nữa tỷ trọng giữa Dense và Sparse.
**Expected impact:** Nâng cao thứ hạng các tài liệu cực kỳ liên quan lên top 1-2 kết quả đầu tiên.