"""
Task 10 — Generation Có Citation.

Quy trình:
    1. Retrieve chunks từ Task 9
    2. Reorder chunks để tránh hiện tượng "lost in the middle"
    3. Format context kèm nhãn tài liệu (Source label)
    4. Inject context vào System & User Prompt
    5. Gọi LLM (OpenRouter / OpenAI API) sinh câu trả lời có trích dẫn (Citation)
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# CONFIGURATION
# =============================================================================

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

SYSTEM_PROMPT = """Bạn là Trợ lý tư vấn tuyển sinh đại học thông minh (EduSeek Assistant).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ Context được cung cấp — KHÔNG tự bịa đặt hay suy đoán ngoài tài liệu.
2. Mỗi khẳng định quan trọng phải có trích dẫn nguồn ngay phía sau, ví dụ: [Tên tài liệu, trang X].
3. Nếu Context không đủ thông tin để trả lời → Hãy nêu rõ: "Tôi không thể xác minh thông tin này từ dữ liệu hiện có."
4. Trả lời bằng tiếng Việt, trình bày mạch lạc, dùng các thẻ Markdown (danh sách, bảng, in đậm) để thông tin rõ ràng nhất."""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.
    Strategy: Đặt chunks điểm cao nhất ở ĐẦU và CUỐI prompt, chunks kém hơn ở GIỮA.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return chunks

    front = chunks[::2]   # index 0, 2, 4 -> đặt ở đầu
    back = chunks[1::2]   # index 1, 3    -> đặt ở cuối (reversed)
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """Format chunks thành context string kèm nguồn trích dẫn cho LLM prompt."""
    if not chunks:
        return "Không tìm thấy đoạn văn bản phù hợp."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Tài liệu {i}")
        doc_type = chunk.get("metadata", {}).get("type", "standardized")
        page = chunk.get("metadata", {}).get("page", "?")
        content = chunk.get("content", "").strip()

        context_parts.append(
            f"[Tài liệu {i} | Nguồn: {source} | Loại: {doc_type} | Trang: {page}]\n"
            f"{content}"
        )
    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# LOCAL FALLBACK SEARCH (Dùng khi Task 9 / Vector DB chưa được build)
# =============================================================================

def _fallback_file_search(query: str, top_k: int = 5) -> list[dict]:
    """Tìm kiếm thông minh trên các file .md trong data/standardized/ khi DB chưa sẵn sàng"""
    std_dir = PROJECT_ROOT / "data" / "standardized"
    if not std_dir.exists():
        return []

    q_lower = query.lower()
    keywords = [k for k in re.findall(r'\w+', q_lower) if len(k) > 1]
    priority_words = ["ielts", "bách khoa", "hust", "rmit", "khtn", "điểm chuẩn", "xét tuyển", "học phí", "tsa", "ngành"]

    matched_chunks = []

    for md_file in std_dir.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            sections = content.split("\n\n")
            for sec in sections:
                sec_text = sec.strip()
                if len(sec_text) < 30:
                    continue
                sec_lower = sec_text.lower()

                # Score matching
                score = sum(2.5 if kw in priority_words else 1.0 for kw in keywords if kw in sec_lower)

                if "ielts" in q_lower and "ielts" in sec_lower:
                    score += 6.0
                if "điểm chuẩn" in q_lower and "điểm chuẩn" in sec_lower:
                    score += 6.0
                if "bách khoa" in q_lower and ("bách khoa" in sec_lower or "dhbk" in sec_lower):
                    score += 3.0

                if score > 2.0:
                    matched_chunks.append({
                        "content": sec_text[:500],
                        "metadata": {"source": md_file.name, "type": md_file.parent.name},
                        "score": score,
                        "source": "fallback_file_search"
                    })
        except Exception:
            continue

    matched_chunks.sort(key=lambda x: x["score"], reverse=True)
    return matched_chunks[:top_k]


# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.
    """
    # Step 1: Retrieve chunks từ Task 9
    try:
        from .task9_retrieval_pipeline import retrieve
        chunks = retrieve(query, top_k=top_k)
    except Exception:
        chunks = []

    # Fallback nếu Task 9 chưa trả về dữ liệu (do DB chưa build)
    if not chunks:
        chunks = _fallback_file_search(query, top_k=top_k)

    retrieval_src = chunks[0].get("source", "hybrid") if chunks else "fallback"

    # Step 2: Reorder chunks chống Lost in the middle
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4 & 5: Call LLM API (OpenRouter hoặc OpenAI)
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    if api_key and chunks:
        try:
            from openai import OpenAI
            base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
            client = OpenAI(api_key=api_key, base_url=base_url)

            user_message = f"Dưới đây là Context thông tin trích xuất từ dữ liệu tuyển sinh:\n\n{context}\n\n---\n\nCâu hỏi: {query}"

            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"⚠️ *Không thể gọi LLM API ({e})*. Hiển thị dữ liệu trích xuất trực tiếp:\n\n" + \
                     "\n\n".join([f"• **[{c['metadata'].get('source')}]**: {c['content']}" for c in chunks[:3]])
    else:
        # Khi chưa điền API Key
        if chunks:
            answer = (
                f"### 📄 Kết quả truy xuất từ dữ liệu tuyển sinh chính thức (`{retrieval_src}`)\n\n"
                f"*(Lưu ý: Hãy thêm `OPENROUTER_API_KEY` vào file `.env` để LLM tự động tổng hợp câu trả lời)*\n\n"
            )
            for i, c in enumerate(chunks[:3], 1):
                src_name = c['metadata'].get('source', 'Unknown')
                answer += f"**[{i}] Nguồn `{src_name}`**:\n> {c['content'][:350]}...\n\n"
        else:
            answer = "Tôi không tìm thấy thông tin phù hợp cho câu hỏi này trong cơ sở dữ liệu tuyển sinh."

    # Step 6: Return result dictionary
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_src,
    }


if __name__ == "__main__":
    test_queries = [
        "Điều kiện xét tuyển thẳng bằng IELTS vào Bách Khoa Hà Nội?",
        "Điểm chuẩn trúng tuyển ĐH Bách Khoa Hà Nội các năm gần đây?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
