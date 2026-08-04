"""
Task 8 — PageIndex Vectorless RAG.

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding (mục lục, chương, phần) của document.

Chức năng:
    1. upload_documents(): Upload tài liệu lên PageIndex (hoặc indexing cấu trúc)
    2. pageindex_search(): Truy vấn theo cấu trúc tài liệu (Vectorless RAG Fallback),
       trả về list of dict với 'source': 'pageindex'.
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_DOC_IDS_FILE = Path(__file__).parent.parent / "data" / ".pageindex_doc_ids.json"


def upload_documents() -> list[str]:
    """
    Upload toàn bộ markdown documents lên PageIndex API (nếu có API Key),
    hoặc indexing cấu trúc tài liệu.
    """
    doc_ids = []
    if PAGEINDEX_API_KEY:
        try:
            from pageindex.client import PageIndexClient
            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            
            for md_file in STANDARDIZED_DIR.rglob("*.md"):
                resp = client.submit_document(str(md_file))
                doc_id = resp.get("doc_id") or resp.get("id")
                if doc_id:
                    doc_ids.append(doc_id)
                    print(f"  ✓ Uploaded to PageIndex: {md_file.name} -> {doc_id}")
            
            CACHE_DOC_IDS_FILE.write_text(json.dumps(doc_ids), encoding="utf-8")
        except Exception as e:
            print(f"⚠️ Không thể upload lên PageIndex API: {e}")
    else:
        print("ℹ️ PAGEINDEX_API_KEY chưa được thiết lập. Hệ thống sẽ sử dụng Local Structural Document Tree.")

    return doc_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex (hoặc Structural Document Search).
    Dùng làm Fallback khi Hybrid Search có score kém (< 0.48).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval bắt buộc
        }
    """
    results = []

    # 1. Thử gọi PageIndex Client chính thức nếu có API Key
    if PAGEINDEX_API_KEY:
        try:
            from pageindex.client import PageIndexClient
            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            
            doc_ids = []
            if CACHE_DOC_IDS_FILE.exists():
                try:
                    doc_ids = json.loads(CACHE_DOC_IDS_FILE.read_text(encoding="utf-8"))
                except Exception:
                    doc_ids = []

            for doc_id in doc_ids[:3]:
                resp = client.submit_query(doc_id=doc_id, query=query)
                retrieval_id = resp.get("retrieval_id") or resp.get("id")
                if retrieval_id:
                    retrieval = client.get_retrieval(retrieval_id)
                    for node in retrieval.get("retrieved_nodes", [])[:2]:
                        for group in node.get("relevant_contents", []):
                            for item in group:
                                content = item.get("relevant_content", "").strip()
                                if content:
                                    results.append({
                                        "content": content,
                                        "score": 0.85,
                                        "metadata": {"section": item.get("section_title", "General")},
                                        "source": "pageindex"
                                    })
        except Exception as e:
            print(f"⚠️ PageIndex API Search fallback sang Local Structural Tree: {e}")

    # 2. Structural Document Tree Fallback (Khi không có PageIndex SDK/API Key hoặc API lỗi)
    if not results:
        q_terms = [t.lower() for t in re.findall(r'\w+', query) if len(t) > 1]
        
        if STANDARDIZED_DIR.exists():
            for md_file in STANDARDIZED_DIR.rglob("*.md"):
                try:
                    text = md_file.read_text(encoding="utf-8")
                    # Phân tích cấu trúc theo tiêu đề Markdown (#, ##, ###)
                    headers = re.split(r'\n(?=#+\s)', text)
                    for h_block in headers:
                        h_block = h_block.strip()
                        if len(h_block) < 30:
                            continue
                        
                        h_lower = h_block.lower()
                        match_count = sum(1 for term in q_terms if term in h_lower)
                        
                        if match_count > 0:
                            first_line = h_block.split('\n')[0]
                            clean_section = re.sub(r'^#+\s*', '', first_line).strip()
                            
                            results.append({
                                "content": h_block[:600],
                                "score": round(0.5 + (match_count * 0.1), 3),
                                "metadata": {
                                    "source": md_file.name,
                                    "section": clean_section,
                                    "type": md_file.parent.name
                                },
                                "source": "pageindex"
                            })
                except Exception:
                    continue

        results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("ℹ️ Đang chạy thử nghiệm PageIndex Vectorless Search (Local Structural Mode):")
    else:
        print("Uploading documents...")
        upload_documents()

    print("\nTest query:")
    test_results = pageindex_search("điểm chuẩn tuyển sinh bách khoa", top_k=3)
    for r in test_results:
        print(f"[{r['score']:.3f}] [{r['source']}] {r['metadata'].get('source','?')}: {r['content'][:100]}...")
