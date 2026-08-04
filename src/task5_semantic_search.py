"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.
"""

from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer

# Import trực tiếp cấu hình từ Task 4 để đảm bảo đồng bộ model và database
from src.task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

# Cache model và collection để không phải load lại nhiều lần gây chậm
_embedding_model = None
_chroma_collection = None

def get_embedding_model():
    """Khởi tạo và trả về model SentenceTransformer."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model

def get_collection():
    """Khởi tạo và trả về ChromaDB collection."""
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_collection = client.get_collection(name=COLLECTION_NAME)
    return _chroma_collection

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    model = get_embedding_model()
    collection = get_collection()

    # Bước 1: Embed query bằng cùng model ở Task 4
    query_vector = model.encode(query).tolist()

    # Bước 2: Query vector store
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    # Kiểm tra an toàn nếu không có tài liệu nào trong DB
    if not results["documents"] or not results["documents"][0]:
        return output

    # Bước 3: Xử lý và format kết quả
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # ChromaDB trả về Cosine Distance. Convert sang Cosine Similarity.
        score = max(0.0, 1.0 - dist)  
        output.append({
            "content": doc, 
            "score": round(score, 4), 
            "metadata": meta
        })

    # Đảm bảo danh sách được sắp xếp giảm dần theo điểm
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]

if __name__ == "__main__":
    # Test
    results = semantic_search("what is the tuition fee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")