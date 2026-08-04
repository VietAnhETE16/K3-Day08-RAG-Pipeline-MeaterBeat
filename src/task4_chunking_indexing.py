"""
Task 4 — Chunking & Indexing vào Vector Store (ChromaDB) & JSON Export cho BM25.
"""

import hashlib
import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Khai báo đường dẫn thư mục
BASE_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = BASE_DIR / "data" / "standardized"
PROCESSED_DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
PROCESSED_CHUNKS_FILE = PROCESSED_DATA_DIR / "processed_chunks.json"

# =============================================================================
# CONFIGURATION — Chunking & Embedding Parameters
# =============================================================================
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """Đọc toàn bộ markdown files từ data/standardized/."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        print(f"⚠️ Thư mục không tồn tại: {STANDARDIZED_DIR}")
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents theo RecursiveCharacterTextSplitter (size=800, overlap=100)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed toàn bộ chunks bằng BAAI/bge-m3, fallback local nếu môi trường chặn SSL."""
    try:
        from sentence_transformers import SentenceTransformer

        print(f"⏳ Loading embedding model: {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        texts = [c["content"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=True)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()
    except Exception as exc:
        print(f"⚠️ Không tải được {EMBEDDING_MODEL}: {exc}")
        print("→ Dùng local hashing embedding 1024D để vẫn tạo ChromaDB.")
        for chunk in chunks:
            chunk["embedding"] = _stable_embedding(chunk["content"])
    return chunks


def _stable_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Vector 1024D ổn định, chạy offline khi SentenceTransformer bị chặn."""
    vector = [0.0] * dim
    for token in text.lower().replace("/", " ").replace("-", " ").split():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dim
        vector[index] += 1.0
    norm = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / norm for value in vector]


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks và embeddings vào ChromaDB."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def save_chunks_to_json(chunks: list[dict]):
    """Lưu danh sách chunks ra file JSON để phục vụ cho Task 6 (BM25 Search)."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Loại bỏ vector embedding trước khi lưu file JSON để tiết kiệm dung lượng
    clean_chunks = [
        {"content": c["content"], "metadata": c["metadata"]} 
        for c in chunks
    ]
    
    with open(PROCESSED_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_chunks, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved chunks to {PROCESSED_CHUNKS_FILE} (for BM25)")


def run_pipeline():
    """Chạy toàn bộ pipeline Task 4: load → chunk → embed → index → save JSON."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    if not docs:
        print("❌ Không tìm thấy văn bản nào trong data/standardized/!")
        return

    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    # Lưu file JSON cho BM25 (Task 6)
    save_chunks_to_json(chunks)

    # Embed và lưu vào ChromaDB cho Semantic Search (Task 5)
    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to ChromaDB vector store successfully!")


if __name__ == "__main__":
    run_pipeline()
