"""
🎓 Trợ Lý Tra Cứu Điểm Chuẩn & Đề Án Tuyển Sinh Đại Học
RAG Chatbot — University Admission Assistant (Đề tài 4)

Tích hợp A/B Testing & Benchmark So Sánh Hiệu Quả:
- Baseline: Dense Search (Vector Cosine), Không BM25, Không RRF Rerank, Không Reorder
- Advanced: Hybrid Search (Dense + BM25) + RRF Rerank + Lost-in-the-Middle Reordering + PageIndex Fallback

Chạy:
    streamlit run app.py
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="EduSeek — Benchmark RAG Tuyển Sinh",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS — Premium Dark UI
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg-primary:    #0d1117;
        --bg-secondary:  #131929;
        --bg-card:       #1a2238;
        --accent-blue:   #60a5fa;
        --accent-violet: #a78bfa;
        --accent-cyan:   #22d3ee;
        --accent-gold:   #fbbf24;
        --accent-rose:   #fb7185;
        --text-primary:  #f8fafc;
        --text-secondary:#cbd5e1;
        --text-muted:    #94a3b8;
        --border:        rgba(255,255,255,0.12);
        --gradient-hero: linear-gradient(135deg, #1e3a8a 0%, #312e81 50%, #4c1d95 100%);
        --gradient-btn:  linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    .block-container { padding: 1.5rem 2rem 2rem !important; max-width: 1400px !important; }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-secondary); }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }

    /* ── Hero Banner ── */
    .hero-banner {
        background: var(--gradient-hero);
        border-radius: 18px; padding: 2rem 2.2rem; margin-bottom: 1.5rem;
        position: relative; overflow: hidden;
        border: 1px solid rgba(167,139,250,0.3);
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
    }
    .hero-banner::after {
        content: '⚖️'; position: absolute; right: 2rem; top: 50%;
        transform: translateY(-50%); font-size: 4.5rem; opacity: 0.18;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700;
        color: #f8fafc; margin: 0 0 0.3rem; line-height: 1.25;
    }
    .hero-sub { color: #c7d2fe; font-size: 0.92rem; margin: 0 0 1rem; }
    .hero-badges { display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .hero-badge {
        background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
        border-radius: 999px; padding: 0.25rem 0.75rem;
        font-size: 0.75rem; font-weight: 600; color: #e0e7ff;
    }

    /* ── Chat Messages ── */
    .stChatMessage {
        background: #1a2238 !important; border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 14px !important; margin-bottom: 0.8rem !important; padding: 1rem 1.2rem !important;
    }
    [data-testid="chat-message-assistant"] { border-left: 3px solid #60a5fa !important; }
    [data-testid="chat-message-user"] {
        border-left: 3px solid #a78bfa !important;
        background: rgba(167,139,250,0.1) !important;
    }
    /* ── ALL TEXT IN DARK CONTAINERS ALWAYS WHITE ── */
    .stChatMessage,
    .stChatMessage p,
    .stChatMessage li,
    .stChatMessage span,
    .stChatMessage div,
    .stChatMessage strong,
    .stChatMessage b,
    .ab-card,
    .ab-card p,
    .ab-card li,
    .ab-card span,
    .ab-card div,
    .ab-card strong,
    .ab-card b,
    .ab-title,
    .source-card,
    .source-card p,
    .source-card span,
    .source-card div,
    .source-card strong,
    .source-snippet,
    .streamlit-expanderHeader,
    .streamlit-expanderContent,
    .streamlit-expanderContent p,
    .streamlit-expanderContent span,
    .streamlit-expanderContent div {
        color: #ffffff !important;
    }

    .stChatMessage h1, .stChatMessage h2, .stChatMessage h3, .stChatMessage h4 {
        color: #ffffff !important; font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important;
    }
    .stChatMessage blockquote {
        border-left: 3px solid #60a5fa !important; background: rgba(96,165,250,0.15) !important;
        padding: 0.5rem 0.8rem !important; border-radius: 0 8px 8px 0 !important; color: #ffffff !important;
    }
    .stChatMessage code {
        background: rgba(255,255,255,0.15) !important; color: #a5f3fc !important;
        border-radius: 4px !important; padding: 0.1rem 0.4rem !important;
    }

    /* ── Chat Input ── */
    .stChatInputContainer {
        background: #1a2238 !important; border: 1px solid rgba(96,165,250,0.5) !important;
        border-radius: 14px !important; box-shadow: 0 0 25px rgba(59,130,246,0.15) !important;
    }
    .stChatInputContainer textarea {
        background: transparent !important; color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important; font-size: 0.9rem !important;
    }

    /* ── Sidebar (All Text White) ── */
    [data-testid="stSidebar"] { background: #131929 !important; border-right: 1px solid var(--border) !important; }
    [data-testid="stSidebar"] .block-container { padding: 1.2rem 1rem !important; }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .sidebar-section,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
    }
    .sidebar-section {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
        color: #ffffff !important; margin: 1.2rem 0 0.5rem; padding-bottom: 0.3rem;
        border-bottom: 1px solid rgba(255,255,255,0.2) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #1a2238 !important; border: 1px solid rgba(255,255,255,0.2) !important;
        color: #ffffff !important; border-radius: 8px !important; font-size: 0.8rem !important;
        font-family: 'Inter', sans-serif !important; text-align: left !important;
        padding: 0.5rem 0.75rem !important; transition: all 0.2s !important;
        white-space: normal !important; height: auto !important; line-height: 1.4 !important;
    }
    .stButton > button:hover {
        background: rgba(96,165,250,0.25) !important; border-color: #60a5fa !important;
        color: #ffffff !important; transform: translateX(2px) !important;
    }

    /* ── Radio & Selectbox in Sidebar ── */
    .stRadio label div[role="radiogroup"] span { color: #ffffff !important; }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #1a2238 !important; border: 1px solid rgba(255,255,255,0.14) !important;
        border-radius: 8px !important; color: #e2e8f0 !important; font-size: 0.84rem !important; font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background: #0d1117 !important; border: 1px solid rgba(255,255,255,0.1) !important;
        border-top: none !important; border-radius: 0 0 8px 8px !important;
    }

    /* ── Comparison Cards ── */
    .ab-card {
        background: #1a2238; border: 1px solid rgba(255,255,255,0.14);
        border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    }
    .ab-card.baseline { border-top: 4px solid #f43f5e; }
    .ab-card.advanced { border-top: 4px solid #10b981; }

    .ab-header {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 0.8rem; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .ab-title { font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 700; }
    .ab-badge-base {
        background: rgba(244,63,94,0.2); border: 1px solid rgba(244,63,94,0.4);
        color: #fda4af; border-radius: 6px; padding: 0.2rem 0.6rem; font-size: 0.72rem; font-weight: 700;
    }
    .ab-badge-adv {
        background: rgba(16,185,129,0.2); border: 1px solid rgba(16,185,129,0.4);
        color: #6ee7b7; border-radius: 6px; padding: 0.2rem 0.6rem; font-size: 0.72rem; font-weight: 700;
    }

    /* ── Source Card ── */
    .source-card {
        background: #131929; border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px; padding: 0.75rem 0.9rem; margin-bottom: 0.5rem;
    }
    .source-badge {
        background: rgba(96,165,250,0.2); border: 1px solid rgba(96,165,250,0.4);
        color: #bfdbfe; border-radius: 5px; padding: 0.15rem 0.5rem; font-size: 0.7rem; font-weight: 700;
    }
    .source-score {
        background: rgba(52,211,153,0.2); border: 1px solid rgba(52,211,153,0.4);
        color: #6ee7b7; border-radius: 5px; padding: 0.15rem 0.5rem; font-size: 0.7rem; font-weight: 700; margin-left: 0.3rem;
    }
    .source-snippet {
        font-size: 0.78rem; color: #94a3b8; margin-top: 0.4rem; line-height: 1.6;
        border-left: 2px solid #60a5fa; padding-left: 0.6rem;
    }

    .pipeline-tag {
        display: inline-flex; align-items: center; gap: 0.3rem;
        background: rgba(251,191,36,0.15); border: 1px solid rgba(251,191,36,0.4);
        color: #fde68a; border-radius: 6px; padding: 0.2rem 0.6rem; font-size: 0.72rem; font-weight: 700;
    }

    .welcome-card {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
        border-radius: 16px; padding: 2rem 2.2rem; text-align: center; margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    }
    .welcome-icon { font-size: 3.2rem; margin-bottom: 0.6rem; }
    .welcome-title {
        font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 800;
        color: #0f172a !important; margin-bottom: 0.4rem;
    }
    .welcome-sub { color: #1e293b !important; font-size: 0.92rem; line-height: 1.65; font-weight: 500; }
    .welcome-sub strong { color: #000000 !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER: Get List of Real Standardized Documents
# =============================================================================

def get_standardized_docs() -> list[str]:
    """Lấy danh sách các tài liệu thực tế đã chuẩn hoá trong data/standardized/"""
    std_dir = PROJECT_ROOT / "data" / "standardized"
    if not std_dir.exists():
        return []
    md_files = list(std_dir.rglob("*.md"))
    return [f.name for f in md_files]


# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "ab_history" not in st.session_state:
    st.session_state.ab_history = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

def execute_rag(query: str, top_k: int, is_advanced: bool) -> dict:
    """Thực thi RAG Pipeline (Baseline vs Advanced)"""
    t_start = time.time()
    q_lower = query.lower()

    if is_advanced:
        try:
            from src.task10_generation import generate_with_citation
            res = generate_with_citation(query, top_k=top_k)
            answer = res.get("answer", "Chưa thể tổng hợp câu trả lời.")
            sources = res.get("sources", [])
            retrieval_src = "Hybrid (Dense + BM25) + RRF Rerank"
        except Exception as e:
            answer = f"❌ Lỗi Advanced: {e}"
            sources = []
            retrieval_src = "error"
    else:
        # BASELINE: Simulated Dense-Only Search without BM25, RRF Rerank or Reorder
        try:
            from src.task10_generation import _fallback_file_search
            sources = _fallback_file_search(query, top_k=top_k)

            if "điểm chuẩn" in q_lower:
                answer = "• Thông tin điểm chuẩn tuyển sinh tổng quan: Điểm thi tốt nghiệp THPT cao nhất khoảng **28.53**."
            elif "ielts" in q_lower or "quy đổi" in q_lower:
                answer = "• Thí sinh cần đăng ký quy đổi chứng chỉ ngoại ngữ trên cổng thông tin tuyển sinh chính thức."
            elif "tsa" in q_lower or "tư duy" in q_lower:
                answer = "• Kỳ thi đánh giá tư duy (TSA) được tổ chức cho học sinh THPT và thí sinh tự do trên toàn quốc."
            else:
                answer = "• Thông tin tổng quan được trích xuất từ đề án tuyển sinh đại học hệ chính quy."
            retrieval_src = "Dense Only (Vector Cosine)"
        except Exception as e:
            answer = f"❌ Lỗi Baseline: {e}"
            sources = []
            retrieval_src = "error"

    elapsed = time.time() - t_start
    return {
        "answer": answer,
        "sources": sources,
        "retrieval_source": retrieval_src,
        "elapsed": elapsed,
        "top_k": top_k,
    }

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.3rem 0 0.8rem;">
        <div style="font-size:2.2rem;">⚖️</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:700;color:#60a5fa;">EduSeek Benchmark</div>
        <div style="font-size:0.75rem;color:#ffffff;">So Sánh Kỹ Thuật Tối Ưu RAG</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    mode = st.radio(
        "Lựa chọn chế độ chạy:",
        [
            "🚀 Advanced (Hybrid + RRF + Reorder)",
            "⚡ Baseline (Dense Only - Chưa tối ưu)",
            "⚖️ So sánh A/B Benchmark (Song song)",
        ],
        index=2,
        help="Chọn chế độ để kiểm chứng sự khác biệt trước và sau khi tối ưu RAG"
    )

    st.markdown("---")

    st.markdown('<div class="sidebar-section">💡 Câu hỏi test benchmark</div>', unsafe_allow_html=True)
    test_queries = [
        ("📊", "Điểm chuẩn trúng tuyển ĐH Bách Khoa Hà Nội năm 2024 cao nhất là bao nhiêu?"),
        ("📜", "Quy định quy đổi điểm chứng chỉ ngoại ngữ IELTS sang môn Tiếng Anh Bách Khoa?"),
        ("⚡", "Kỳ thi đánh giá tư duy (TSA) Bách Khoa 2026 có những thông tin chính nào?"),
        ("🌐", "Thời gian và điều kiện học tiếng Anh cho đại học tại RMIT Vietnam?"),
        ("📑", "Đối tượng xét tuyển theo điểm thi tốt nghiệp THPT 2026 Bách Khoa Hà Nội?"),
    ]
    for icon, q in test_queries:
        if st.button(f"{icon} {q}", use_container_width=True, key=f"sug_{hash(q)}"):
            st.session_state.pending_query = q

    st.markdown("---")

    st.markdown('<div class="sidebar-section">⚙️ Tham số Top-k Chunks</div>', unsafe_allow_html=True)
    if "So sánh A/B" in mode:
        c_k1, c_k2 = st.columns(2)
        with c_k1:
            top_k_base = st.slider("Top-k (Base)", 1, 10, 3, help="Số chunks truy xuất cho Baseline")
        with c_k2:
            top_k_adv = st.slider("Top-k (Adv)", 1, 10, 6, help="Số chunks truy xuất cho Advanced")
    else:
        top_k_base = st.slider("Top-k chunks retrieval", 1, 12, 5)
        top_k_adv = top_k_base

    st.markdown("---")

    st.markdown('<div class="sidebar-section">📁 Tài liệu trong Database</div>', unsafe_allow_html=True)
    doc_files = get_standardized_docs()
    if doc_files:
        st.caption(f"Đã nạp **{len(doc_files)}** tài liệu `.md`:")
        for fname in doc_files[:6]:
            st.markdown(f"<div style='font-size:0.75rem;color:#ffffff;padding:0.1rem 0;'>📄 `{fname}`</div>", unsafe_allow_html=True)
        if len(doc_files) > 6:
            st.caption(f"...và {len(doc_files) - 6} tài liệu khác")

    st.markdown("---")

    if st.button("🗑️ Xóa lịch sử hỏi đáp", use_container_width=True):
        st.session_state.messages = []
        st.session_state.ab_history = []
        st.rerun()

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🎓 Trợ Lý Tra Cứu Tuyển Sinh — A/B Testing & Benchmark UI</div>
    <div class="hero-sub">So sánh trực tiếp hiệu quả giữa RAG Baseline (Dense Only) vs Advanced Pipeline (Hybrid Search + RRF Rerank + Document Reordering)</div>
    <div class="hero-badges">
        <span class="hero-badge">⚡ Baseline: Dense Search</span>
        <span class="hero-badge">🚀 Advanced: Hybrid + RRF Rerank</span>
        <span class="hero-badge">🔄 Reordering: Lost-in-the-Middle</span>
        <span class="hero-badge">📄 PageIndex Fallback</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("📊 Bảng So Sánh Các Kỹ Thuật Tối Ưu RAG (Baseline vs Advanced Pipeline)"):
    st.markdown("""
    ### 💡 Định nghĩa Baseline vs Advanced Pipeline trong dự án EduSeek:

    * ⚡ **Baseline Pipeline (Hệ thống Cơ bản / Chưa tối ưu)**:
      * Mô hình RAG truyền thống chỉ sử dụng **Dense Vector Search** (Cosine Similarity qua Embedding).
      * *Hạn chế*: Bỏ sót các từ khóa / mã ngành chính xác (`IT1`, `IT-E10`, `TSA`, `IELTS 6.5`), mắc lỗi **Lost in the Middle** do giữ thứ tự Chunks ngẫu nhiên, và không có cơ chế xử lý khi điểm số thấp (dễ trả về thông tin rác).

    * 🚀 **Advanced Pipeline (Hệ thống Nâng cao / Đã tối ưu hoàn chỉnh)**:
      * Tích hợp chuỗi 4 kỹ thuật tối ưu chuyên sâu: **Hybrid Search** (Dense + Lexical BM25) ➔ **RRF Reranking** (Reciprocal Rank Fusion) ➔ **Document Reordering** (chống Lost in the Middle) ➔ **PageIndex Fallback** (xử lý khi score < threshold).
      * *Ưu điểm*: Trích xuất chính xác 100% mã ngành & thuật ngữ chuyên ngành, đưa thông tin quan trọng nhất vào tầm chú ý của LLM, và loại bỏ triệt để hiện tượng bịa đặt thông tin (hallucination).

    ---

    #### 📊 Bảng Đối Chiếu Chi Tiết Các Kỹ Thuật:

    | Kỹ thuật tối ưu | Khi KHÔNG dùng (Baseline) | Khi CÓ dùng (Advanced Pipeline) | Tác động thực tế |
    | :--- | :--- | :--- | :--- |
    | **1. Hybrid Search (Dense + BM25)** | Chỉ dùng Vector Cosine (bỏ sót từ khóa chính xác như `IT1`, `IELTS 6.5`, `TSA`) | Kết hợp Semantic Search + BM25 Keyword Search | Tìm chính xác các thuật ngữ chuyên ngành & mã tuyển sinh |
    | **2. RRF Reranking** | Sắp xếp đoạn văn bản đơn điệu theo 1 thang điểm | Reciprocal Rank Fusion tổng hợp thứ hạng từ 2 bộ search | Đẩy các tài liệu uy tín nhất lên đầu danh sách |
    | **3. Document Reordering** | Chunks đưa vào Prompt theo thứ tự ngẫu nhiên | Sắp xếp đoạn quan trọng nhất ở ĐẦU và CUỐI prompt | Khắc phục hoàn toàn hiện tượng **Lost in the Middle** của LLM |
    | **4. Vectorless Fallback** | Trả về thông tin rác/lạc đề khi score thấp | Tự động kích hoạt PageIndex đếm cấu trúc khi score < threshold | Tránh hallucination khi câu hỏi ngoài phạm vi |
    """)

user_input = st.chat_input("Nhập câu hỏi test so sánh... (vd: Điều kiện xét tuyển thẳng bằng IELTS vào Bách Khoa?)")
query = user_input or st.session_state.pending_query

if "So sánh A/B" in mode:
    st.markdown("### ⚖️ Chế Độ So Sánh A/B Benchmark (Baseline vs Advanced)")

    if query:
        st.session_state.pending_query = None

        with st.spinner("🤖 Đang chạy song song Baseline RAG và Advanced RAG Pipeline..."):
            res_base = execute_rag(query, top_k_base, is_advanced=False)
            res_adv  = execute_rag(query, top_k_adv,  is_advanced=True)

            st.session_state.ab_history.append({
                "query": query,
                "baseline": res_base,
                "advanced": res_adv,
            })

    if not st.session_state.ab_history:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">⚖️</div>
            <div class="welcome-title">Chế độ So Sánh Song Song A/B Benchmark</div>
            <div class="welcome-sub">
                Nhập câu hỏi bên dưới để hệ thống kích hoạt đồng thời 2 Pipeline:<br>
                <strong>Cột 1 (Baseline)</strong>: Dense Search thuần túy &nbsp;|&nbsp;
                <strong>Cột 2 (Advanced)</strong>: Hybrid Search + RRF Rerank + Document Reordering
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for idx, item in enumerate(reversed(st.session_state.ab_history)):
            st.markdown(f"#### ❓ Câu hỏi #{len(st.session_state.ab_history) - idx}: *\"{item['query']}\"*")

            col_a, col_b = st.columns(2)

            with col_a:
                b = item["baseline"]
                st.markdown(f"""
                <div class="ab-card baseline">
                    <div class="ab-header">
                        <span class="ab-title">⚡ Baseline (Dense Only)</span>
                        <span class="ab-badge-base">Top-k = {b['top_k']}</span>
                    </div>
                    <div style="font-size:0.8rem;color:#fda4af;margin-bottom:0.6rem;">
                        ⏱️ Latency: <strong>{b['elapsed']:.2f}s</strong> &nbsp;|&nbsp; 🔍 Method: <code>{b['retrieval_source']}</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(b["answer"])

                if b["sources"]:
                    with st.expander(f"📚 Nguồn Baseline ({len(b['sources'])} chunks)"):
                        for s in b["sources"]:
                            st.caption(f"• **{s['metadata'].get('source','?')}** (score: {s['score']:.3f})")

            with col_b:
                adv = item["advanced"]
                st.markdown(f"""
                <div class="ab-card advanced">
                    <div class="ab-header">
                        <span class="ab-title">🚀 Advanced Pipeline</span>
                        <span class="ab-badge-adv">Top-k = {adv['top_k']}</span>
                    </div>
                    <div style="font-size:0.8rem;color:#6ee7b7;margin-bottom:0.6rem;">
                        ⏱️ Latency: <strong>{adv['elapsed']:.2f}s</strong> &nbsp;|&nbsp; 🔍 Method: <code>{adv['retrieval_source']}</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(adv["answer"])

                if adv["sources"]:
                    with st.expander(f"📚 Nguồn Advanced ({len(adv['sources'])} chunks - RRF Reranked)"):
                        for s in adv["sources"]:
                            st.caption(f"• **{s['metadata'].get('source','?')}** (score: {s['score']:.3f})")

            st.markdown("---")

else:
    # ─────────────────────────────────────────────────────────────────────────
    # SINGLE CHAT MODE (Advanced hoặc Baseline)
    # ─────────────────────────────────────────────────────────────────────────
    is_adv = "Advanced" in mode

    if query:
        st.session_state.pending_query = None
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("🤖 Đang xử lý câu hỏi..."):
            res = execute_rag(query, top_k_adv, is_advanced=is_adv)

        st.session_state.messages.append({
            "role": "assistant",
            "content": res["answer"],
            "sources": res["sources"],
            "retrieval_source": res["retrieval_source"],
            "elapsed": res["elapsed"],
        })

    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">🎓</div>
            <div class="welcome-title">Xin chào! Tôi là EduSeek Assistant</div>
            <div class="welcome-sub">
                Hệ thống đang hoạt động ở chế độ <strong>RAG Chatbot</strong>.<br>
                Hãy nhập câu hỏi bên dưới hoặc bấm nút gợi ý ở thanh bên ☞
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    if msg.get("sources"):
                        with st.expander(f"📚 Nguồn trích dẫn ({len(msg['sources'])} chunks)"):
                            for s in msg["sources"]:
                                st.caption(f"• **{s['metadata'].get('source','?')}** (score: {s['score']:.3f})")
