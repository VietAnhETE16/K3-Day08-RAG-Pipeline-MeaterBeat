"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():

    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print("Folder data/landing/legal/ does not exist.")
        return

    legal_files = [f for f in legal_dir.iterdir() if f.suffix.lower() in (".pdf", ".docx", ".doc")]
    if not legal_files:
        print("No legal PDF/DOCX files found in data/landing/legal/.")
        return

    try:
        from markitdown import MarkItDown
        md = MarkItDown()
    except ImportError:
        print("⚠️ MarkItDown chưa sẵn sàng, đang cài đặt...")
        return

    for filepath in legal_files:
        print(f"Converting: {filepath.name}")
        try:
            result = md.convert(str(filepath))
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(result.text_content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")
        except Exception as e:
            print(f"  ❌ Error converting {filepath.name}: {e}")



def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print("Folder data/landing/news/ does not exist.")
        return

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                output_path = output_dir / f"{filepath.stem}.md"

                # Thêm metadata header
                title = data.get("title", "Unknown")
                source_url = data.get("source_url", data.get("url", "N/A"))
                school = data.get("school_name", "N/A")
                category = data.get("category", "N/A")
                crawled_at = data.get("crawled_at", data.get("date_crawled", "N/A"))
                body_content = data.get("content", data.get("content_markdown", ""))

                header = f"# {title}\n\n"
                header += f"**Truong:** {school}\n"
                header += f"**DanhMuc:** {category}\n"
                header += f"**Source:** {source_url}\n"
                header += f"**Crawled:** {crawled_at}\n\n---\n\n"

                content = header + body_content
                output_path.write_text(content, encoding="utf-8")
                print(f"  ✓ Saved: {output_path}")
            except Exception as e:
                print(f"  ❌ Error converting {filepath.name}: {e}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()

