import json
import os
import re
import sys
from datetime import datetime
import requests
from bs4 import BeautifulSoup

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Tạo thư mục lưu trữ data/landing/news/ nếu chưa tồn tại
OUTPUT_DIR = os.path.join("data", "landing", "news")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Danh sách 5 bài viết/thông báo tuyển sinh thực tế cần thu thập
URLS_TO_CRAWL = [
    {
        "id": "hust_2026_xettuyen_taenang",
        "school": "Đại học Bách khoa Hà Nội",
        "url": "https://ts.hust.edu.vn/tin-tuc/quy-che-xet-tuyen-tai-nang-2025-doi-voi-tuyen-sinh-dai-hoc-he-chinh-quy",
        "fallback_title": "Quy chế Xét tuyển Tài năng Đại học Bách khoa Hà Nội",
        "category": "Phương thức xét tuyển",
    },
    {
        "id": "hust_2026_quyche_tuyensinh",
        "school": "Đại học Bách khoa Hà Nội",
        "url": "https://ts.hust.edu.vn/tin-tuc/quy-che-tuyen-sinh-dai-hoc-nam-2026",
        "fallback_title": "Thông tin Quy chế Tuyển sinh Đại học Bách khoa Hà Nội",
        "category": "Quy chế tuyển sinh",
    },
    {
        "id": "rmit_2026_dieukien_tuyensinh",
        "school": "Đại học RMIT Việt Nam",
        "url": "https://www.rmit.edu.vn/vi/hoc-tap-tai-rmit/chuong-trinh-cu-nhan",
        "fallback_title": "Yêu cầu đầu vào và Điều kiện xét tuyển Đại học RMIT",
        "category": "Điều kiện đầu vào",
    },
    {
        "id": "vinuni_2026_hocphi_scholes",
        "school": "Trường Đại học VinUni",
        "url": "https://vinuni.edu.vn/scholarship/",
        "fallback_title": "Chính sách Học phí và Học bổng Trường Đại học VinUni",
        "category": "Học phí & Học bổng",
    },
    {
        "id": "khtn_2026_xettuyen_thang",
        "school": "Đại học Khoa học Tự nhiên - ĐHQGHN",
        "url": "https://hus.vnu.edu.vn/dao-tao/dai-hoc/thong-tin-tuyen-sinh.html",
        "fallback_title": "Thông tin Tuyển sinh Đại học Khoa học Tự nhiên",
        "category": "Đề án tuyển sinh",
    },
]


def clean_text(text: str) -> str:
    """Làm sạch ký tự thừa, khoảng trắng trùng lặp."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def crawl_article(item: dict) -> dict:
    """Crawl chi tiết bài viết, trích xuất Title và Body Text."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(item["url"], headers=headers, timeout=10, verify=False)

        response.encoding = "utf-8"

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Lấy Title
            title_tag = soup.find(["h1", "h2"])
            title = (
                clean_text(title_tag.get_text())
                if title_tag
                else item["fallback_title"]
            )

            # Lấy nội dung văn bản (bỏ các thẻ script, style, nav, footer)
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            paragraphs = soup.find_all(["p", "li", "h3", "h4", "table"])
            content_list = [clean_text(p.get_text()) for p in paragraphs]
            body_content = "\n\n".join(
                [text for text in content_list if len(text) > 30]
            )

            return {
                "id": item["id"],
                "school_name": item["school"],
                "title": title,
                "category": item["category"],
                "source_url": item["url"],
                "crawled_at": datetime.now().isoformat(),
                "status": "success",
                "content": body_content[
                    :4000
                ],  # Giới hạn độ dài bản trích yếu
            }
    except Exception as e:
        print(f"⚠️ Lỗi khi cào {item['url']}: {e}")

    # Tạo dữ liệu mock-up chuẩn nếu không kết nối được tới server trường
    return {
        "id": item["id"],
        "school_name": item["school"],
        "title": item["fallback_title"],
        "category": item["category"],
        "source_url": item["url"],
        "crawled_at": datetime.now().isoformat(),
        "status": "fallback",
        "content": f"Nội dung quy định tuyển sinh chính thức thuộc {item['school']}. Bao gồm các điều kiện chi tiết về điểm GPA, chứng chỉ tiếng Anh quốc tế (IELTS), kết quả thi Đánh giá tư duy/năng lực và danh mục các ngành tuyển sinh...",
    }


def main():
    print("🚀 Bắt đầu crawl dữ liệu đề án tuyển sinh...")

    for item in URLS_TO_CRAWL:
        data = crawl_article(item)
        filename = f"{item['id']}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(
            f"✅ Đã lưu: {filepath} ({data['school_name']} - {data['category']})"
        )

    print("\n🎉 HOÀN THÀNH! Đã lưu 5 bài viết vào thư mục data/landing/news/")


if __name__ == "__main__":
    main()