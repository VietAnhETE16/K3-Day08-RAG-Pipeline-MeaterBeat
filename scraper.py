import json
import os
import re
import sys
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import urllib3

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Tạo thư mục lưu trữ data/landing/news/ nếu chưa tồn tại
OUTPUT_DIR = os.path.join("data", "landing", "news")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Danh sách các bài viết / thông báo tuyển sinh chính thức của Đại học Bách khoa Hà Nội
URLS_TO_CRAWL = [
    {
        "id": "hust_2026_quyche_tuyensinh",
        "school": "Đại học Bách khoa Hà Nội",
        "url": "https://ts.hust.edu.vn/tin-tuc/quy-che-tuyen-sinh-dai-hoc-nam-2026",
        "fallback_title": "Thông tin Quy chế Tuyển sinh Đại học Bách khoa Hà Nội năm 2026",
        "category": "Quy chế tuyển sinh",
    },
    {
        "id": "hust_2026_xettuyen_taenang",
        "school": "Đại học Bách khoa Hà Nội",
        "url": "https://ts.hust.edu.vn/tin-tuc/quy-che-xet-tuyen-tai-nang-2025-doi-voi-tuyen-sinh-dai-hoc-he-chinh-quy",
        "fallback_title": "Quy định về Phương thức Xét tuyển tài năng (XTTN) Đại học Bách khoa Hà Nội",
        "category": "Phương thức xét tuyển",
    },
    {
        "id": "hust_2026_danhgia_tuduy_tsa",
        "school": "Đại học Bách khoa Hà Nội",
        "url": "https://ts.hust.edu.vn/tin-tuc/de-an-to-chuc-ky-thi-danh-gia-tu-duy-nam-2023",
        "fallback_title": "Đề án tổ chức Kỳ thi Đánh giá tư duy (TSA) Đại học Bách khoa Hà Nội",
        "category": "Kỳ thi Đánh giá tư duy",
    },
    {
        "id": "hust_2026_quydoi_ngoai_ngu",
        "school": "Đại học Bách khoa Hà Nội",
        "url": "https://ts.hust.edu.vn/tin-tuc/huong-dan-quy-doi-diem-chung-chi-ngoai-ngu-nam-2022",
        "fallback_title": "Hướng dẫn Quy đổi điểm chứng chỉ ngoại ngữ (IELTS/TOEFL) Đại học Bách khoa Hà Nội",
        "category": "Chứng chỉ ngoại ngữ",
    },
    {
        "id": "hust_2026_diemchuan_caccnam",
        "school": "Đại học Bách khoa Hà Nội",
        "url": "https://ts.hust.edu.vn/tin-tuc/diem-chuan-trung-tuyen-dai-hoc-he-chinh-quy-nam-2023",
        "fallback_title": "Bảng Điểm chuẩn trúng tuyển Đại học hệ chính quy Đại học Bách khoa Hà Nội",
        "category": "Điểm chuẩn các năm",
    },
    {
        "id": "hust_2026_xacnhan_nhaphoc",
        "school": "Đại học Bách khoa Hà Nội",
        "url": "https://ts.hust.edu.vn/tin-tuc/huong-dan-thu-tuc-xac-nhan-nhap-hoc-doi-voi-thi-sinh-trung-tuyen-theo-hinh-thuc-xet-diem-thi-tot-nghiep-thpt",
        "fallback_title": "Hướng dẫn thủ tục Xác nhận nhập học Đại học Bách khoa Hà Nội",
        "category": "Hướng dẫn nhập học",
    },
]


def clean_text(text: str) -> str:
    """Làm sạch ký tự thừa, khoảng trắng trùng lặp."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def crawl_article(item: dict) -> dict:
    """Crawl chi tiết bài viết, trích xuất Title và Body Text đầy đủ nhất."""
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

            # Lấy nội dung văn bản (bỏ các thẻ script, style, nav, footer, header)
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            paragraphs = soup.find_all(["p", "li", "h3", "h4", "table", "tr"])
            content_list = []
            for p in paragraphs:
                txt = clean_text(p.get_text())
                if len(txt) > 20 and txt not in content_list:
                    content_list.append(txt)

            body_content = "\n\n".join(content_list)

            return {
                "id": item["id"],
                "school_name": item["school"],
                "title": title,
                "category": item["category"],
                "source_url": item["url"],
                "crawled_at": datetime.now().isoformat(),
                "status": "success",
                "content": body_content,
            }
    except Exception as e:
        print(f"⚠️ Lỗi khi cào {item['url']}: {e}")

    return {
        "id": item["id"],
        "school_name": item["school"],
        "title": item["fallback_title"],
        "category": item["category"],
        "source_url": item["url"],
        "crawled_at": datetime.now().isoformat(),
        "status": "fallback",
        "content": f"Nội dung thông báo chính thức thuộc {item['school']}. Chi tiết tra cứu tại {item['url']}.",
    }


def main():
    print("🚀 Bắt đầu crawl toàn bộ bài viết/thông báo tuyển sinh ĐẠI HỌC BÁCH KHOA HÀ NỘI...")

    for item in URLS_TO_CRAWL:
        data = crawl_article(item)
        filename = f"{item['id']}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(
            f"✅ Đã lưu: {filepath} ({data['school_name']} - {data['category']} - {len(data['content'])} chars)"
        )

    print(f"\n🎉 HOÀN THÀNH! Đã lưu {len(URLS_TO_CRAWL)} bài viết vào thư mục data/landing/news/")


if __name__ == "__main__":
    main()