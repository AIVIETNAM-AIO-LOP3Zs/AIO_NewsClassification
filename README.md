# AI News Classification API

Hệ thống phân loại tin tức sử dụng FastAPI và các mô hình học máy.

## Cấu trúc thư mục

- `app/`: Source code chính của backend (FastAPI)
  - `api/v1/`: Chứa các router API
  - `core/`: Chứa các file cấu hình, logging
  - `ml/`: Chứa mã xử lý ML/Inference
  - `schemas/`: Chứa các Pydantic models
  - `services/`: Chứa logic xử lý phân loại
- `data/`: Dữ liệu và tài liệu về dataset.
- `models/`: Thư mục chứa các mô hình đã huấn luyện (VD: BERT weights, TF-IDF pkl, ...)
- `notebooks/`: Các Jupyter notebooks (Training, Preprocessing, EDA)
- `docs/`: Chứa các tài liệu thiết kế và plan dự án.
- `tests/`: Unit tests cho hệ thống.

---

## 🚀 Hướng dẫn cài đặt và sử dụng bằng Docker (Khuyên dùng)

Để chạy hệ thống nhanh gọn nhất, chúng ta sẽ sử dụng Docker Compose. Có hai trường hợp phổ biến: Chạy hệ thống nội bộ (chỉ truy cập từ `localhost`) và Chạy hệ thống kèm Ngrok (để có public URL chia sẻ ra ngoài).

### Bước 1: Chuẩn bị môi trường & Model
1. Copy file `.env.example` thành `.env`
   ```bash
   cp .env.example .env
   ```
2. Nếu bạn muốn sử dụng Ngrok (Trường hợp 2), mở file `.env` và thêm chuỗi token của bạn vào biến `NGROK_AUTHTOKEN`. *(Lấy token tại [dashboard của Ngrok](https://dashboard.ngrok.com/get-started/your-authtoken))*
3. Đảm bảo bạn đã đưa đầy đủ các file model đã train vào thư mục `models/`.

### Trường hợp 1: Chạy hệ thống KHÔNG có Ngrok (Local Only)
Lựa chọn này sẽ không bật service Ngrok.
```bash
docker compose up --build -d
```
- API nội bộ sẽ khả dụng ở: `http://localhost:8000`
- API Documentation (Swagger): `http://localhost:8000/docs`

### Trường hợp 2: Chạy hệ thống CÓ kèm Ngrok (Public API)
Lựa chọn này sẽ bật cả backend và tự động cấu hình Ngrok để expose cổng `8000` ra public internet thông qua profile `ngrok`.
```bash
docker compose --profile ngrok up --build -d
```
- Để lấy Public URL do Ngrok cung cấp, hãy truy cập vào Web Interface của Ngrok tại: **`http://localhost:4040`**. Tại đây bạn sẽ thấy một link dạng `https://<random-id>.ngrok-free.app`.
- Bạn có thể gửi request đến URL public đó hoàn toàn tương tự như khi gọi qua `localhost:8000`.

---

## 💻 Hướng dẫn chạy Native (Không dùng Docker)

Nếu bạn cần debug trực tiếp và không muốn dùng Docker:

```bash
# 1. Cài đặt dependencies và tạo môi trường tự động bằng uv
uv sync

# 2. Kích hoạt môi trường ảo
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 3. Chạy server FastAPI
uvicorn app.main:app --reload --port 8000
```

---

## 📖 API Endpoints chính

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/api/v1/news/health` | Kiểm tra trạng thái hệ thống và xem model nào đang load |
| `GET` | `/api/v1/news/model/info` | Thông tin chi tiết của mô hình (version, labels, configs) |
| `POST` | `/api/v1/news/classify` | Phân loại 1 bài báo đơn lẻ |
| `POST` | `/api/v1/news/classify/batch` | Phân loại cùng lúc một danh sách các bài báo |

**Ví dụ gọi API phân loại bài viết (cURL):**
```bash
curl -X POST http://localhost:8000/api/v1/news/classify \
  -H "Content-Type: application/json" \
  -d '{
        "title": "Kinh tế thị trường", 
        "content": "Thị trường chứng khoán hôm nay có nhiều biến động..."
      }'
```

**Ví dụ cấu trúc dữ liệu trả về:**
```json
{
  "status": "success",
  "model_version": "news_classifier",
  "result": {
    "label": "kinh_te",
    "confidence": 0.89,
    "probabilities": {
      "kinh_te": 0.89,
      "the_thao": 0.05
    },
    "below_threshold": false
  }
}
```