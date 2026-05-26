# 📰 BBC News Classification API

Hệ thống phân loại tin tức tự động dựa trên nội dung bài viết, sử dụng **FastAPI** hiệu năng cao kết hợp với hai kiến trúc mô hình học máy:
1. **Lightweight ML**: TF-IDF kết hợp với mô hình học máy Linear Support Vector Classifier (LinearSVC).
2. **Deep Learning / Transformer**: Mô hình BERT (`bert-base-uncased`) được tinh chỉnh (fine-tuned) trên tập dữ liệu tin tức.

Hệ thống hỗ trợ phân loại các bài viết tiếng Anh vào **5 danh mục chính**:
*   ⚽ **Sport** (Thể thao)
*   💼 **Business** (Kinh doanh)
*   ⚖️ **Politics** (Chính trị)
*   💻 **Tech** (Công nghệ)
*   🎭 **Entertainment** (Giải trí)

---

## 📂 Cấu trúc thư mục dự án

```text
├── app/                  # Mã nguồn chính của ứng dụng FastAPI
│   ├── api/              # Định nghĩa Router API các phiên bản (v1)
│   ├── core/             # Cấu hình hệ thống, logging, biến môi trường
│   ├── ml/               # Logic quản lý vòng đời và suy luận (inference) của model
│   ├── schemas/          # Định nghĩa kiểu dữ liệu truyền nhận (Pydantic models)
│   ├── services/         # Logic nghiệp vụ xử lý phân loại chính
│   └── static/           # Giao diện Single Page Application (SPA) tích hợp sẵn
├── data/                 # Thư mục chứa tập dữ liệu huấn luyện (bbc_clean.csv,...)
├── models/               # Nơi lưu trữ các artifacts / trọng số mô hình đã huấn luyện
│   ├── tfidf/            # Chứa các file pkl của TF-IDF pipeline
│   └── bert/             # Chứa checkpoint đã fine-tuned của BERT
├── tests/                # Bộ kiểm thử tự động (Unit/Integration tests)
├── Dockerfile            # Cấu hình build Docker image dạng Multi-stage tối ưu
├── docker-compose.yml    # Cấu hình container chạy API và Tunnel Ngrok
├── pyproject.toml        # Quản lý dependency và metadata dự án qua Astral uv
├── train_tfidf.py        # Script huấn luyện mô hình TF-IDF SVC nhanh gọn
└── train_bert.py         # Script fine-tuning mô hình BERT Transformer
```

---

## 🚀 Hướng dẫn triển khai từng bước chi tiết

Theo dõi và thực hiện lần lượt các bước dưới đây để cài đặt, chạy huấn luyện, chạy server native hoặc đóng gói với Docker.

### Bước 1: Clone dự án và truy cập thư mục nguồn

Mở terminal của bạn và thực hiện các lệnh sau để tải mã nguồn dự án về máy:

```bash
# Clone dự án từ repository (Thay thế bằng đường dẫn Git thực tế nếu có)
git clone <repository_url>

# Di chuyển vào thư mục dự án
cd AIO_NewsClassification
```

### Bước 2: Cài đặt và cấu hình môi trường ảo bằng `uv`

Dự án này sử dụng công cụ **`uv`** (Astral uv) — trình quản lý gói Python thế hệ mới cực nhanh viết bằng Rust thay thế cho `pip` truyền thống.

1. **Cài đặt `uv`** (nếu máy bạn chưa cài đặt):
   ```bash
   # Dành cho macOS / Linux:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Dành cho Windows (PowerShell):
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. **Khởi tạo virtual environment và đồng bộ dependencies**:
   `uv` sẽ tự động đọc cấu hình từ `pyproject.toml`, tạo môi trường ảo `.venv` và cài đặt các phiên bản package chuẩn xác:
   ```bash
   # Đồng bộ và cài đặt tất cả thư viện (bao gồm cả môi trường Dev)
   uv sync
   
   # Kích hoạt môi trường ảo vừa tạo
   source .venv/bin/activate    # macOS / Linux
   # Hoặc nếu chạy Windows:
   # .venv\Scripts\activate
   ```

### Bước 3: Thiết lập các biến môi trường (Copy `.env`)

Hệ thống đọc cấu hình từ tệp tin `.env`. Bạn cần tạo tệp này từ tệp mẫu:

```bash
# Tạo file .env từ file template có sẵn
cp .env.example .env
```

Mở tệp `.env` vừa tạo và cấu hình các giá trị phù hợp với nhu cầu sử dụng của bạn:
*   `ACTIVE_MODELS`: Các mô hình sẽ được nạp vào bộ nhớ khi API khởi động (Ví dụ: `tfidf` để chạy siêu nhẹ, `bert` nếu có GPU hoặc tài nguyên mạnh, hoặc cả hai `tfidf,bert`).
*   `BERT_DEVICE`: Thiết bị chạy BERT (`cpu`, `cuda` trên GPU Nvidia, hoặc `mps` trên chip Apple Silicon).
*   `NGROK_AUTHTOKEN`: Token cá nhân lấy từ [Dashboard Ngrok](https://dashboard.ngrok.com/) nếu bạn muốn public API ra ngoài internet.

---

## 🏋️ Huấn luyện Mô hình (Model Training)

Trước khi chạy API, bạn cần có các artifacts mô hình đã được huấn luyện để hệ thống có thể nạp vào bộ nhớ. Đảm bảo tập dữ liệu **`data/bbc_clean.csv`** đã tồn tại đầy đủ.

### Lựa chọn A: Huấn luyện mô hình TF-IDF + LinearSVC (Nhẹ & Nhanh)
Script này sẽ xử lý vector hóa văn bản bằng TF-IDF và huấn luyện mô hình phân loại tuyến tính chỉ trong vài giây.

```bash
# Chạy script huấn luyện TF-IDF
uv run train_tfidf.py
```
> [!NOTE]
> Sau khi chạy xong, bộ trọng số sẽ được xuất tự động ra thư mục: `models/tfidf/` bao gồm các tệp `news_classifier.pkl`, `tfidf_vectorizer.pkl`, `label_encoder.pkl` và `label_mapping.json`.

### Lựa chọn B: Fine-tuning mô hình BERT Transformer (Độ chính xác cao)
Script này thực hiện quá trình tinh chỉnh mô hình Transformer `bert-base-uncased` trên toàn bộ tập dữ liệu bằng PyTorch.

```bash
# Chạy script fine-tune BERT
uv run train_bert.py
```
> [!IMPORTANT]
> Quá trình chạy BERT có thể mất từ 10 - 30 phút tùy thuộc vào thiết bị phần cứng của bạn (khuyên dùng máy có GPU/MPS). Kết quả checkpoint mô hình sẽ được lưu vào: `models/bert/bbc-bert-finetuned/`.

---

## 💻 1. Chạy Native (Không dùng Docker - Phù hợp cho Debug)

Nếu bạn muốn chạy thử nghiệm nhanh chóng trên máy local:

```bash
# Đảm bảo đã kích hoạt virtual environment và cấu hình .env đầy đủ
# Khởi chạy server FastAPI bằng Uvicorn
uv run uvicorn app.main:app --reload --port 8000
```

*   **API Local**: `http://localhost:8000`
*   **Trang kiểm tra tài liệu API tương tác (Swagger UI)**: `http://localhost:8000/docs`
*   **Tài liệu API dạng Redoc**: `http://localhost:8000/redoc`
*   **Giao diện Web Client SPA**: Truy cập ngay trang chủ `http://localhost:8000` để thử nghiệm giao diện phân loại trực quan.

---

## 🐳 2. Triển khai với Docker & Docker Compose (Khuyên dùng trong Production)

Docker giúp đóng gói toàn bộ ứng dụng và chạy đồng nhất trên mọi môi trường mà không cần cài đặt cấu hình thủ công.

> [!TIP]
> File `docker-compose.yml` được cấu hình sử dụng **Bind-mounting** thư mục local `models/` vào container (`ro` - read-only). Điều này giúp bạn có thể cập nhật các mô hình đã train ở máy chủ bên ngoài mà không cần phải thực hiện rebuild lại Docker Image!

### Trường hợp 1: Chạy API nội bộ (Local Only)
Chạy dịch vụ API FastAPI trên môi trường container và mở cổng local.

```bash
# Xây dựng image và chạy container ở chế độ background
docker compose up --build -d
```
*   Ứng dụng khả dụng tại địa chỉ: `http://localhost:8000`
*   Theo dõi logs hệ thống: `docker compose logs -f api`

### Trường hợp 2: Chạy API và tự động mở Public Tunnel qua Ngrok
Trường hợp bạn cần chia sẻ API ra bên ngoài internet cho đối tác hoặc ứng dụng khách sử dụng mà không cần cấu hình IP tĩnh hay port-forwarding.

```bash
# Đảm bảo đã khai báo NGROK_AUTHTOKEN trong file .env
# Khởi chạy kèm profile ngrok
docker compose --profile ngrok up --build -d
```
1.  Truy cập vào trang quản lý Web Interface của Ngrok trên local tại: **`http://localhost:4040`**.
2.  Lấy Public URL ngẫu nhiên sinh ra dạng: `https://<random-subdomain>.ngrok-free.app`.
3.  Sử dụng URL này để gửi các request phân loại từ bất kỳ đâu trên thế giới!

---

## 📖 Chi tiết các API Endpoints chính

### 1. Kiểm tra trạng thái hệ thống
*   **Method**: `GET`
*   **Endpoint**: `/api/v1/news/health`
*   **Mô tả**: Trả về trạng thái hoạt động của hệ thống và danh sách các mô hình ML đã được nạp thành công vào bộ nhớ.

```bash
curl -X GET http://localhost:8000/api/v1/news/health
```

*Phản hồi mẫu:*
```json
{
  "status": "healthy",
  "loaded_models": ["tfidf", "bert"],
  "app_version": "1.0.0"
}
```

### 2. Lấy thông tin chi tiết mô hình
*   **Method**: `GET`
*   **Endpoint**: `/api/v1/news/model/info`
*   **Mô tả**: Cung cấp chi tiết các nhãn phân loại, phiên bản model, ngưỡng tin cậy tối thiểu và độ dài văn bản tối đa.

```bash
curl -X GET http://localhost:8000/api/v1/news/model/info
```

### 3. Phân loại bài viết đơn lẻ
*   **Method**: `POST`
*   **Endpoint**: `/api/v1/news/classify`
*   **Query Parameter**: `model_type` (tùy chọn: `tfidf` hoặc `bert` để ghi đè model mặc định)

```bash
curl -X POST http://localhost:8000/api/v1/news/classify?model_type=tfidf \
  -H "Content-Type: application/json" \
  -d '{
        "title": "Arsenal secure dramatic victory in final minutes",
        "content": "The match was heading towards a goalless draw until a brilliant strike in the 92nd minute sealed the three points for the home side..."
      }'
```

*Phản hồi mẫu:*
```json
{
  "result": {
    "label": "sport",
    "confidence": 0.9845,
    "probabilities": {
      "sport": 0.9845,
      "business": 0.0034,
      "politics": 0.0012,
      "tech": 0.0082,
      "entertainment": 0.0027
    },
    "below_threshold": false,
    "model_used": "tfidf"
  },
  "model_version": "news_classifier"
}
```

### 4. Phân loại hàng loạt (Batch Classification)
*   **Method**: `POST`
*   **Endpoint**: `/api/v1/news/classify/batch`
*   **Mô tả**: Gửi danh sách tối đa 32 bài viết để phân loại đồng thời nhằm tăng hiệu năng xử lý.

```bash
curl -X POST http://localhost:8000/api/v1/news/classify/batch \
  -H "Content-Type: application/json" \
  -d '{
        "articles": [
          {
            "title": "Tech stocks slide",
            "content": "Tech companies experienced a significant drop in stock values amid inflation concerns..."
          },
          {
            "title": "New smartphone release",
            "content": "A major tech company unveiled its latest smartphone model featuring artificial intelligence integration..."
          }
        ],
        "model_type": "bert"
      }'
```

---

## 🧪 Chạy Kiểm thử Tự động (Running Tests)

Để kiểm tra độ ổn định của hệ thống và tính đúng đắn của logic suy luận:

```bash
# Chạy bộ test suite thông qua pytest trong môi trường ảo
uv run pytest -v
```

> [!TIP]
> Các test case được giả lập (stubbed/mocked) các đối tượng mô hình lớn, giúp kiểm tra hệ thống nhanh chóng mà không yêu cầu tài nguyên GPU hay nạp tệp mô hình BERT thật.