# GVHMR: World-Grounded Human Motion Recovery & Web API Service

[![Paper](https://img.shields.io/badge/Paper-ArXiv%3A2409.06662-B31B1B.svg)](https://arxiv.org/abs/2409.06662)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://zju3dv.github.io/gvhmr)
[![SIGGRAPH Asia 2024](https://img.shields.io/badge/Conference-SIGGRAPH%20Asia%202024-green.svg)](https://sa2024.siggraph.org/)
[![Docker GPU](https://img.shields.io/badge/Docker-CUDA%2012.1-blue?logo=docker)](Dockerfile)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)](app/main.py)

> **World-Grounded Human Motion Recovery via Gravity-View Coordinates**  
> Zehong Shen\*, Huaijin Pi\*, Yan Xia, Zhi Cen, Sida Peng†, Zechen Hu, Hujun Bao, Ruizhen Hu, Xiaowei Zhou  
> *SIGGRAPH Asia 2024*

---

## 📌 Tổng Quan (Overview)

Dự án này tích hợp mô hình AI tiên tiến **GVHMR (SIGGRAPH Asia 2024)** để phôi phục chuyển động người 3D chuẩn xác theo không gian tọa độ trọng lực thế giới (World-Grounded 3D Human Motion Recovery), kết hợp với **Hệ thống Server FastAPI chuyên nghiệp** hỗ trợ xử lý đa người (Multi-Person Tracking & Recovery), tác vụ ngầm bất đồng bộ (Asynchronous Background Jobs), tính toán chỉ số đánh giá (MPJPE / PA-MPJPE) và đóng gói **Docker Container GPU**.

<p align="center">
    <img src="docs/example_video/project_teaser.gif" alt="GVHMR Teaser" width="90%"/>
</p>

---

## 🔥 Tính Năng Nổi Bật (Key Features)

- ⚡ **Hệ Thống Web API Chuẩn Doanh Nghiệp (FastAPI Architecture)**:
  - Cấu trúc thư mục chuẩn **Controller - Service - Model** giúp dễ dàng bảo trì và mở rộng.
  - Tích hợp xác thực bảo mật API Key (`X-API-Key`), hỗ trợ lưu trữ trạng thái tác vụ qua SQLite / MySQL.
  - Xử lý bất đồng bộ (Background Tasks) cho các video thời lượng lớn mà không làm nghẽn HTTP thread.
- 👥 **Hỗ Trợ Đa Người (Multi-Person 3D Motion Recovery)**:
  - Tự động theo vết nhiều người trong video (Top-N Person Tracking).
  - Trích xuất và render mesh 3D độc lập cho từng người (In-Camera Mesh & Global World Motion).
- 📊 **API Đánh Giá Sai Số 3D (Metrics Evaluation API)**:
  - Tính toán tự động chỉ số sai số **MPJPE** và **PA-MPJPE** (Procrustes Alignment).
  - Tự động chuyển đổi và căn chỉnh giữa các hệ khớp **SMPL 22/24 joints** và **H36M 17 joints**.
- 🐳 **Đóng Gói Docker & GPU Acceleration**:
  - Hỗ trợ chạy container trên **PyTorch 2.3.0 + CUDA 12.1**.
  - Tích hợp sẵn `docker-compose.yml` hỗ trợ NVIDIA Container Toolkit.

---

## 📂 Cấu Trúc Dự Án (Project Structure)

```text
GVHMR-Demo/
├── app/                          # Hệ thống FastAPI Server App
│   ├── controllers/              # API Controllers & Handlers (/api/v1/jobs, upload, metrics)
│   ├── services/                 # Logic nghiệp vụ (Job queue, Model inference, Render video)
│   ├── models/                   # Database ORM & Pydantic Data Schemas
│   ├── database.py               # Kết nối CSDL (SQLite / MySQL)
│   └── main.py                   # Điểm khởi chạy FastAPI Server
├── hmr4d/                        # Mô hình GVHMR & Pipeline Tiền xử lý (ViTPose, Tracker, DPVO)
├── tools/                        # Script chạy CLI, Demo & Train
│   ├── demo/                     # Demo dự đoán cho 1 video hoặc thư mục video
│   ├── train.py                  # Script huấn luyện & Đánh giá mô hình
│   └── unitest/                  # Unit tests cho API & Metrics
├── docs/                         # Hướng dẫn cài đặt & Ví dụ minh họa
├── inputs/                       # Nơi lưu trữ Model Checkpoints (.ckpt)
├── Dockerfile                    # Docker configuration cho môi trường GPU CUDA 12.1
├── docker-compose.yml            # Docker Compose orchestration
├── requirements.txt              # Danh sách các thư viện phụ thuộc Python
└── swagger.yaml                  # Tài liệu OpenAPI / Swagger UI Specification
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Server (Quick Start)

### Cách 1: Sử dụng Docker & Docker Compose (Khuyên dùng)

1. **Chuẩn bị Checkpoint**:
   Đặt file checkpoint `gvhmr_siga24_release.ckpt` vào thư mục `inputs/checkpoints/gvhmr/`.

2. **Khởi chạy bằng Docker Compose**:
   ```bash
   docker-compose up -d --build
   ```

3. **Kiểm tra trạng thái Container**:
   Server sẽ mở tại port `8000`. Bạn có thể kiểm tra sức khỏe hệ thống tại:
   `http://localhost:8000/api/v1/health`

---

### Cách 2: Chạy Trực Tiếp Cục Bộ (Local Setup)

1. **Yêu cầu môi trường**:
   - Python **3.10+**
   - PyTorch **2.3.0+cu121** & CUDA 12.1
   - FFmpeg (đã thêm vào PATH hệ thống)

2. **Cài đặt thư viện phụ thuộc**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Khởi chạy API Server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Truy cập Swagger UI Documentation**:
   Mở trình duyệt truy cập: `http://localhost:8000/docs` hoặc `http://localhost:8000/redoc`

---

## 📡 Danh Sách API Endpoints Chính

| HTTP Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Kiểm tra sức khỏe Server & GPU status |
| `POST` | `/api/v1/video/upload` | Tải video đầu vào lên server (`X-API-Key`) |
| `POST` | `/api/v1/jobs` | Khởi tạo tác vụ suy luận 3D ngầm cho video |
| `GET` | `/api/v1/jobs/{job_id}` | Theo dõi tiến độ & trạng thái xử lý tác vụ |
| `GET` | `/api/v1/download/{filename}` | Tải xuống video kết quả 3D mesh đã render |
| `POST` | `/api/v1/metrics/evaluate` | Tính toán sai số MPJPE & PA-MPJPE |

---

## 💻 Sử Dụng CLI (Command Line Demo)

Nếu muốn chạy trực tiếp bằng dòng lệnh không qua API Server:

```bash
# Xử lý 1 video duy nhất (thêm -s nếu camera cố định)
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s

# Xử lý toàn bộ thư mục video
python tools/demo/demo_folder.py -f inputs/demo/folder_in -d outputs/demo/folder_out -s
```

### Đánh giá & Huấn luyện lại (Train & Evaluate)

```bash
# Chạy Test trên 3DPW, RICH và EMDB
python tools/train.py global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt

# Huấn luyện mô hình
python tools/train.py exp=gvhmr/mixed/mixed
```

---

## 📚 Tri Thức Khoa Học (Citation)

Nếu bạn sử dụng mã nguồn hoặc mô hình GVHMR cho mục đích nghiên cứu, vui lòng trích dẫn bài báo gốc:

```bibtex
@inproceedings{shen2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2024}
}
```

---

## 🙏 Lời Cảm Ơn (Acknowledgements)

Chúng tôi chân thành cảm ơn tác giả của các dự án nguồn mở: [WHAM](https://github.com/yohanshin/WHAM), [4D-Humans](https://github.com/shubham-goel/4D-Humans), và [ViTPose-Pytorch](https://github.com/gpastal24/ViTPose-Pytorch).
