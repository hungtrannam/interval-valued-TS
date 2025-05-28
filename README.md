# Interval-Valued Time Series Forecasting

## 📁 Cấu trúc
```bash
interval-valued-TS/
├── .github/workflows/        # Cấu hình CI/CD
├── MODELS/                   # Huấn luyện mô hình
├── dataset/                  # Dữ liệu đầu vào csv
├── builtEnv.sh               # Thiết lập môi trường ảo
├── gitUp.sh                  # Tự động commit & push
├── requirements.txt          # Thư viện Python cần thiết
└── README.md                 # Hướng dẫn
```

## 📁 Hướng dẫn

```bash
# Thiết lập môi trường python (Linux)
bash builtEnv.sh
source .venv/bin/activate
```

hoặc # Thiết lập môi trường python (Windows)
```bash
pip install -r requirements.txt
```

### Huấn luyện mô hình

```bash
./run_optun.sh
```
