# Beijing Air Quality Dashboard

Dashboard trực quan hóa toàn bộ kết quả phân tích dự án Beijing Air Quality với Machine Learning và Semi-Supervised Learning.

## Tính Năng

### Yêu Cầu 1: Preprocessing & EDA
- Tổng quan dataset (420K records, 2013-2017)
- Phân tích chất lượng dữ liệu (missing values, duplicates)
- Phân phối các biến khí quyển và thời tiết
- Ma trận tương quan và feature relationships

### Yêu Cầu 2: Supervised Modeling
- **Classification**: 5 models (Random Forest best: 84.9% accuracy)
- **Regression**: 4 models (Gradient Boosting best: RMSE 30.12)
- **Time Series**: ARIMA forecasting (MAE 35.21)
- Feature importance và model insights

### Yêu Cầu 3: Semi-Supervised Learning
- **Labeled Size Impact**: So sánh 5%, 10%, 20% labeled data
- **Model Comparison**: HistGradientBoosting vs Random Forest (+19.1% F1)
- **Adaptive τ Schedule**: Fixed vs Aggressive schedule (+3.4% F1)
- **Cross-Analysis**: Best config 10% + HGBC + Aggressive τ (F1=0.5088)

## Cài Đặt

### 1. Tạo Virtual Environment (Khuyến Nghị)

```bash
conda create -n dashboard_env python=3.11
conda activate dashboard_env
```

### 2. Cài Đặt Dependencies

```bash
cd dashboard
pip install -r requirements.txt
```

### 3. Chạy Dashboard

```bash
streamlit run app.py
```

Dashboard sẽ mở tại: `http://localhost:8501`

## Cấu Trúc Thư Mục

```
dashboard/
├── app.py                          # Trang chủ
├── pages/
│   ├── 1_Preprocessing_EDA.py     # Yêu cầu 1
│   ├── 2_Supervised_Modeling.py   # Yêu cầu 2
│   └── 3_Semi_Supervised.py       # Yêu cầu 3
├── assets/                         # Hình ảnh, CSS, JS
├── utils/                          # Helper functions
├── requirements.txt                # Dependencies
└── README.md                       # Tài liệu này
```

## Dữ Liệu

Dashboard tự động load dữ liệu từ:
- `../data/processed/dataset_for_semi.parquet` - Dữ liệu đã xử lý
- `../data/processed/*_experiments/` - Kết quả thí nghiệm

**Lưu ý:** View Splitting Experiment đang chạy nên chưa có trên dashboard.

## Công Nghệ

- **Frontend**: Streamlit + Tailwind CSS
- **Visualization**: Plotly (interactive charts)
- **Data**: Pandas + PyArrow (parquet)
- **Styling**: Ocean blue gradient theme với smooth animations

## Tính Năng Đặc Biệt

- **Responsive Design**: Tương thích mobile và desktop
- **Interactive Charts**: Hover, zoom, download visualizations
- **Real-time Data**: Tự động load latest experiment results
- **Clean UI**: Minimal icons, focus on data
- **Smooth Animations**: Fade in, slide up effects với CSS

## Performance

- **Load time**: < 2 giây
- **Memory**: ~200MB (với full dataset)
- **Charts**: Plotly WebGL cho large datasets

## Troubleshooting

### Dashboard không khởi động
```bash
# Kiểm tra Streamlit version
streamlit --version

# Reinstall nếu cần
pip install --upgrade streamlit
```

### Không load được dữ liệu
```bash
# Kiểm tra đường dẫn data directory
ls ../data/processed/

# Chạy preprocessing notebook trước nếu chưa có data
```

### Charts không hiển thị
```bash
# Clear Streamlit cache
streamlit cache clear
```

## Customization

### Thay đổi màu chủ đạo
Sửa gradient trong `app.py`:
```css
background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 50%, #0369a1 100%);
```

### Thêm page mới
1. Tạo file trong `pages/` với prefix số (e.g., `4_New_Page.py`)
2. Streamlit tự động detect và thêm vào sidebar

## Tác Giả

- Data Mining Team
- January 2026
- Contact: datamining@example.com

## License

MIT License - Xem LICENSE.txt trong project root

---

**Lưu ý:** Dashboard này phục vụ mục đích học tập và nghiên cứu. Dữ liệu từ UCI Machine Learning Repository.
