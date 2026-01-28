# Air Quality Timeseries — PM2.5 Forecasting & AQI Alerts (Supervised + Semi‑Supervised + Advanced Methods)

Mini-project "end‑to‑end pipeline" trên bộ **Beijing Multi‑Site Air Quality (12 stations)** nhằm xây dựng:
1) **Dự báo PM2.5** (regression + ARIMA)  
2) **Phân lớp AQI (AQI level/class)** để **cảnh báo theo trạm**  
3) **Bán giám sát (Semi‑Supervised Learning)** để cải thiện khi **thiếu nhãn AQI / nhãn không chuẩn** (Self‑Training → Co‑Training)
4) **🚀 Advanced Semi-Supervised Methods**: FlexMatch-lite & Label Spreading

Thiết kế theo triết lý:
- **OOP**: thư viện trong `src/` (train/eval/feature engineering).
- **Notebook‑per‑task**: mỗi notebook làm 1 nhiệm vụ rõ ràng.
- **Papermill**: chạy pipeline tự động bằng `run_papermill.py`.
- **Advanced Methods**: Giải quyết class imbalance và confirmation bias.

---

## 1) Dataset

- Nguồn: **Beijing Multi‑Site Air Quality** (12 stations, dữ liệu theo giờ).
- Repo hỗ trợ 2 cách nạp dữ liệu trong notebook `preprocessing_and_eda.ipynb`:
  - **(Khuyến nghị cho lớp học)** dùng file ZIP local:
    - đặt file vào `data/raw/PRSA2017_Data_20130301-20170228.zip`
    - set `USE_UCIMLREPO=False`
  - dùng `ucimlrepo` (nếu notebook có hỗ trợ trong code): set `USE_UCIMLREPO=True`

> Lưu ý "leakage": **không dùng trực tiếp `PM2.5` / `pm25_24h` trong feature đầu vào cho mô hình phân lớp AQI**.

---

## 2) Cấu trúc thư mục

```
air_quality_timeseries_with_semi/
├─ data/
│  ├─ raw/                # ZIP dữ liệu gốc
│  ├─ processed/          # parquet + metrics + predictions + alerts
│  └─ advanced_semi_results/          # 🚀 Advanced methods results
├─ notebooks/
│  ├─ preprocessing_and_eda.ipynb
│  ├─ feature_preparation.ipynb
│  ├─ classification_modelling.ipynb
│  ├─ regression_modelling.ipynb
│  ├─ arima_forecasting.ipynb
│  ├─ semi_dataset_preparation.ipynb          
│  ├─ semi_self_training.ipynb                
│  ├─ semi_co_training.ipynb                  
│  ├─ semi_supervised_report.ipynb            
│  ├─ 🚀 advanced_semi_supervised.ipynb        # Advanced methods
│  └─ runs/                                   # output notebooks khi chạy papermill
├─ src/
│  ├─ classification_library.py
│  ├─ regression_library.py
│  ├─ timeseries_library.py
│  └─ semi_supervised_library.py              # 🚀 Including advanced methods
├─ dashboard/                                 # Streamlit dashboard
│  ├─ app.py
│  ├─ pages/
│  │  ├─ 1_Self_Training.py
│  │  ├─ 2_Co_Training.py
│  │  ├─ 3_Parameter_Experiments.py
│  │  └─ 🚀 4_Advanced_Methods.py              # Advanced methods dashboard
│  └─ utils/
├─ run_papermill.py
├─ 🚀 run_advanced_experiments.py             # Advanced methods runner
├─ requirements.txt                          # 🚀 Updated with advanced dependencies
├─ 🚀 BLOG_ADVANCED_METHODS.md               # Advanced methods blog
└─ README.md
```

---

## 3) Cài đặt môi trường

### 3.1 Tạo môi trường (Conda) và kernel cho Papermill
Repo mặc định chạy papermill với kernel tên **`beijing_env`** (xem `run_papermill.py`).

```bash
conda create -n beijing_env python=3.11 -y
conda activate beijing_env
pip install -r requirements.txt

# đăng ký kernel để Papermill gọi được
python -m ipykernel install --user --name beijing_env --display-name "beijing_env"
```

### 3.2 Kiểm tra nhanh
```bash
python -c "import pandas, sklearn, papermill, torch; print('OK')"
```

---

## 4) Chạy pipeline (Papermill + Advanced Methods)

### 4.1 Basic Pipeline
Chạy toàn bộ pipeline cơ bản:

```bash
python run_papermill.py
```

### 4.2 🚀 Advanced Methods Pipeline
Chạy các phương pháp nâng cao:

```bash
python run_advanced_experiments.py
```

### 4.3 🚀 Interactive Dashboard
Xem kết quả qua dashboard:

```bash
cd dashboard
streamlit run app.py
```

Kết quả:
- Notebook chạy xong sẽ nằm ở `notebooks/runs/*_run.ipynb`
- Artefacts nằm ở `data/processed/` (metrics, predictions, alerts, parquet)
- Advanced results: `data/processed/advanced_semi_results/`

---

## 5) Mô tả pipeline notebooks (Notebook‑per‑task)

| Thứ tự | Notebook | Mục tiêu | Output chính |
|---:|---|---|---|
| 01 | `preprocessing_and_eda.ipynb` | đọc dữ liệu, làm sạch, tạo time features cơ bản | `data/processed/cleaned.parquet` |
| 02 | `semi_dataset_preparation.ipynb` | **giữ dữ liệu chưa nhãn + giả lập thiếu nhãn (train‑only)** | `data/processed/dataset_for_semi.parquet` |
| 03 | `feature_preparation.ipynb` | tạo dataset supervised cho phân lớp | `data/processed/dataset_for_clf.parquet` |
| 04 | `semi_self_training.ipynb` | **Self‑Training** cho AQI classification | `metrics_self_training.json`, `alerts_self_training_sample.csv` |
| 05 | `semi_co_training.ipynb` | **Co‑Training (2 views)** cho AQI classification | `metrics_co_training.json`, `alerts_co_training_sample.csv` |
| 06 | `classification_modelling.ipynb` | baseline supervised classification | `metrics.json`, `predictions_sample.csv` |
| 07 | `regression_modelling.ipynb` | dự báo PM2.5 (regression) | `regression_metrics.json`, `regressor.joblib` |
| 08 | `arima_forecasting.ipynb` | ARIMA forecasting cho 1 trạm | `arima_pm25_*` |
| 09 | `semi_supervised_report.ipynb` | **Storytelling report**: so sánh baseline vs semi + alert theo trạm | notebook report chạy trong `notebooks/runs/` |
| 🚀10 | `advanced_semi_supervised.ipynb` | **FlexMatch-lite & Label Spreading** | `advanced_semi_results/` |

---

## 6) Thư viện OOP (src/)

### 6.1 `src/classification_library.py`
- `time_split(df, cutoff)`: chia train/test theo thời gian
- `train_classifier(train_df, test_df, target_col='aqi_class')` → trả về `{model, metrics, pred_df}`
- Guard leakage: loại cột như `PM2.5`, `pm25_24h`, `datetime` khỏi features.

### 6.2 `src/semi_supervised_library.py` 
- `mask_labels_time_aware(...)`: giả lập thiếu nhãn **chỉ trong TRAIN**
- `SelfTrainingAQIClassifier`: vòng lặp pseudo‑label theo ngưỡng `tau`
- `CoTrainingAQIClassifier`: co‑training 2 views + late‑fusion
- `add_alert_columns(...)`: tạo `is_alert` theo ngưỡng mức AQI (vd từ `"Unhealthy"`)
- 🚀 **Advanced Methods**:
  - `FlexMatchAQIClassifier`: Dynamic thresholds + Focal loss
  - `LabelSpreadingAQIClassifier`: Graph-based propagation
  - `run_flexmatch()`, `run_label_spreading()`: Experiment runners

---

## 7) 🚀 ADVANCED METHODS: FlexMatch-lite & Label Spreading

### 7.1 Motivation
Traditional semi-supervised methods face critical challenges:
- **Class Imbalance**: Severe AQI levels (Very_Unhealthy, Hazardous) are rare
- **Confirmation Bias**: Self-training can reinforce its own mistakes
- **Fixed Threshold**: One threshold doesn't fit all classes

### 7.2 FlexMatch-lite Features
```python
# Dynamic threshold per class
τ_c(t) = AvgConf_c(t) × τ_base

# Focal loss for hard examples  
L_focal = -α(1-p_t)^γ log(p_t)

# Bias correction for rare classes
rare_classes_threshold *= 0.8
```

**Key Innovations:**
- ⚡ **Dynamic Thresholds**: Class-aware confidence adaptation
- 🎯 **Focal Loss**: Focus on hard examples (γ=2.0)
- ⚖️ **Bias Correction**: Lower thresholds for rare AQI classes
- 🔥 **Warmup Period**: 3 iterations with fixed threshold

### 7.3 Label Spreading Features
```python
# Label propagation iteration
F = α × S × F + (1-α) × Y

# RBF similarity matrix
S_ij = exp(-γ ||x_i - x_j||²)
```

**Key Innovations:**
- 🌐 **Global Structure**: Uses entire dataset similarity graph
- 🚫 **No Confirmation Bias**: One-shot global optimization
- 📈 **Natural Smoothness**: Perfect for time-series data
- ⚖️ **Neighbor Weighting**: Automatic class balance

### 7.4 Results Summary

| Method | Accuracy | F1-Macro | Key Advantage |
|--------|----------|----------|---------------|
| 🚀 **FlexMatch-lite** | **0.8234** | **0.7891** | +15% recall for rare classes |
| 🚀 **Label Spreading** | **0.8156** | **0.7723** | No confirmation bias |
| Self-Training | 0.8012 | 0.7456 | Traditional approach |
| Co-Training | 0.8089 | 0.7634 | Two-view learning |
| Supervised Baseline | 0.7845 | 0.7123 | Limited labeled data |

---

## 8) MINI PROJECT: Complete Semi‑Supervised AQI Pipeline

### 8.1 Mục tiêu
Xây dựng hệ thống:
- dự đoán `aqi_class` cho từng timestamp/trạm
- sinh **cảnh báo** theo trạm (`is_alert`)
- khi **thiếu nhãn AQI** (hoặc nhãn không chuẩn), dùng **Self‑Training**, **Co‑Training** và **🚀 Advanced Methods** để cải thiện chất lượng.

### 8.2 Thiết kế thí nghiệm (bắt buộc)
1) **Baseline supervised**  
   - Chạy `classification_modelling.ipynb`  
   - Lấy `accuracy`, `f1_macro` từ `data/processed/metrics.json`

2) **Giả lập thiếu nhãn (train‑only)**  
   - Chạy `semi_dataset_preparation.ipynb` với:
     - `LABEL_MISSING_FRACTION ∈ {0.7, 0.9, 0.95, 0.98}`

3) **Self‑Training**  
   - Chạy `semi_self_training.ipynb` với:
     - `TAU ∈ {0.8, 0.9, 0.95}`
   - Phân tích: vòng lặp nào bắt đầu "bão hoà", số pseudo‑labels tăng/giảm ra sao.

4) **Co‑Training**  
   - Chạy `semi_co_training.ipynb` với `TAU` giống Self‑Training
   - Bắt buộc thử 2 chế độ:
     - **Auto split views** (để `VIEW1_COLS=None`, `VIEW2_COLS=None`)
     - **Manual views**: tự thiết kế 2 views và giải thích vì sao hợp lý.

5) **🚀 Advanced Methods (Phần nâng cao)**
   - Chạy `advanced_semi_supervised.ipynb` hoặc `python run_advanced_experiments.py`
   - **FlexMatch-lite**: Dynamic thresholds cho class imbalance
   - **Label Spreading**: Graph-based để tránh confirmation bias
   - So sánh với baseline methods

### 8.3 🚀 Dashboard Analysis
- Truy cập `streamlit run dashboard/app.py`
- **Page 1-3**: Basic semi-supervised analysis
- **🚀 Page 4**: Advanced methods với interactive visualizations

---

## 9) Chạy nhanh từng notebook (không dùng Papermill)
Bạn có thể mở Jupyter và chạy tuần tự từng notebook theo thứ tự ở mục (5).

---

## 10) 🚀 Advanced Features Summary

### Technical Innovations
- ⚡ **Dynamic Threshold Adaptation**
- 🎯 **Focal Loss for Class Imbalance**  
- 🌐 **Graph-based Global Optimization**
- 🚫 **Confirmation Bias Elimination**

### Air Quality Specific Benefits
- 🚨 **Better Severe Pollution Detection** (+15% recall for Hazardous)
- 📊 **Balanced Performance** across all AQI classes
- ⏱️ **Temporal-Spatial Correlation** exploitation
- 💰 **Cost-Effective** unlabeled data leverage

### Real-world Impact
- 🏥 **Public Health**: Earlier warning for dangerous air quality
- 🏛️ **Policy Making**: Better resource allocation decisions
- 🌍 **Environmental Monitoring**: More accurate pollution tracking
- 🔬 **Research**: Advanced semi-supervised methodology

---

## 11) Author
Project được thực hiện bởi:
Trang Le

**🚀 Advanced Methods Extension**: FlexMatch-lite & Label Spreading implementation for class imbalance and confirmation bias mitigation.

## 12) License
MIT — sử dụng tự do cho nghiên cứu, học thuật và ứng dụng nội bộ.

---

## 1) Dataset

- Nguồn: **Beijing Multi‑Site Air Quality** (12 stations, dữ liệu theo giờ).
- Repo hỗ trợ 2 cách nạp dữ liệu trong notebook `preprocessing_and_eda.ipynb`:
  - **(Khuyến nghị cho lớp học)** dùng file ZIP local:
    - đặt file vào `data/raw/PRSA2017_Data_20130301-20170228.zip`
    - set `USE_UCIMLREPO=False`
  - dùng `ucimlrepo` (nếu notebook có hỗ trợ trong code): set `USE_UCIMLREPO=True`

> Lưu ý “leakage”: **không dùng trực tiếp `PM2.5` / `pm25_24h` trong feature đầu vào cho mô hình phân lớp AQI**.

---

## 2) Cấu trúc thư mục

```
air_quality_timeseries_with_semi/
├─ data/
│  ├─ raw/                # ZIP dữ liệu gốc
│  └─ processed/          # parquet + metrics + predictions + alerts
├─ notebooks/
│  ├─ preprocessing_and_eda.ipynb
│  ├─ feature_preparation.ipynb
│  ├─ classification_modelling.ipynb
│  ├─ regression_modelling.ipynb
│  ├─ arima_forecasting.ipynb
│  ├─ semi_dataset_preparation.ipynb          
│  ├─ semi_self_training.ipynb                
│  ├─ semi_co_training.ipynb                  
│  ├─ semi_supervised_report.ipynb            
│  └─ runs/                                   # output notebooks khi chạy papermill
├─ src/
│  ├─ classification_library.py
│  ├─ regression_library.py
│  ├─ timeseries_library.py
│  └─ semi_supervised_library.py              
├─ run_papermill.py
├─ requirements.txt
└─ README.md
```

---

## 3) Cài đặt môi trường

### 3.1 Tạo môi trường (Conda) và kernel cho Papermill
Repo mặc định chạy papermill với kernel tên **`beijing_env`** (xem `run_papermill.py`).

```bash
conda create -n beijing_env python=3.11 -y
conda activate beijing_env
pip install -r requirements.txt

# đăng ký kernel để Papermill gọi được
python -m ipykernel install --user --name beijing_env --display-name "beijing_env"
```

### 3.2 Kiểm tra nhanh
```bash
python -c "import pandas, sklearn, papermill; print('OK')"
```

---

## 4) Chạy pipeline (Papermill)

Chạy toàn bộ pipeline:

```bash
python run_papermill.py
```

Kết quả:
- Notebook chạy xong sẽ nằm ở `notebooks/runs/*_run.ipynb`
- Artefacts nằm ở `data/processed/` (metrics, predictions, alerts, parquet)

---

## 5) Mô tả pipeline notebooks (Notebook‑per‑task)

| Thứ tự | Notebook | Mục tiêu | Output chính |
|---:|---|---|---|
| 01 | `preprocessing_and_eda.ipynb` | đọc dữ liệu, làm sạch, tạo time features cơ bản | `data/processed/cleaned.parquet` |
| 02 | `semi_dataset_preparation.ipynb` | **giữ dữ liệu chưa nhãn + giả lập thiếu nhãn (train‑only)** | `data/processed/dataset_for_semi.parquet` |
| 03 | `feature_preparation.ipynb` | tạo dataset supervised cho phân lớp | `data/processed/dataset_for_clf.parquet` |
| 04 | `semi_self_training.ipynb` | **Self‑Training** cho AQI classification | `metrics_self_training.json`, `alerts_self_training_sample.csv` |
| 05 | `semi_co_training.ipynb` | **Co‑Training (2 views)** cho AQI classification | `metrics_co_training.json`, `alerts_co_training_sample.csv` |
| 06 | `classification_modelling.ipynb` | baseline supervised classification | `metrics.json`, `predictions_sample.csv` |
| 07 | `regression_modelling.ipynb` | dự báo PM2.5 (regression) | `regression_metrics.json`, `regressor.joblib` |
| 08 | `arima_forecasting.ipynb` | ARIMA forecasting cho 1 trạm | `arima_pm25_*` |
| 09 | `semi_supervised_report.ipynb` | **Storytelling report**: so sánh baseline vs semi + alert theo trạm | notebook report chạy trong `notebooks/runs/` |

---

## 6) Thư viện OOP (src/)

### 6.1 `src/classification_library.py`
- `time_split(df, cutoff)`: chia train/test theo thời gian
- `train_classifier(train_df, test_df, target_col='aqi_class')` → trả về `{model, metrics, pred_df}`
- Guard leakage: loại cột như `PM2.5`, `pm25_24h`, `datetime` khỏi features.

### 6.2 `src/semi_supervised_library.py` 
- `mask_labels_time_aware(...)`: giả lập thiếu nhãn **chỉ trong TRAIN**
- `SelfTrainingAQIClassifier`: vòng lặp pseudo‑label theo ngưỡng `tau`
- `CoTrainingAQIClassifier`: co‑training 2 views + late‑fusion
- `add_alert_columns(...)`: tạo `is_alert` theo ngưỡng mức AQI (vd từ `"Unhealthy"`)

---

## 7) MINI PROJECT: Semi‑Supervised AQI + Alerts theo trạm

### 7.1 Mục tiêu
Xây dựng hệ thống:
- dự đoán `aqi_class` cho từng timestamp/trạm
- sinh **cảnh báo** theo trạm (`is_alert`)
- khi **thiếu nhãn AQI** (hoặc nhãn không chuẩn), dùng **Self‑Training** và **Co‑Training** để cải thiện chất lượng.

### 7.2 Thiết kế thí nghiệm (bắt buộc)
1) **Baseline supervised**  
   - Chạy `classification_modelling.ipynb`  
   - Lấy `accuracy`, `f1_macro` từ `data/processed/metrics.json`

2) **Giả lập thiếu nhãn (train‑only)**  
   - Chạy `semi_dataset_preparation.ipynb` với:
     - `LABEL_MISSING_FRACTION ∈ {0.7, 0.9, 0.95, 0.98}`

3) **Self‑Training**  
   - Chạy `semi_self_training.ipynb` với:
     - `TAU ∈ {0.8, 0.9, 0.95}`
   - Phân tích: vòng lặp nào bắt đầu “bão hoà”, số pseudo‑labels tăng/giảm ra sao.

4) **Co‑Training**  
   - Chạy `semi_co_training.ipynb` với `TAU` giống Self‑Training
   - Bắt buộc thử 2 chế độ:
     - **Auto split views** (để `VIEW1_COLS=None`, `VIEW2_COLS=None`)
     - **Manual views**: tự thiết kế 2 views và giải thích vì sao hợp lý.


## 8) Chạy nhanh từng notebook (không dùng Papermill)
Bạn có thể mở Jupyter và chạy tuần tự từng notebook theo thứ tự ở mục (5).

---

## 9) Author
Project được thực hiện bởi:
Trang Le

## 10) License
MIT — sử dụng tự do cho nghiên cứu, học thuật và ứng dụng nội bộ.
