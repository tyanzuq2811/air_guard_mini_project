"""
Beijing Air Quality Analysis Dashboard
=======================================
Dashboard phân tích chất lượng không khí Beijing với Semi-Supervised Learning
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Beijing Air Quality Dashboard",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Ocean Blue Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 50%, #0369a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0369a1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 2rem;
        border-radius: 1rem;
        border-left: 4px solid #0ea5e9;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(14, 165, 233, 0.1);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(14, 165, 233, 0.15);
    }
    
    .requirement-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px solid #e0f2fe;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .requirement-card:hover {
        border-color: #0ea5e9;
        box-shadow: 0 10px 15px -3px rgba(14, 165, 233, 0.1);
    }
    
    .requirement-number {
        display: inline-block;
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        color: white;
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 50%;
        text-align: center;
        line-height: 2.5rem;
        font-weight: 700;
        font-size: 1.2rem;
        margin-right: 1rem;
    }
    
    .highlight-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #0369a1;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">� Beijing Air Quality Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Phân Tích Chất Lượng Không Khí với Machine Learning & Semi-Supervised Learning</p>', unsafe_allow_html=True)

st.markdown("---")

# Project Overview
st.markdown('<h2 class="section-title">🎯 Tổng Quan Dự Án</h2>', unsafe_allow_html=True)

st.markdown("""
Dashboard này tổng hợp toàn bộ kết quả từ Mini Project **Beijing Air Quality Analysis**, 
bao gồm việc ứng dụng các thuật toán **Semi-Supervised Learning** (Self-Training và Co-Training) 
để phân loại chất lượng không khí với dữ liệu có nhãn hạn chế.

**Sử dụng sidebar bên trái** để điều hướng giữa các yêu cầu.
""")

# Key Statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="⏱️ Thời Gian",
        value="2013-2017",
        delta="5 năm dữ liệu",
        help="Dữ liệu từ 12 trạm quan trắc tại Beijing"
    )

with col2:
    st.metric(
        label="📍 Số Trạm",
        value="12 trạm",
        help="12 trạm quan trắc khí quyển tại Beijing"
    )

with col3:
    st.metric(
        label="💾 Tổng Mẫu",
        value="420,768",
        help="Số lượng records sau làm sạch"
    )

with col4:
    st.metric(
        label="🏆 Best F1-Macro",
        value="0.5343",
        delta="+13.3%",
        help="Self-Training vs Supervised Baseline"
    )

st.markdown("---")

# Dataset Information
st.markdown('<h2 class="section-title">� Dataset Information</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ### 🗂️ Nguồn Dữ Liệu
    
    - **Dataset**: Beijing Multi-Site Air Quality Data (2013-2017)
    - **Nguồn**: UCI Machine Learning Repository
    - **Đặc trưng**: 
      - **Air Pollutants**: PM2.5, PM10, SO2, NO2, CO, O3
      - **Meteorological**: Nhiệt độ, độ ẩm, áp suất, gió, mưa
      - **Temporal**: Hour, day, month, season
      - **Spatial**: 12 station locations
    - **Target**: 6 lớp AQI (Good, Moderate, Unhealthy, Very Unhealthy, Hazardous)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ### 📊 Kết Quả Chính
    
    - **Self-Training**: F1-macro **0.5343** (+13.3% vs baseline)
      - Best τ: 0.90
      - Optimal labeled: 10%
      - Pseudo-labels: 350K (91% unlabeled)
    
    - **Co-Training**: F1-macro **0.4507** (-15.6% vs Self-Training)
      - View independence: 33.3%
      - Conclusion: KHÔNG phù hợp cho dataset này
    
    - **Recommendation**: Dùng **Self-Training** với HGBC
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Requirements Overview
st.markdown('<h2 class="section-title">� Cấu Trúc Dự Án (3 Yêu Cầu)</h2>', unsafe_allow_html=True)

# Requirement 1
st.markdown('<div class="requirement-card">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown('<div class="requirement-number">1</div>', unsafe_allow_html=True)
with col2:
    st.markdown("""
    ### � Self-Training Algorithm
    
    Huấn luyện thuật toán Self-training với mô hình baseline trên dữ liệu không nhãn.
    
    **Nội dung:**
    - ✅ So sánh **3 ngưỡng τ** (0.80, 0.90, 0.95) - **Yêu cầu bắt buộc**
    - ✅ Diễn biến qua **10 vòng lặp** với bảng + biểu đồ
    - ✅ Phân tích: Model tự tin gán nhãn lúc nào? Xu hướng tăng/giảm?
    - ✅ Val F1 giảm ở vòng nào? Nguyên nhân? (Confirmation bias)
    - ✅ So sánh với **Supervised Baseline**: F1 cải thiện **+13.3%**
    - ✅ Per-class performance: Lớp nào hưởng lợi? (Unhealthy_for_Sensitive +50%)
    
    **Kết luận**: τ=0.90 tối ưu, nên dừng ở vòng 5-6 (early stopping)
    """)
st.markdown('</div>', unsafe_allow_html=True)

# Requirement 2
st.markdown('<div class="requirement-card">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown('<div class="requirement-number">2</div>', unsafe_allow_html=True)
with col2:
    st.markdown("""
    ### 🔀 Co-Training Algorithm
    
    Huấn luyện Co-training với **2 models** trên **2 views đặc trưng** khác nhau.
    
    **Nội dung:**
    - ✅ Mô tả **2 views**: 
      - View 1: Primary pollutants (PM2.5, PM10, SO2, CO) + Meteorological (36 features)
      - View 2: Secondary pollutants (NO2, O3) + Station info (30 features)
    - ✅ View independence: **33.3%** (quá thấp!)
    - ✅ Thiết lập: τ=0.90 cho cả 2 models, max 500 pseudo/iter
    - ✅ Diễn biến: 2 models có cải thiện **song song** không?
      - **KHÔNG**: Cả 2 đều degrading (-15% Val F1)
    - ✅ So sánh: Co-Training **-15.6%** vs Self-Training
    
    **Kết luận**: Co-Training THẤT BẠI. Nguyên nhân:
    - View không đủ độc lập (33.3%)
    - Features highly correlated
    - Information loss khi split
    - **→ Dùng Self-Training!**
    """)
st.markdown('</div>', unsafe_allow_html=True)

# Requirement 3
st.markdown('<div class="requirement-card">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown('<div class="requirement-number">3</div>', unsafe_allow_html=True)
with col2:
    st.markdown("""
    ### 🧪 Parameter Comparison Experiments
    
    Thực hiện **5 experiments** thay đổi tham số để hiểu tác động của các yếu tố.
    
    **Experiments:**
    1. **Thay đổi τ** (BẮT BUỘC): τ=0.90 tối ưu (+13.3%)
    2. **Kích thước labeled**: 10% sweet spot, 20% diminishing return (-3.1%)
    3. **Model khác**: HGBC >> RandomForest (+19.1% - Impact LỚN NHẤT!)
    4. **Adaptive τ schedule**: Aggressive tốt hơn Fixed (+3.4%, ROI thấp)
    5. **View splitting khác**: Pollutant-based > Current (+7.9%) nhưng vẫn thua Self-Training (-15.6%)
    
    **Ranking Impact:**
    - 🔥🔥🔥 Model Architecture (+19.1%)
    - 🔥🔥🔥 Method Choice (Self vs Co, +18.5%)
    - 🔥🔥 Labeled Size (+8.1%)
    - 🔥🔥 Confidence τ (+13.3%)
    - 🔥 Adaptive τ (+3.4%)
    """)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Key Findings
st.markdown('<h2 class="section-title">� Phát Hiện Chính</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.markdown("""
    ### ✅ Self-Training Wins
    
    - **F1-Macro**: 0.5343
    - **Improvement**: +13.3% vs baseline
    - **τ optimal**: 0.90
    - **Labeled**: Chỉ cần 10%
    - **Pseudo-labels**: 350K (91% pool)
    - **Best for**: Low-dim tabular data
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.markdown("""
    ### ❌ Co-Training Fails
    
    - **F1-Macro**: 0.4507
    - **vs Self-Training**: -15.6%
    - **Independence**: 33.3% (quá thấp)
    - **Views**: Không đủ độc lập
    - **Nguyên nhân**: Features correlated
    - **Conclusion**: Không phù hợp
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Best Config
    
    - **Method**: Self-Training
    - **Model**: HistGradientBoosting
    - **Labeled**: 10% (~42K)
    - **τ**: 0.90 (Fixed)
    - **Iterations**: 10
    - **Early Stopping**: Vòng 5-6
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Navigation Guide
st.markdown('<h2 class="section-title">🧭 Hướng Dẫn Sử Dụng</h2>', unsafe_allow_html=True)

st.info("""
**Cách điều hướng Dashboard:**

1. **Sử dụng sidebar bên trái** để chuyển giữa các trang:
   - � **Self-Training**: Yêu cầu 1 - So sánh τ, diễn biến 10 vòng, per-class analysis
   - 🔄 **Co-Training**: Yêu cầu 2 - 2 views, diễn biến 2 models, phân tích thất bại
   - 🧪 **Parameter Experiments**: Yêu cầu 3 - 5 experiments với tham số khác nhau

2. **Mỗi trang có**:
   - Metrics cards với số liệu quan trọng
   - Bảng so sánh chi tiết
   - Biểu đồ trực quan (từ experiments đã chạy)
   - Phân tích, nhận xét, kết luận

3. **Tải dữ liệu**:
   - Dashboard tự động load từ `data/processed/`
   - Nếu thiếu file: Chạy lại notebooks tương ứng
""")

st.markdown("---")

# Architecture Overview
st.markdown('<h2 class="section-title">🏛️ Kiến Trúc Dự Án</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ### 📁 Data
    
    - `data/raw/`: Dữ liệu gốc
    - `data/processed/`: Kết quả experiments
      - `self_training_experiments/`
      - `view_splitting_experiments/`
      - `labeled_size_experiments/`
      - `model_comparison_experiments/`
      - `hybrid_tau_experiments/`
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ### 📓 Notebooks
    
    - `preprocessing_and_eda.ipynb`
    - `classification_modelling.ipynb`
    - `regression_modelling.ipynb`
    - `arima_forecasting.ipynb`
    - `semi_supervised_*.ipynb`
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ### 📊 Dashboard
    
    - `app.py`: Landing page (trang này)
    - `pages/1_Self_Training.py`
    - `pages/2_Co_Training.py`
    - `pages/3_Parameter_Experiments.py`
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 1rem; margin-top: 2rem;'>
    <p style='font-size: 1.1rem; font-weight: 600; color: #0369a1; margin-bottom: 0.5rem;'>
        Beijing Air Quality Analysis Dashboard
    </p>
    <p style='margin-bottom: 0.5rem;'>
        Data Mining Mini Project | Beijing Multi-Site Air Quality (2013-2017)
    </p>
    <p style='font-size: 0.9rem;'>
        🎓 Semi-Supervised Learning | Self-Training & Co-Training
    </p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>
        📚 Documentation: <a href='../BLOG_PARAMETER_COMPARISON.md' style='color: #0ea5e9; text-decoration: none;'>BLOG_PARAMETER_COMPARISON.md</a>
    </p>
</div>
""", unsafe_allow_html=True)
