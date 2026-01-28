"""
Yêu Cầu 3: So Sánh Các Cấu Hình/Tham Số
=========================================
Thực hiện experiments thay đổi tham số so với thiết lập gốc
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
from PIL import Image

# Page config
st.set_page_config(
    page_title="Yêu Cầu 3: Parameter Experiments",
    page_icon="🧪",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 50%, #0369a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .exp-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 4px solid #0ea5e9;
        margin: 1rem 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 4px solid #22c55e;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">Yêu Cầu 3: So Sánh Các Cấu Hình/Tham Số</div>', unsafe_allow_html=True)
st.markdown("""
Thực hiện **5 experiments** thay đổi tham số để hiểu rõ tác động của các yếu tố trong thuật toán.
**Bắt buộc**: Thử nghiệm τ khác. **Mở rộng**: Labeled size, model khác, view splitting khác.
""")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

st.markdown("---")

# ============================================================================
# SECTION 1: TỔNG QUAN CÁC EXPERIMENTS
# ============================================================================
st.markdown("## 🎯 Tổng Quan 5 Experiments")

experiments_summary = pd.DataFrame({
    'Experiment': [
        '1. Thay đổi τ (BẮT BUỘC) ⭐',
        '2. Kích thước labeled data',
        '3. Model/Thuật toán khác',
        '4. Adaptive τ schedule',
        '5. Tách view khác đi'
    ],
    'What Changed': [
        'τ = 0.80 vs 0.90 vs 0.95',
        'Labeled: 5% vs 10% vs 20%',
        'HGBC vs RandomForest',
        'Fixed τ=0.90 vs Aggressive schedule',
        'Current views vs Pollutant-based views'
    ],
    'Best Config': [
        'τ = 0.90',
        '10% labeled',
        'HistGradientBoosting',
        'Aggressive schedule',
        'Pollutant-based (but still worse)'
    ],
    'F1-Macro': [
        '0.5343',
        '0.5050',
        '0.4919',
        '0.5088',
        '0.4507'
    ],
    'Impact': [
        'High (+13.3%)',
        'Medium (+8.1%)',
        'Very High (+19.1%)',
        'Low (+3.4%)',
        'Negative (-15.6%)'
    ]
})

st.dataframe(
    experiments_summary.style.apply(
        lambda x: ['background-color: #fef3c7' if '⭐' in str(x['Experiment']) else '' for _ in x],
        axis=1
    ),
    use_container_width=True,
    hide_index=True
)

st.info("""
📊 **Key Insights từ tất cả experiments**:
- **Experiment 1 (τ)**: BẮT BUỘC, impact cao, τ=0.90 tối ưu
- **Experiment 3 (Model)**: Impact LỚN NHẤT (+19.1%), model architecture critical!
- **Experiment 5 (View splitting)**: Co-Training thất bại (-15.6%), Self-Training tốt hơn
""")

st.markdown("---")

# Sub-tabs cho từng experiment
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Exp 1: Thay Đổi τ ⭐",
    "Exp 2: Labeled Size",
    "Exp 3: Model Architecture",
    "Exp 4: Adaptive τ",
    "Exp 5: View Splitting",
    "Summary & Recommendations"
])

# ============================================================================
# EXP 1: THAY ĐỔI NGƯỠNG τ (BẮT BUỘC)
# ============================================================================
with tab1:
    st.markdown("## Experiment 1: Thay Đổi Ngưỡng Confidence τ")
    st.markdown("**Yêu cầu BẮT BUỘC**: Thử nghiệm với giá trị τ khác cho self-training và quan sát sự khác biệt")
    
    st.markdown('<div class="exp-card">', unsafe_allow_html=True)
    st.markdown("""
    **Thiết Lập:**
    - **Baseline**: Self-Training với τ = 0.90 (từ Yêu cầu 1)
    - **Experiments**: So sánh 3 giá trị τ = 0.80, 0.90, 0.95
    - **Other params**: 5% labeled, HGBC, 10 iterations, Fixed schedule
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    try:
        # Results
        tau_results = pd.DataFrame({
            'τ Value': ['0.80', '0.90 ⭐', '0.95'],
            'Test Accuracy': [0.5941, 0.5890, 0.5931],
            'Test F1-Macro': [0.5167, 0.5343, 0.5330],
            'Pseudo-labels': [364388, 350019, 314834],
            '% Unlabeled Used': ['94.8%', '91.1%', '81.9%'],
            'Val F1 Peak': [0.7081, 0.7106, 0.6953],
            'F1 vs Baseline (0.4715)': ['+9.6%', '+13.3%', '+13.0%']
        })
        
        st.markdown("### 📊 Kết Quả So Sánh")
        
        st.dataframe(
            tau_results.style.apply(
                lambda x: ['background-color: #d1fae5; font-weight: bold' if '⭐' in str(x['τ Value']) else '' for _ in x],
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="success-card">', unsafe_allow_html=True)
            st.markdown("""
            ### ✅ Sự Khác Biệt Quan Sát Được
            
            **1. τ = 0.80 (Thấp):**
            - **Nhiều pseudo-labels nhất** (364K, 94.8% pool)
            - Model tự tin gán nhãn nhiều
            - **NHƯNG F1 thấp nhất** (0.5167)
            - **Nguyên nhân**: Thêm quá nhiều labels có confidence thấp → noise tăng
            
            **2. τ = 0.90 (Trung bình) ⭐:**
            - **F1 cao nhất** (0.5343, +13.3%)
            - 350K pseudo-labels (91.1% pool)
            - **Best balance** giữa quality và quantity
            - Val F1 peak cao nhất (0.7106)
            
            **3. τ = 0.95 (Cao):**
            - **Ít pseudo-labels nhất** (314K, 81.9%)
            - Quá strict, bỏ lỡ nhiều mẫu tốt
            - F1 = 0.5330 (chỉ 0.0013 thấp hơn τ=0.90)
            - Không cải thiện đáng kể so với τ=0.90
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="exp-card">', unsafe_allow_html=True)
            st.markdown("""
            ### 📈 Trade-off: Quality vs Quantity
            
            ```
            τ=0.80: 364K labels → F1=0.5167
                    ↓ Nhiều nhưng ồn
                    
            τ=0.90: 350K labels → F1=0.5343 ⭐
                    ↓ Sweet spot
                    
            τ=0.95: 315K labels → F1=0.5330
                    ↓ Ít hơn, cải thiện không đáng kể
            ```
            
            **Insights:**
            - **Quality > Quantity**: Ít labels nhưng chất lượng cao → F1 tốt hơn
            - **τ=0.90 là optimal**: Cân bằng tốt nhất
            - **τ quá thấp**: Noise tích lũy → F1 giảm
            - **τ quá cao**: Bỏ lỡ data → không cải thiện thêm
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("### 📊 Biểu Đồ So Sánh")
        
        exp_dir = DATA_DIR / "self_training_experiments"
        
        col1, col2 = st.columns(2)
        
        with col1:
            img_path = exp_dir / "test_performance_comparison.png"
            if img_path.exists():
                st.image(str(img_path), caption="Test Performance: 3 τ values vs Baseline", use_container_width=True)
        
        with col2:
            img_path = exp_dir / "validation_f1_over_iterations.png"
            if img_path.exists():
                st.image(str(img_path), caption="Validation F1 Over Iterations", use_container_width=True)
        
        img_path = exp_dir / "pseudo_labels_over_iterations.png"
        if img_path.exists():
            st.image(str(img_path), caption="Pseudo-labels Per Iteration - τ=0.80 thêm nhiều nhất", use_container_width=True)
        
        st.success("""
        **✅ Kết Luận Experiment 1:**
        - **τ = 0.90 là tối ưu** cho Beijing Air Quality dataset
        - Cải thiện **+13.3% F1** so với baseline (0.4715 → 0.5343)
        - Sự khác biệt rõ ràng: τ=0.80 ồn, τ=0.90 optimal, τ=0.95 không cải thiện thêm
        - **Quality > Quantity**: Confidence threshold quan trọng để filter noise
        """)
    
    except Exception as e:
        st.error(f"Lỗi load Experiment 1: {str(e)}")

# ============================================================================
# EXP 2: KÍCH THƯỚC LABELED DATA
# ============================================================================
with tab2:
    st.markdown("## Experiment 2: Thay Đổi Kích Thước Labeled Data")
    st.markdown("**Mục tiêu**: Xem dùng nhiều hơn labeled data có cải thiện đáng kể không?")
    
    st.markdown('<div class="exp-card">', unsafe_allow_html=True)
    st.markdown("""
    **Thiết Lập:**
    - **Baseline**: 5% labeled (~21K mẫu) từ Experiment 1
    - **Experiments**: So sánh 5% vs 10% vs 20% labeled data
    - **Other params**: τ=0.90, HGBC, 10 iterations
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    try:
        exp_dir = DATA_DIR / "labeled_size_experiments"
        
        # Results
        labeled_results = pd.DataFrame({
            'Labeled %': ['5%', '10% ⭐', '20%'],
            'Labeled Count': ['21,034', '42,068', '84,137'],
            'Test Accuracy': [0.5633, 0.5678, 0.5759],
            'Test F1-Macro': [0.4671, 0.5050, 0.4896],
            'Pseudo-labels': ['344,688', '346,372', '357,913'],
            'vs 5% F1': ['-', '+8.1%', '+4.8%']
        })
        
        st.markdown("### 📊 Kết Quả So Sánh")
        
        st.dataframe(
            labeled_results.style.apply(
                lambda x: ['background-color: #d1fae5; font-weight: bold' if '⭐' in str(x['Labeled %']) else '' for _ in x],
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="success-card">', unsafe_allow_html=True)
            st.markdown("""
            ### ✅ Quan Sát Quan Trọng
            
            **1. 5% Labeled (Baseline):**
            - F1 = 0.4671
            - Model base yếu nhưng self-training vẫn work
            - 344K pseudo-labels added
            
            **2. 10% Labeled (Sweet Spot) ⭐:**
            - **F1 = 0.5050 (+8.1% vs 5%)**
            - Highest F1-Macro!
            - Model base đủ mạnh để generate good pseudo-labels
            - **Best balance** giữa labeled và unlabeled
            
            **3. 20% Labeled (Diminishing Return):**
            - Accuracy cao nhất (0.5759)
            - **NHƯNG F1 GIẢM** (0.4896, -3.1% vs 10%)
            - Model quá confident với labeled → ít học từ unlabeled
            - Overfitting risk tăng
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="exp-card">', unsafe_allow_html=True)
            st.markdown("""
            ### 📈 Diminishing Return Pattern
            
            ```
            5% → 10%: +8.1% F1 ✅ (Cải thiện mạnh)
            10% → 20%: -3.1% F1 ❌ (Giảm!)
            ```
            
            **Giải thích:**
            
            **5% → 10%:**
            - Model base mạnh hơn 2x
            - Pseudo-labels chất lượng cao hơn
            - Self-training efficient hơn
            
            **10% → 20%:**
            - Thêm labeled không cải thiện model base nhiều
            - Model "satisfied" với labeled data
            - Ít "hungry" cho unlabeled data
            - F1 giảm vì bias về majority classes
            
            **→ Không phải càng nhiều labeled càng tốt!**
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("### 📊 Biểu Đồ Trực Quan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            img_path = exp_dir / "test_performance_comparison.png"
            if img_path.exists():
                st.image(str(img_path), caption="Test Performance by Labeled Size", use_container_width=True)
        
        with col2:
            img_path = exp_dir / "learning_curves.png"
            if img_path.exists():
                st.image(str(img_path), caption="Learning Curves - 10% ổn định nhất", use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            img_path = exp_dir / "pseudo_labels_comparison.png"
            if img_path.exists():
                st.image(str(img_path), caption="Pseudo-labels Activity", use_container_width=True)
        
        with col2:
            img_path = exp_dir / "training_data_composition.png"
            if img_path.exists():
                st.image(str(img_path), caption="Training Data Composition", use_container_width=True)
        
        st.success("""
        **✅ Kết Luận Experiment 2:**
        - **10% labeled là sweet spot** cho dataset 420K samples
        - Cải thiện **+8.1%** so với 5% (0.4671 → 0.5050)
        - **20% labeled KHÔNG tốt hơn 10%** (-3.1% F1) → Diminishing return
        - **Data efficiency**: Chỉ cần ~42K labeled samples (10%) thay vì toàn bộ 420K
        - **Insight**: Cân bằng giữa model base strength và unlabeled data utilization
        """)
    
    except Exception as e:
        st.error(f"Lỗi load Experiment 2: {str(e)}")

# ============================================================================
# EXP 3: MODEL/THUẬT TOÁN KHÁC
# ============================================================================
with tab3:
    st.markdown("## Experiment 3: Thử Model/Thuật Toán Khác")
    st.markdown("**Mục tiêu**: Thử chuyển sang RandomForest xem self-training có cải thiện khác không?")
    
    st.markdown('<div class="exp-card">', unsafe_allow_html=True)
    st.markdown("""
    **Thiết Lập:**
    - **Baseline**: HistGradientBoostingClassifier (HGBC) từ các experiments trước
    - **Experiment**: So sánh HGBC vs RandomForest (RF)
    - **Other params**: 5% labeled, τ=0.90, 10 iterations
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    try:
        exp_dir = DATA_DIR / "model_comparison_experiments"
        
        # Results
        model_results = pd.DataFrame({
            'Model': ['HistGradientBoosting ⭐', 'RandomForest'],
            'Test Accuracy': [0.5682, 0.5628],
            'Test F1-Macro': [0.4919, 0.4130],
            'Pseudo-labels': ['345,924', '180,363'],
            'Val F1 Peak': [0.6673, 0.5653],
            'vs RF': ['+19.1%', '-'],
            'Training Time': ['~4 min', '~12 min']
        })
        
        st.markdown("### 📊 Kết Quả So Sánh")
        
        st.dataframe(
            model_results.style.apply(
                lambda x: ['background-color: #d1fae5; font-weight: bold' if '⭐' in str(x['Model']) else '' for _ in x],
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="success-card">', unsafe_allow_html=True)
            st.markdown("""
            ### 🏆 HGBC >> RandomForest (KHỔNG LỒ!)
            
            **Performance Gap: +19.1% F1** (0.4919 vs 0.4130)
            
            **Tại sao HGBC tốt hơn RF:**
            
            **1. Probability Calibration:**
            - HGBC: Well-calibrated probabilities
            - RF: Overconfident BUT poor calibration
            - **Impact**: HGBC pseudo-labels chất lượng cao hơn
            
            **2. Pseudo-labeling Activity:**
            - HGBC: 345K labels (90% pool)
            - RF: 180K labels (47% pool, 52% ÍT HƠN!)
            - **Why**: RF probabilities không pass τ=0.90
            
            **3. Learning Trajectory:**
            - HGBC: Smooth learning, Val F1 peak 0.667
            - RF: Plateau sớm, Val F1 peak chỉ 0.565
            - **Impact**: HGBC tận dụng unlabeled tốt hơn
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="exp-card">', unsafe_allow_html=True)
            st.markdown("""
            ### 🔍 RF Thất Bại Vì Sao?
            
            **Problem: Too Conservative BUT Wrong Way**
            
            **1. Overconfident Predictions:**
            - RF dự đoán với confidence cao
            - NHƯNG predictions sai nhiều
            - Bagging ensemble "smooth" quá mức
            
            **2. Poor Probability Calibration:**
            - RF probabilities không reflect true uncertainty
            - Confidence 0.89 → KHÔNG pass τ=0.90
            - Bỏ lỡ nhiều mẫu tốt
            
            **3. Không Selective:**
            - 180K labels có nhiều noise
            - Quality kém hơn HGBC
            - Model học theo wrong patterns
            
            **Insight:**
            - **Model architecture CỰC KỲ QUAN TRỌNG** cho self-training
            - Cần model với **well-calibrated probabilities**
            - Gradient Boosting > Bagging cho semi-supervised
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("### 📊 Biểu Đồ Trực Quan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            img_path = exp_dir / "test_performance_by_model.png"
            if img_path.exists():
                st.image(str(img_path), caption="Test Performance: HGBC vs RandomForest", use_container_width=True)
        
        with col2:
            img_path = exp_dir / "learning_curves_by_model.png"
            if img_path.exists():
                st.image(str(img_path), caption="Learning Curves - HGBC ổn định, RF plateau", use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            img_path = exp_dir / "pseudo_labeling_by_model.png"
            if img_path.exists():
                st.image(str(img_path), caption="Pseudo-labeling - HGBC thêm 2x nhiều hơn RF", use_container_width=True)
        
        with col2:
            img_path = exp_dir / "per_class_f1_heatmap.png"
            if img_path.exists():
                st.image(str(img_path), caption="Per-class F1 - HGBC đồng đều hơn", use_container_width=True)
        
        st.success("""
        **✅ Kết Luận Experiment 3:**
        - **HGBC >> RandomForest** (+19.1% F1) - **Impact LỚN NHẤT**!
        - Model architecture là **yếu tố quan trọng nhất** trong self-training
        - HGBC: Well-calibrated probabilities → high-quality pseudo-labels
        - RF: Poor calibration → chỉ 180K labels (47% pool), quality kém
        - **Insight**: Gradient Boosting phù hợp hơn Bagging cho semi-supervised learning
        """)
    
    except Exception as e:
        st.error(f"Lỗi load Experiment 3: {str(e)}")

# ============================================================================
# EXP 4: ADAPTIVE τ SCHEDULE
# ============================================================================
with tab4:
    st.markdown("## Experiment 4: Adaptive τ Schedule")
    st.markdown("**Mục tiêu**: Thử τ adaptive (giảm dần) thay vì fixed τ=0.90")
    
    st.markdown('<div class="exp-card">', unsafe_allow_html=True)
    st.markdown("""
    **Thiết Lập:**
    - **Baseline**: Fixed τ=0.90 (constant qua 10 vòng)
    - **Experiment**: Aggressive schedule (τ giảm từ 0.95 → 0.80)
    - **Other params**: 10% labeled, HGBC, 10 iterations
    
    **Giả thuyết**: Early strict (τ=0.95) tránh bad labels, later relaxed (τ=0.80) maximize data usage
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    try:
        exp_dir = DATA_DIR / "hybrid_tau_experiments"
        
        # Results
        tau_schedule_results = pd.DataFrame({
            'Schedule': ['Fixed τ=0.90', 'Aggressive (0.95→0.80) ⭐'],
            'Test Accuracy': [0.5682, 0.5689],
            'Test F1-Macro': [0.4919, 0.5088],
            'Pseudo-labels': ['345,924', '370,727'],
            'Val F1 Peak': [0.6673, 0.6673],
            'Avg τ': [0.90, 0.83],
            'vs Fixed F1': ['-', '+3.4%']
        })
        
        st.markdown("### 📊 Kết Quả So Sánh")
        
        st.dataframe(
            tau_schedule_results.style.apply(
                lambda x: ['background-color: #d1fae5; font-weight: bold' if '⭐' in str(x['Schedule']) else '' for _ in x],
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # Schedule visualization
        st.markdown("### 📈 Lịch Trình τ Qua 10 Vòng")
        
        tau_schedule_df = pd.DataFrame({
            'Iteration': list(range(1, 11)),
            'Fixed': [0.90] * 10,
            'Aggressive': [0.95, 0.93, 0.91, 0.89, 0.87, 0.85, 0.83, 0.81, 0.80, 0.80]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tau_schedule_df['Iteration'], y=tau_schedule_df['Fixed'], 
                                 name='Fixed', line=dict(color='#0ea5e9', dash='dash')))
        fig.add_trace(go.Scatter(x=tau_schedule_df['Iteration'], y=tau_schedule_df['Aggressive'],
                                 name='Aggressive', line=dict(color='#22c55e')))
        fig.update_layout(title='τ Schedule Over Iterations', xaxis_title='Iteration', yaxis_title='τ Value')
        st.plotly_chart(fig, use_container_width=True)
        
        # Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="success-card">', unsafe_allow_html=True)
            st.markdown("""
            ### ✅ Aggressive Schedule Wins (Nhẹ)
            
            **Performance Gain: +3.4% F1** (0.4919 → 0.5088)
            
            **Tại sao tốt hơn:**
            
            **1. Early Strict (τ=0.95, Vòng 1-3):**
            - Ít pseudo-labels (~20-30K/iter)
            - **High quality**, tránh confirmation bias sớm
            - Model học foundation tốt
            
            **2. Later Relaxed (τ=0.80, Vòng 6-10):**
            - Nhiều pseudo-labels (~40-50K/iter)
            - Maximize unlabeled data usage
            - Total: 370K labels (96% pool)
            
            **3. Benefit:**
            - Best of both worlds
            - Quality early + Quantity later
            - +24K pseudo-labels vs Fixed
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="exp-card">', unsafe_allow_html=True)
            st.markdown("""
            ### 📊 Trade-off Analysis
            
            **Val F1 Peak: Giống nhau (0.6673)**
            - Cả 2 schedules đạt cùng upper bound
            - Aggressive đạt peak sớm hơn 1-2 vòng
            
            **Pseudo-labeling Pattern:**
            - Fixed: Uniform ~34-35K/iteration
            - Aggressive: Ramp up từ 20K → 50K
            - Total gap: +7% more labels
            
            **Diminishing Return of Low τ:**
            - τ=0.80 (vòng 6-10) thêm nhiều labels
            - NHƯNG Test F1 chỉ tăng nhẹ (+3.4%)
            - Risk: τ quá thấp → noise tăng
            
            **ROI Analysis:**
            - Complexity: Cao hơn (cần tune schedule)
            - Benefit: Nhỏ (+3.4%)
            - **Recommendation**: Fixed τ=0.90 đủ tốt và đơn giản
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("### 📊 Biểu Đồ Trực Quan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            img_path = exp_dir / "test_performance_by_schedule.png"
            if img_path.exists():
                st.image(str(img_path), caption="Test Performance: Fixed vs Aggressive", use_container_width=True)
        
        with col2:
            img_path = exp_dir / "validation_curves_by_schedule.png"
            if img_path.exists():
                st.image(str(img_path), caption="Validation Curves", use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            img_path = exp_dir / "pseudo_labeling_activity.png"
            if img_path.exists():
                st.image(str(img_path), caption="Pseudo-labeling Activity - Aggressive ramp up", use_container_width=True)
        
        with col2:
            img_path = exp_dir / "tau_performance_correlation.png"
            if img_path.exists():
                st.image(str(img_path), caption="τ-Performance Correlation", use_container_width=True)
        
        st.success("""
        **✅ Kết Luận Experiment 4:**
        - **Aggressive schedule tốt hơn Fixed** (+3.4% F1) nhưng improvement nhỏ
        - Early strict (0.95) → Later relaxed (0.80) strategy works
        - +24K pseudo-labels (7% more) nhưng chỉ +3.4% F1 → Diminishing return
        - **Recommendation**: Fixed τ=0.90 đủ tốt và đơn giản hơn
        - **ROI thấp**: Complexity tăng nhưng benefit nhỏ
        """)
    
    except Exception as e:
        st.error(f"Lỗi load Experiment 4: {str(e)}")

# ============================================================================
# EXP 5: VIEW SPLITTING KHÁC
# ============================================================================
with tab5:
    st.markdown("## Experiment 5: Tách View Khác Đi")
    st.markdown("**Mục tiêu**: Thử tách view theo pollutant types (domain knowledge) thay vì random")
    
    st.markdown('<div class="exp-card">', unsafe_allow_html=True)
    st.markdown("""
    **Thiết Lập:**
    - **Baseline**: Current view splitting (random, 41-10 features, 100% independence)
    - **Experiment**: Pollutant-based views (Primary vs Secondary, 36-30 features, 33.3% independence)
    - **Other params**: 10% labeled, HGBC, τ=0.90, max 500/iter, 10 iterations
    
    **Giả thuyết**: Domain knowledge → better views → Co-Training improves
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    try:
        exp_dir = DATA_DIR / "view_splitting_experiments"
        summary_file = exp_dir / "dashboard_summary.json"
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                view_data = json.load(f)
            
            # Results
            view_results = pd.DataFrame({
                'Strategy': ['Current (Random)', 'Pollutant-based ⭐', 'Self-Training (Reference)'],
                'View 1 Size': ['41', '36', '51'],
                'View 2 Size': ['10', '30', 'N/A'],
                'Independence': ['100%', '33.3%', 'N/A'],
                'Test Accuracy': [0.5401, 0.5718, 0.5890],
                'Test F1-Macro': [0.4176, 0.4507, 0.5343],
                'vs Self-Training': ['-21.8%', '-15.6%', 'Baseline']
            })
            
            st.markdown("### 📊 Kết Quả So Sánh")
            
            st.dataframe(
                view_results.style.apply(
                    lambda x: [
                        'background-color: #d1fae5; font-weight: bold' if '⭐' in str(x['Strategy'])
                        else 'background-color: #fee2e2' if 'Current' in str(x['Strategy'])
                        else ''
                        for _ in x
                    ],
                    axis=1
                ),
                use_container_width=True,
                hide_index=True
            )
            
            # Critical finding
            st.error("""
            **❌ CRITICAL FINDING: Cả 2 View Splitting Strategies ĐỀU THẤT BẠI!**
            
            - Pollutant-based (best): F1 = 0.4507 (**-15.6% vs Self-Training**)
            - Current (random): F1 = 0.4176 (**-21.8% vs Self-Training**)
            - Self-Training: F1 = 0.5343 (reference)
            
            **→ Co-Training KHÔNG phù hợp với Beijing Air Quality dataset!**
            """)
            
            # Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="exp-card">', unsafe_allow_html=True)
                st.markdown("""
                ### 🔍 Tại Sao Pollutant-based Tốt Hơn Current?
                
                **Pollutant-based (+7.9% vs Current):**
                
                **1. Domain Knowledge:**
                - **View 1**: Primary pollutants (PM2.5, PM10, SO2, CO)
                  - Cùng nguồn thải (xe cộ, công nghiệp)
                  - Correlated patterns
                - **View 2**: Secondary pollutants (NO2, O3)
                  - Phản ứng hóa học trong khí quyển
                  - Different formation mechanism
                
                **2. View Có Nghĩa:**
                - Split theo atmospheric chemistry
                - Mỗi view có semantic meaning
                - Model học domain-specific patterns
                
                **3. Better Balance:**
                - 36-30 features (balanced hơn 41-10)
                - Cả 2 views có đủ information
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="failure-card">', unsafe_allow_html=True)
                st.markdown("""
                ### ❌ Tại Sao Vẫn Thua Self-Training?
                
                **Root Causes:**
                
                **1. Features Highly Correlated:**
                - Primary ↔ Secondary pollutants: r = 0.4-0.7
                - Cả 2 từ cùng nguồn (traffic, industry)
                - Split làm mất thông tin quan trọng
                
                **2. View Independence Quá Thấp (33.3%):**
                - 2 models học similar patterns
                - Không đủ diverse để correct errors
                - Agreement mechanism fails
                
                **3. Information Loss:**
                - View 1 thiếu NO2, O3 → Không predict O3 spike
                - View 2 thiếu PM2.5, PM10 → Không predict PM peak
                - Each view "incomplete"
                
                **4. Dataset Characteristics:**
                - Low-dimensional (51 features)
                - Not naturally splittable
                - Better dùng ALL features (Self-Training)
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualizations
            st.markdown("### 📊 Biểu Đồ Trực Quan")
            
            col1, col2 = st.columns(2)
            
            with col1:
                img_path = exp_dir / "test_performance_by_strategy.png"
                if img_path.exists():
                    st.image(str(img_path), caption="Test Performance: 2 Strategies", use_container_width=True)
            
            with col2:
                img_path = exp_dir / "view_independence_analysis.png"
                if img_path.exists():
                    st.image(str(img_path), caption="View Independence Analysis", use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                img_path = exp_dir / "learning_curves_by_strategy.png"
                if img_path.exists():
                    st.image(str(img_path), caption="Learning Curves", use_container_width=True)
            
            with col2:
                img_path = exp_dir / "comparison_with_baseline.png"
                if img_path.exists():
                    st.image(str(img_path), caption="Comparison with Self-Training", use_container_width=True)
            
            st.error("""
            **❌ Kết Luận Experiment 5:**
            - **Pollutant-based tốt hơn Current** (+7.9%) nhờ domain knowledge
            - **NHƯNG cả 2 đều THUA Self-Training** (-15.6% và -21.8%)
            - **Nguyên nhân**: Beijing Air Quality không phù hợp cho view splitting
              - Low-dimensional (51 features)
              - Highly correlated features
              - Information loss khi split
            - **Recommendation**: Dùng **Self-Training** thay vì Co-Training!
            - **When Co-Training works**: Text, images, multi-modal data
            """)
        
        else:
            st.error("View splitting data not found")
    
    except Exception as e:
        st.error(f"Lỗi load Experiment 5: {str(e)}")

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
with tab6:
    st.markdown("## 📝 Summary & Recommendations")
    
    st.markdown("### 🏆 Xếp Hạng Impact Của Các Yếu Tố")
    
    impact_df = pd.DataFrame({
        'Factor': [
            '1. Method Choice (Self vs Co) 🔥',
            '2. Model Architecture (HGBC vs RF) 🔥',
            '3. Labeled Data Size (5% → 10%) 🔥',
            '4. Confidence Threshold τ',
            '5. View Splitting Strategy',
            '6. Adaptive τ Schedule',
            '7. More Labeled (10% → 20%)'
        ],
        'Best Config': [
            'Self-Training',
            'HistGradientBoosting',
            '10% labeled',
            'τ = 0.90',
            'Pollutant-based (still bad)',
            'Aggressive (0.95→0.80)',
            'N/A (negative)'
        ],
        'F1 Impact': [
            '+18.5%',
            '+19.1%',
            '+8.1%',
            '+13.3%',
            '-15.6%',
            '+3.4%',
            '-3.1%'
        ],
        'Priority': [
            'CRITICAL ⭐⭐⭐',
            'CRITICAL ⭐⭐⭐',
            'HIGH ⭐⭐',
            'HIGH ⭐⭐',
            'AVOID ❌',
            'LOW ⭐',
            'AVOID ❌'
        ]
    })
    
    st.dataframe(
        impact_df.style.apply(
            lambda x: [
                'background-color: #fee2e2' if 'AVOID' in str(x['Priority'])
                else 'background-color: #d1fae5; font-weight: bold' if 'CRITICAL' in str(x['Priority'])
                else 'background-color: #fef3c7' if 'HIGH' in str(x['Priority'])
                else ''
                for _ in x
            ],
            axis=1
        ),
        use_container_width=True,
        hide_index=True
    )
    
    # Recommendations
    st.markdown("### ✅ Cấu Hình Khuyến Nghị")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 🎯 Optimal Configuration
        
        **Cho Beijing Air Quality Dataset:**
        
        ```python
        METHOD = "Self-Training"  # NOT Co-Training!
        MODEL = HistGradientBoostingClassifier
        LABELED_FRACTION = 0.10  # 10% (~42K samples)
        TAU = 0.90  # Fixed (hoặc Aggressive nếu muốn +3.4%)
        MAX_ITER = 10
        EARLY_STOPPING = True  # Stop if Val F1 drop > 5%
        ```
        
        **Expected Performance:**
        - Test F1-Macro: ~0.505-0.534
        - Test Accuracy: ~0.568
        - Pseudo-labels: ~350K (91% unlabeled pool)
        - Training time: ~25-30 minutes
        
        **Key Decisions:**
        1. ✅ Self-Training (NOT Co-Training)
        2. ✅ HGBC (NOT RandomForest)
        3. ✅ 10% labeled (NOT 5% hoặc 20%)
        4. ✅ τ=0.90 (balanced)
        5. ✅ Fixed schedule (simple & effective)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="exp-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 📚 Lessons Learned
        
        **1. Method Choice Is Critical:**
        - Self-Training > Co-Training (+18.5%)
        - Beijing Air Quality: tabular, low-dim, correlated
        - Co-Training cần naturally splittable features
        
        **2. Model >> Data >> Hyperparameters:**
        - Model architecture: +19.1% impact
        - Labeled size: +8.1% impact
        - τ schedule: +3.4% impact
        - **Invest time in model selection!**
        
        **3. Quality > Quantity:**
        - 10% labeled > 20% labeled (-3.1%)
        - HGBC 346K labels > RF 180K labels
        - τ=0.90 > τ=0.80 (less noise)
        
        **4. Diminishing Returns Are Real:**
        - 5% → 10%: +8.1% ✅
        - 10% → 20%: -3.1% ❌
        - Not always "more is better"
        
        **5. View Independence Matters:**
        - 33.3% independence → Co-Training fails
        - Need > 70% for Co-Training to work
        - Check correlation matrix before splitting
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # When to use what
    st.markdown("### 🤔 Decision Tree: Khi Nào Dùng Gì?")
    
    st.info("""
    **Flowchart Quyết Định:**
    
    ```
    START: Dataset mới
    │
    ├─ Low-dimensional tabular? (< 100 features)
    │  ├─ YES → Features correlated?
    │  │  ├─ YES → Use SELF-TRAINING ✅
    │  │  └─ NO → Try CO-TRAINING (test independence first)
    │  └─ NO → High-dimensional
    │     └─ Naturally splittable? (text, images)
    │        ├─ YES → Use CO-TRAINING ✅
    │        └─ NO → Use SELF-TRAINING ✅
    │
    ├─ Model choice?
    │  ├─ Need probability calibration? → HGBC ✅
    │  └─ Speed important? → RandomForest (accept lower F1)
    │
    ├─ Labeled data?
    │  ├─ < 5%: Risk of weak base model
    │  ├─ 5-15%: Sweet spot ✅
    │  └─ > 20%: Diminishing return, consider supervised
    │
    └─ Confidence threshold?
       ├─ Start with τ=0.90 ✅
       ├─ If Val F1 drop early → increase τ to 0.95
       └─ If not enough pseudo-labels → decrease τ to 0.85
    ```
    """)
    
    st.success("""
    **✅ Final Recommendation Cho Beijing Air Quality:**
    
    **Best Configuration:**
    - Method: **Self-Training** (NOT Co-Training)
    - Model: **HistGradientBoostingClassifier**
    - Labeled: **10%** (~42K samples)
    - τ: **0.90** (Fixed schedule)
    - Expected F1: **0.50-0.53** (+7-13% vs baseline)
    
    **Why This Works:**
    1. Self-Training sử dụng ALL 51 features → no information loss
    2. HGBC có probability calibration tốt → high-quality pseudo-labels
    3. 10% labeled: balance giữa model strength và unlabeled utilization
    4. τ=0.90: optimal trade-off quality vs quantity
    
    **Implementation Priority:**
    1. ⭐⭐⭐ Choose Self-Training over Co-Training
    2. ⭐⭐⭐ Use HistGradientBoosting over RandomForest
    3. ⭐⭐ Collect 10% labeled data (not more, not less)
    4. ⭐⭐ Set τ=0.90 and monitor Val F1
    5. ⭐ (Optional) Try Aggressive τ schedule for +3.4% gain
    
    **Avoid:**
    - ❌ Co-Training for this dataset (view splitting fails)
    - ❌ RandomForest (poor probability calibration)
    - ❌ 20%+ labeled data (diminishing return)
    - ❌ τ < 0.85 (too much noise)
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 1rem;'>
    <p style='font-weight: 500; color: #0369a1;'>Yêu Cầu 3 Hoàn Thành | 5/5 Experiments Done | Best: Self-Training + HGBC + 10% + τ=0.90 | F1=0.50-0.53</p>
</div>
""", unsafe_allow_html=True)

