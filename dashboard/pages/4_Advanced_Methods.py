"""
Yêu Cầu Nâng Cao: Advanced Semi-Supervised Methods
==================================================
FlexMatch-lite và Label Spreading - Giải quyết class imbalance và confirmation bias
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
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Page config
st.set_page_config(
    page_title="Advanced Semi-Supervised Methods",
    page_icon="🚀",
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
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 50%, #991b1b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .advanced-card {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        padding: 2rem;
        border-radius: 1rem;
        border-left: 4px solid #dc2626;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .method-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 4px solid #16a34a;
        margin: 1rem 0;
    }
    
    .innovation-tag {
        background: linear-gradient(135deg, #dc2626, #b91c1c);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    .key-insight {
        background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #f59e0b;
        margin: 0.5rem 0;
    }
    
    .comparison-table {
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="main-title">🚀 Advanced Semi-Supervised Methods</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="advanced-card">
    <h3>🎯 Mục tiêu nâng cao</h3>
    <p>Phát triển 2 phương pháp semi-supervised tiên tiến để giải quyết các thách thức của phương pháp truyền thống:</p>
    <ul>
        <li><strong>FlexMatch-lite:</strong> Dynamic threshold + Focal loss → Xử lý class imbalance</li>
        <li><strong>Label Spreading:</strong> Graph-based propagation → Tránh confirmation bias</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_advanced_results():
    """Load advanced methods results"""
    try:
        results_dir = Path(__file__).parent.parent / "data" / "processed" / "advanced_semi_results"
        
        # Load FlexMatch results
        flexmatch_file = results_dir / "flexmatch_results.json"
        if flexmatch_file.exists():
            with open(flexmatch_file, 'r') as f:
                flexmatch_results = json.load(f)
        else:
            flexmatch_results = None
            
        # Load Label Spreading results
        label_spreading_file = results_dir / "label_spreading_results.json"
        if label_spreading_file.exists():
            with open(label_spreading_file, 'r') as f:
                label_spreading_results = json.load(f)
        else:
            label_spreading_results = None
            
        # Load comparison
        comparison_file = results_dir / "method_comparison.csv"
        if comparison_file.exists():
            comparison_df = pd.read_csv(comparison_file)
        else:
            comparison_df = None
            
        return flexmatch_results, label_spreading_results, comparison_df
        
    except Exception as e:
        st.error(f"Error loading advanced results: {e}")
        return None, None, None

# Load results
flexmatch_results, label_spreading_results, comparison_df = load_advanced_results()

if flexmatch_results is None and label_spreading_results is None:
    st.markdown("""
    <div class="advanced-card">
        <h3>⚠️ Chưa có kết quả</h3>
        <p>Vui lòng chạy notebook <code>advanced_semi_supervised.ipynb</code> để tạo kết quả.</p>
        <p>Notebook này sẽ:</p>
        <ol>
            <li>Train FlexMatch-lite với dynamic thresholds</li>
            <li>Train Label Spreading với graph-based propagation</li>
            <li>So sánh với các phương pháp baseline</li>
            <li>Lưu kết quả vào <code>data/processed/advanced_semi_results/</code></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Show methodology anyway
    st.markdown("---")
    st.markdown("## 📚 Methodology Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="method-card">
            <h4>🔄 FlexMatch-lite</h4>
            <span class="innovation-tag">DYNAMIC THRESHOLDS</span>
            <ul>
                <li><strong>Adaptive thresholds:</strong> τc = AvgConfc × τbase</li>
                <li><strong>Focal loss:</strong> LFocal = -α(1-pt)γ log(pt)</li>
                <li><strong>Class-aware:</strong> Lower threshold for rare classes</li>
                <li><strong>Warmup period:</strong> Fixed threshold first 3 iterations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="method-card">
            <h4>🌐 Label Spreading</h4>
            <span class="innovation-tag">GRAPH-BASED</span>
            <ul>
                <li><strong>Global structure:</strong> RBF/KNN kernel similarity</li>
                <li><strong>Smooth propagation:</strong> α regularization parameter</li>
                <li><strong>No confirmation bias:</strong> One-shot global optimization</li>
                <li><strong>Natural balance:</strong> Neighbor-weighted propagation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.stop()

# Main dashboard with results
tab1, tab2, tab3, tab4 = st.tabs(["🔄 FlexMatch Analysis", "🌐 Label Spreading", "📊 Comprehensive Comparison", "🎯 Key Insights"])

with tab1:
    st.markdown("## 🔄 FlexMatch-lite: Dynamic Threshold Analysis")
    
    if flexmatch_results:
        # Configuration display
        config = flexmatch_results['config']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Base Threshold (τbase)", f"{config['tau_base']:.2f}")
        with col2:
            st.metric("Max Iterations", config['max_iter'])
        with col3:
            st.metric("Focal Alpha", f"{config['focal_alpha']:.2f}")
        with col4:
            st.metric("Focal Gamma", f"{config['focal_gamma']:.1f}")
        
        # Performance metrics
        test_metrics = flexmatch_results['test_metrics']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Test Accuracy", f"{test_metrics['accuracy']:.4f}")
        with col2:
            st.metric("F1-Macro Score", f"{test_metrics['f1_macro']:.4f}")
        with col3:
            st.metric("Test Samples", f"{test_metrics['n_test']:,}")
        
        # Training history
        if 'history' in flexmatch_results and flexmatch_results['history']:
            history_df = pd.DataFrame(flexmatch_results['history'])
            
            # Dynamic thresholds visualization
            st.markdown("### 📈 Dynamic Threshold Evolution")
            
            threshold_data = []
            for _, row in history_df.iterrows():
                if 'class_thresholds' in row and isinstance(row['class_thresholds'], dict):
                    for cls, threshold in row['class_thresholds'].items():
                        threshold_data.append({
                            'Iteration': row['iter'],
                            'Class': cls,
                            'Threshold': threshold
                        })
            
            if threshold_data:
                threshold_df = pd.DataFrame(threshold_data)
                
                fig = px.line(threshold_df, x='Iteration', y='Threshold', 
                             color='Class', title='Class-wise Dynamic Thresholds',
                             markers=True)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div class="key-insight">
                    <strong>💡 Key Insight:</strong> Dynamic thresholds adapt to each class's confidence level.
                    Rare classes (Hazardous, Very_Unhealthy) get lower thresholds to improve recall.
                </div>
                """, unsafe_allow_html=True)
            
            # Training progress
            st.markdown("### 📊 Training Progress")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=history_df['iter'], y=history_df['val_accuracy'],
                                       mode='lines+markers', name='Accuracy', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=history_df['iter'], y=history_df['val_f1_macro'],
                                       mode='lines+markers', name='F1-Macro', line=dict(color='red')))
                fig.update_layout(title='Validation Performance', height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(history_df, x='iter', y='new_pseudo',
                           title='Pseudo Labels Added per Iteration')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("## 🌐 Label Spreading: Graph-based Analysis")
    
    if label_spreading_results:
        # Configuration display
        config = label_spreading_results['config']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Kernel Type", config['kernel'].upper())
        with col2:
            st.metric("Gamma (RBF)", config['gamma'])
        with col3:
            st.metric("Alpha (Regularization)", f"{config['alpha']:.1f}")
        with col4:
            st.metric("Neighbors (KNN)", config['n_neighbors'])
        
        # Performance metrics
        test_metrics = label_spreading_results['test_metrics']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Test Accuracy", f"{test_metrics['accuracy']:.4f}")
        with col2:
            st.metric("F1-Macro Score", f"{test_metrics['f1_macro']:.4f}")
        with col3:
            st.metric("Test Samples", f"{test_metrics['n_test']:,}")
        
        # Label propagation info
        if 'history' in label_spreading_results and label_spreading_results['history']:
            history = label_spreading_results['history'][0]
            
            st.markdown("### 🔄 Label Propagation Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Originally Unlabeled", f"{history['originally_unlabeled']:,}")
            with col2:
                st.metric("Labels Propagated", f"{history['labels_propagated']:,}")
            
            st.markdown("""
            <div class="key-insight">
                <strong>💡 Key Insight:</strong> Label Spreading propagates labels through the entire
                unlabeled dataset in one shot using graph structure, avoiding iterative confirmation bias.
            </div>
            """, unsafe_allow_html=True)
        
        # Methodology explanation
        st.markdown("### 🧮 Graph-based Methodology")
        st.markdown("""
        <div class="method-card">
            <h4>Label Spreading Algorithm Steps:</h4>
            <ol>
                <li><strong>Graph Construction:</strong> Build similarity graph using RBF kernel</li>
                <li><strong>Label Matrix Setup:</strong> Initialize with labeled samples (-1 for unlabeled)</li>
                <li><strong>Propagation:</strong> Iterate until convergence using transition matrix</li>
                <li><strong>Regularization:</strong> Balance between labeled constraints and smoothness</li>
            </ol>
            
            <p><strong>Formula:</strong> F = α·S·F + (1-α)·Y</p>
            <ul>
                <li>F = predicted labels</li>
                <li>S = normalized similarity matrix</li>
                <li>Y = initial labeled data</li>
                <li>α = regularization parameter (0.2)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("## 📊 Comprehensive Method Comparison")
    
    if comparison_df is not None:
        # Performance comparison table
        st.markdown("### 🏆 Performance Ranking")
        
        # Style the dataframe
        styled_df = comparison_df.style.format({
            'Accuracy': '{:.4f}',
            'F1-Macro': '{:.4f}'
        }).background_gradient(subset=['Accuracy', 'F1-Macro'], cmap='RdYlGn')
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Visual comparison
        st.markdown("### 📈 Visual Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(comparison_df, x='Method', y='Accuracy',
                        color='Type', title='Test Accuracy Comparison',
                        color_discrete_map={
                            'Advanced Semi-Supervised': '#dc2626',
                            'Basic Semi-Supervised': '#2563eb',
                            'Supervised': '#16a34a'
                        })
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(comparison_df, x='Method', y='F1-Macro',
                        color='Type', title='F1-Macro Score Comparison',
                        color_discrete_map={
                            'Advanced Semi-Supervised': '#dc2626',
                            'Basic Semi-Supervised': '#2563eb',
                            'Supervised': '#16a34a'
                        })
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Improvement analysis
        advanced_methods = comparison_df[comparison_df['Type'] == 'Advanced Semi-Supervised']
        basic_methods = comparison_df[comparison_df['Type'] == 'Basic Semi-Supervised']
        
        if len(advanced_methods) > 0 and len(basic_methods) > 0:
            best_advanced = advanced_methods.loc[advanced_methods['F1-Macro'].idxmax()]
            best_basic = basic_methods.loc[basic_methods['F1-Macro'].idxmax()]
            
            improvement = best_advanced['F1-Macro'] - best_basic['F1-Macro']
            
            st.markdown(f"""
            <div class="advanced-card">
                <h4>🚀 Advanced Methods Improvement</h4>
                <p><strong>Best Advanced:</strong> {best_advanced['Method']} (F1-Macro: {best_advanced['F1-Macro']:.4f})</p>
                <p><strong>Best Basic:</strong> {best_basic['Method']} (F1-Macro: {best_basic['F1-Macro']:.4f})</p>
                <p><strong>Improvement:</strong> +{improvement:.4f} ({improvement/best_basic['F1-Macro']*100:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)

with tab4:
    st.markdown("## 🎯 Key Insights & Analysis")
    
    # Theoretical advantages
    st.markdown("### 🧠 Theoretical Advantages")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="method-card">
            <h4>🔄 FlexMatch-lite Advantages</h4>
            <ul>
                <li><strong>Class Imbalance:</strong> Dynamic thresholds for rare classes</li>
                <li><strong>Focal Loss:</strong> Reduces easy example dominance</li>
                <li><strong>Adaptive Learning:</strong> Confidence-based threshold adjustment</li>
                <li><strong>Bias Correction:</strong> Lower thresholds for Unhealthy+ classes</li>
                <li><strong>Gradual Learning:</strong> Iterative confidence building</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="method-card">
            <h4>🌐 Label Spreading Advantages</h4>
            <ul>
                <li><strong>Global Structure:</strong> Uses entire dataset similarity</li>
                <li><strong>No Confirmation Bias:</strong> One-shot global optimization</li>
                <li><strong>Natural Smoothness:</strong> Perfect for time-series data</li>
                <li><strong>Neighbor Weighting:</strong> Automatic class balance</li>
                <li><strong>Computational Efficiency:</strong> No iterative pseudo-labeling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Problem-specific insights
    st.markdown("### 🏭 Air Quality Specific Benefits")
    
    st.markdown("""
    <div class="advanced-card">
        <h4>Why These Methods Work for AQI Prediction</h4>
        
        <h5>🔄 FlexMatch-lite for AQI:</h5>
        <ul>
            <li><strong>Rare Event Focus:</strong> Very_Unhealthy and Hazardous AQI are rare but critical</li>
            <li><strong>Dynamic Adaptation:</strong> Threshold adapts to seasonal air quality patterns</li>
            <li><strong>Confidence Building:</strong> Gradually learns from easy to hard examples</li>
        </ul>
        
        <h5>🌐 Label Spreading for AQI:</h5>
        <ul>
            <li><strong>Temporal Smoothness:</strong> Air quality changes gradually over time</li>
            <li><strong>Spatial Correlation:</strong> Nearby stations have similar patterns</li>
            <li><strong>Feature Similarity:</strong> Weather conditions create natural clusters</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Practical implications
    st.markdown("### 🎯 Practical Implications")
    
    st.markdown("""
    <div class="key-insight">
        <h4>🚨 For Real-world AQI Monitoring:</h4>
        <ol>
            <li><strong>Early Warning:</strong> Better detection of severe pollution events</li>
            <li><strong>Resource Allocation:</strong> More accurate predictions for intervention planning</li>
            <li><strong>Public Health:</strong> Improved recall for dangerous AQI levels</li>
            <li><strong>Cost Efficiency:</strong> Leverage unlabeled data from monitoring stations</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Future improvements
    st.markdown("### 🔮 Future Improvements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="method-card">
            <h4>🔧 Technical Enhancements</h4>
            <ul>
                <li>Ensemble FlexMatch + Label Spreading</li>
                <li>Multi-view Label Spreading</li>
                <li>Temperature scaling for calibration</li>
                <li>Active learning integration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="method-card">
            <h4>🌍 Domain Extensions</h4>
            <ul>
                <li>Multi-city transfer learning</li>
                <li>Meteorological data integration</li>
                <li>Real-time streaming adaptation</li>
                <li>Uncertainty quantification</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; margin-top: 2rem;">
    <p>🚀 <strong>Advanced Semi-Supervised Learning for Air Quality Prediction</strong></p>
    <p>FlexMatch-lite & Label Spreading Implementation</p>
    <p><em>Pushing the boundaries of semi-supervised learning</em></p>
</div>
""", unsafe_allow_html=True)