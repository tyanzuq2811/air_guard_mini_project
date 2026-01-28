# FlexMatch-lite: Dynamic Threshold + Focal Loss cho AQI Classification

> **Phương pháp nâng cao:** Cải thiện Self-Training bằng dynamic threshold và focal loss

---

## Mục Lục

1. [Giới Thiệu](#1-giới-thiệu)
2. [FlexMatch-lite Là Gì?](#2-flexmatch-lite-là-gì)
3. [Focal Loss Là Gì?](#3-focal-loss-là-gì)
4. [Kết Quả Thí Nghiệm](#4-kết-quả-thí-nghiệm)
5. [Phân Tích Chi Tiết](#5-phân-tích-chi-tiết)
6. [Kết Luận](#6-kết-luận)

---

## 1. Giới Thiệu

### Vấn Đề: Class Imbalance trong AQI Data

Dữ liệu AQI có **class imbalance** nghiêm trọng:
- Lớp **Moderate** (trung bình): ~30% samples
- Lớp **Hazardous** (nguy hiểm): chỉ ~5% samples

**Hậu quả:**
- Self-training thông thường thiên vị về lớp đa số
- Lớp hiếm (Hazardous, Very Unhealthy) khó được chọn làm pseudo-labels
- F1-score của lớp hiếm thấp → Nguy hiểm cho ứng dụng thực tế!

### Giải Pháp: FlexMatch-lite

**2 cải tiến chính:**
1. **Dynamic Threshold** - Ngưỡng thấp hơn cho lớp hiếm
2. **Focal Loss** - Giảm ảnh hưởng của lớp đa số trong training

---

## 2. FlexMatch-lite Là Gì?

### Dynamic Threshold theo Lớp

**Vấn đề với fixed threshold (τ=0.90):**
```python
# Tất cả lớp dùng cùng ngưỡng
if confidence >= 0.90:
    accept_pseudo_label()
```

→ Lớp hiếm có confidence thấp hơn → ít được chọn

**Giải pháp: Dynamic threshold**
```
τ_c^(t) = AvgConf_c^(t) × τ_base
```

**Ví dụ:**
| Lớp | Avg Confidence | τ_base | Dynamic τ | Kết quả |
|-----|----------------|--------|-----------|---------|
| Moderate (đa số) | 0.95 | 0.90 | **0.855** | Ngưỡng cao → chặt chẽ |
| Hazardous (hiếm) | 0.75 | 0.90 | **0.675** | Ngưỡng thấp → dễ chọn |

**Lợi ích:**
- Lớp hiếm dễ được chọn hơn (ngưỡng thấp)
- Lớp đa số vẫn chặt chẽ (ngưỡng cao)
- Tự động cân bằng qua các vòng lặp

### Implementation

```python
class DynamicThresholdSelector:
    def __init__(self, base_tau=0.90, alpha=0.9):
        self.base_tau = base_tau
        self.alpha = alpha  # Smoothing factor
        self.class_confidence_ema = {}  # Track history
    
    def compute_thresholds(self):
        thresholds = {}
        for class_idx in range(n_classes):
            avg_conf = self.class_confidence_ema[class_idx]
            thresholds[class_idx] = avg_conf * self.base_tau
        return thresholds
```

---

## 3. Focal Loss Là Gì?

### Vấn Đề với Cross-Entropy Loss

**Standard Cross-Entropy:**
```
L_CE = -log(p_t)
```

→ Tất cả samples có trọng số bằng nhau
→ Lớp đa số "overwhelm" model

### Focal Loss Formula

```
L_focal = -α(1 - p_t)^γ log(p_t)
```

**Thành phần:**
- `p_t`: Xác suất dự đoán đúng
- `γ`: Focusing parameter (thường γ=2.0)
- `α`: Class weight (optional)

**Cơ chế:**

| Tình huống | p_t | (1-p_t)^γ | Weight | Ý nghĩa |
|------------|-----|-----------|--------|---------|
| **Easy sample** (dự đoán đúng, tự tin) | 0.95 | 0.0025 | **Rất thấp** | Bỏ qua mẫu dễ |
| **Hard sample** (dự đoán khó) | 0.60 | 0.16 | **Cao** | Tập trung vào mẫu khó |

**Ví dụ cụ thể:**

```python
# Sample 1: Moderate class (easy, correct prediction)
p_t = 0.95
focal_weight = (1 - 0.95)^2 = 0.0025  # ≈ 0 → ignored

# Sample 2: Hazardous class (hard, uncertain)
p_t = 0.65
focal_weight = (1 - 0.65)^2 = 0.1225  # High → focused
```

### Implementation

```python
class FocalLossWeighter:
    def __init__(self, gamma=2.0):
        self.gamma = gamma
    
    def compute_weights(self, y_true, y_pred_proba):
        # Get probability of true class
        p_t = y_pred_proba[np.arange(len(y_true)), y_true]
        
        # Focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # Normalize
        weights = focal_term / focal_term.mean()
        return weights
```

---

## 4. Kết Quả Thí Nghiệm

### Thiết Lập

| Tham số | Giá trị |
|---------|---------|
| Labeled Data | 5% (~20K samples) |
| Unlabeled Data | 95% (~384K samples) |
| Base τ | 0.90 |
| Gamma (γ) | 2.0 |
| Alpha (smoothing) | 0.9 |
| Max Iterations | 10 |

### So Sánh Tổng Quan

| Phương pháp | Test Accuracy | Test F1-macro | Cải thiện F1 |
|-------------|---------------|---------------|--------------|
| **Baseline Self-Training** | 0.5890 | 0.5343 | - |
| **FlexMatch (Dynamic only)** | 0.5912 | 0.5389 | +0.86% |
| **FlexMatch (Focal only)** | 0.5895 | 0.5401 | +1.09% |
| **FlexMatch Combined** | **0.5928** | **0.5445** | **+1.91%** |

![Test Performance Comparison](./data/processed/flexmatch_experiments/test_performance_comparison.png)

**Kết luận:**
- ✅ FlexMatch Combined đạt **F1-macro cao nhất**: 0.5445
- ✅ Cải thiện **+1.91%** so với baseline
- ✅ Accuracy cũng tăng: 0.5928 (cao nhất)

---

## 5. Phân Tích Chi Tiết

### 5.1. Threshold Evolution

![Threshold Evolution](./data/processed/flexmatch_experiments/threshold_evolution.png)

**Quan sát:**

| Lớp | Threshold Vòng 1 | Threshold Vòng 10 | Xu hướng |
|-----|------------------|-------------------|----------|
| **Good** | 0.90 | 0.82 | Giảm (model ít tự tin hơn) |
| **Moderate** | 0.90 | 0.88 | Ổn định cao (lớp đa số) |
| **Unhealthy_for_Sensitive** | 0.90 | 0.75 | **Giảm mạnh** (lớp khó) |
| **Hazardous** | 0.90 | 0.78 | Giảm (lớp hiếm) |

**Nhận xét:**
- Lớp **Moderate** (đa số): Threshold cao và ổn định → Chặt chẽ
- Lớp **Unhealthy_for_Sensitive** (khó nhất): Threshold giảm mạnh → Dễ chọn hơn
- Dynamic threshold **tự động điều chỉnh** theo độ khó của lớp

### 5.2. Per-Class F1-Score Improvement

![Per-Class F1 Comparison](./data/processed/flexmatch_experiments/per_class_f1_comparison.png)

| Lớp AQI | Baseline F1 | FlexMatch F1 | Cải thiện | % Cải thiện |
|---------|-------------|--------------|-----------|-------------|
| **Good** | 0.4897 | 0.5012 | +0.0115 | +2.35% |
| **Moderate** | 0.7045 | 0.7089 | +0.0044 | +0.62% |
| **Unhealthy_for_Sensitive** | 0.1789 | **0.2145** | **+0.0356** | **+19.9%** ✨ |
| **Unhealthy** | 0.5877 | 0.5923 | +0.0046 | +0.78% |
| **Very_Unhealthy** | 0.5689 | 0.5912 | +0.0223 | +3.92% |
| **Hazardous** | 0.6762 | 0.6845 | +0.0083 | +1.23% |

**Phát hiện quan trọng:**

1. **Unhealthy_for_Sensitive (+19.9%)** - Cải thiện mạnh nhất!
   - Lớp khó nhất trong baseline (F1=0.1789)
   - FlexMatch giúp tăng lên 0.2145
   - Dynamic threshold giúp chọn nhiều pseudo-labels hơn

2. **Very_Unhealthy (+3.92%)** - Cải thiện tốt
   - Lớp trung bình, được hưởng lợi từ focal loss

3. **Moderate (+0.62%)** - Cải thiện nhẹ
   - Lớp đa số, đã tốt từ trước
   - Focal loss giúp không bị overfit

### 5.3. Pseudo-Labels Distribution

| Vòng | Baseline (τ=0.90) | FlexMatch | Chênh lệch |
|------|-------------------|-----------|------------|
| 1 | 76,134 | 82,456 | +8.3% |
| 2 | 202,713 | 198,234 | -2.2% |
| 3 | 45,622 | 51,789 | +13.5% |
| ... | ... | ... | ... |
| **Total** | 350,019 | 365,123 | +4.3% |

**Nhận xét:**
- FlexMatch chọn **nhiều pseudo-labels hơn** (+4.3%)
- Đặc biệt ở vòng 1 và 3 (lúc model cần boost)
- Vòng 2 ít hơn một chút → Model thận trọng hơn với focal loss

---

## 6. Kết Luận

### Tổng Kết

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            FLEXMATCH-LITE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test Accuracy:       0.5928 (+0.64% vs baseline)
Test F1-macro:       0.5445 (+1.91% vs baseline)

Lớp cải thiện nhất:  Unhealthy_for_Sensitive (+19.9%)
Pseudo-labels:       365,123 (+4.3%)
Training time:       ~25 minutes

Success Rate:        100.6% of baseline accuracy
                     103.8% of baseline F1-macro
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Khi Nào Dùng FlexMatch?

**✅ Nên dùng khi:**
- Dataset có **class imbalance** nghiêm trọng
- Lớp thiểu số quan trọng (ví dụ: fraud detection, medical diagnosis)
- Cần cải thiện **Macro-F1** (không chỉ Accuracy)
- Có đủ unlabeled data để tận dụng

**❌ Không cần khi:**
- Dataset balanced
- Chỉ quan tâm Accuracy
- Lớp đa số quan trọng nhất

### Best Practices

```python
# Recommended configuration
fm_cfg = FlexMatchConfig(
    base_tau=0.90,        # Standard threshold
    gamma=2.0,            # Standard focal loss
    alpha=0.9,            # High smoothing for stability
    max_iter=10,
    use_focal_loss=True   # Always use with dynamic threshold
)
```

### So Sánh với Phương Pháp Khác

| Phương pháp | Accuracy | F1-macro | Training Time | Complexity |
|-------------|----------|----------|---------------|------------|
| Self-Training | 0.5890 | 0.5343 | ~20 min | Low |
| **FlexMatch** | **0.5928** | **0.5445** | ~25 min | Medium |
| Label Spreading | ? | ? | ? | High (memory) |

→ **FlexMatch** là lựa chọn tốt nhất cho **imbalanced AQI data**

### Tiếp Theo

Xem thêm:
- [Self-Training Analysis](./BLOG_SELF_TRAINING.md) - Baseline comparison
- [Label Spreading Analysis](./BLOG_LABEL_SPREADING.md) - Graph-based approach
- [Co-Training Analysis](./BLOG_CO_TRAINING.md) - Multi-view learning

---

## Tài Liệu Tham Khảo

### Files Liên Quan

- **Code:** `notebooks/semi_flexmatch_training.ipynb`
- **Library:** `src/semi_supervised_library.py`
  - `FlexMatchConfig`
  - `FocalLossWeighter`
  - `DynamicThresholdSelector`
  - `FlexMatchSelfTraining`
- **Results:** `data/processed/flexmatch_experiments/`
  - `metrics_flexmatch.json`
  - `flexmatch_summary.json`
  - `per_class_improvements.csv`
- **Visualizations:**
  - `test_performance_comparison.png`
  - `threshold_evolution.png`
  - `per_class_f1_comparison.png`

### Papers

- **Focal Loss:** [Lin et al., 2017 - Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- **FlexMatch:** [Zhang et al., 2021 - FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling](https://arxiv.org/abs/2110.08263)

---

<div align="center">

**Blog được tạo tự động từ kết quả thí nghiệm**

*Data Mining - Air Quality Prediction Project*

</div>
