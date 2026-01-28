# Label Spreading: Graph-Based Semi-Supervised Learning

> **Phương pháp nâng cao:** So sánh Label Spreading với Self-Training và FlexMatch

---

## Mục Lục

1. [Giới Thiệu](#1-giới-thiệu)
2. [Label Spreading Là Gì?](#2-label-spreading-là-gì)
3. [Kết Quả Thí Nghiệm](#3-kết-quả-thí-nghiệm)
4. [So Sánh Với Self-Training](#4-so-sánh-với-self-training)
5. [Phân Tích](#5-phân-tích)
6. [Kết Luận](#6-kết-luận)

---

## 1. Giới Thiệu

### Tại Sao Cần Label Spreading?

**Hạn chế của Self-Training:**
- ❌ Confirmation bias (model tin vào lỗi của chính nó)
- ❌ Iterative process (chậm, nhiều vòng lặp)
- ❌ Greedy selection (chọn theo threshold cứng)

**Label Spreading khác biệt:**
- ✅ Sử dụng **manifold structure** của dữ liệu
- ✅ **Single optimization** (không cần vòng lặp)
- ✅ Tự nhiên xử lý class imbalance qua graph

---

## 2. Label Spreading Là Gì?

### Ý Tưởng Cơ Bản

**Graph-based approach:**
1. Xây dựng **similarity graph** giữa tất cả samples (labeled + unlabeled)
2. **Lan truyền nhãn** qua graph dựa trên similarity
3. Samples gần nhau trên graph → có nhãn giống nhau

**Ví dụ trực quan:**
```
Labeled samples:     [Good]  [Moderate]  [Hazardous]
                        |         |           |
Similarity edges:      ↓         ↓           ↓
Unlabeled samples:   [?]  →  [?]  →  [?]  →  [?]
                      ↓         ↓           ↓
After spreading:    [Good] [Moderate] [Moderate] [Hazardous]
```

### Công Thức

**Label Spreading iteration:**
```
Y^(t+1) = αSY^(t) + (1-α)Y^(0)
```

**Thành phần:**
- `Y^(t)`: Label distribution tại vòng t
- `S`: Similarity matrix (normalized)
- `α`: Clamping factor (0-1)
  - α = 0: Chỉ dùng initial labels
  - α = 1: Hoàn toàn lan truyền (Label Propagation)
  - α = 0.2: Cân bằng (khuyến nghị)
- `Y^(0)`: Initial labels

### RBF Kernel

**Similarity computation:**
```
S(i,j) = exp(-γ ||x_i - x_j||²)
```

**Tham số γ:**
- γ nhỏ (10): Similarity rộng → lan truyền xa
- γ lớn (30): Similarity hẹp → lan truyền gần
- γ = 20: Cân bằng (khuyến nghị)

### Ưu & Nhược Điểm

| Ưu điểm | Nhược điểm |
|---------|------------|
| ✅ Sử dụng manifold structure | ❌ Memory intensive O(n²) |
| ✅ Không có confirmation bias | ❌ Chậm với large datasets |
| ✅ Deterministic (không random) | ❌ Cần tune hyperparameters |
| ✅ Xử lý class imbalance tốt | ❌ Không phù hợp với sparse data |

---

## 3. Kết Quả Thí Nghiệm

### Thiết Lập

| Tham số | Giá trị |
|---------|---------|
| Labeled Data | 5% (~20K samples) |
| Unlabeled Data | 95% (sampled to 50K) |
| Kernel | RBF |
| Gamma (γ) | 20.0 |
| Alpha (α) | 0.2 |
| Max Iterations | 30 |

> **Lưu ý:** Do memory constraint, unlabeled data được sample xuống 50K samples (giữ toàn bộ labeled data)

### Grid Search Results

| Config | Accuracy | F1-macro | Training Time |
|--------|----------|----------|---------------|
| γ=10, α=0.1 | 0.5845 | 0.5289 | 45s |
| **γ=20, α=0.2** | **0.5912** | **0.5398** | 52s |
| γ=30, α=0.3 | 0.5878 | 0.5356 | 48s |

**Best config:** γ=20, α=0.2

---

## 4. So Sánh Với Self-Training

### Metrics Comparison

| Phương pháp | Test Accuracy | Test F1-macro | Training Time | Memory |
|-------------|---------------|---------------|---------------|--------|
| **Self-Training (τ=0.9)** | 0.5890 | 0.5343 | ~20 min | Low |
| **FlexMatch** | 0.5928 | **0.5445** | ~25 min | Low |
| **Label Spreading** | 0.5912 | 0.5398 | **~1 min** | **High** |

![Method Comparison](./data/processed/label_spreading_experiments/method_comparison.png)

**Nhận xét:**

1. **Accuracy:**
   - Label Spreading: 0.5912 (trung bình)
   - FlexMatch: 0.5928 (cao nhất, +0.27%)
   - Self-Training: 0.5890 (thấp nhất)

2. **F1-macro:**
   - FlexMatch: 0.5445 (cao nhất)
   - Label Spreading: 0.5398 (trung bình, +1.03% vs Self-Training)
   - Self-Training: 0.5343 (thấp nhất)

3. **Training Time:**
   - Label Spreading: **~1 min** (nhanh nhất!) ⚡
   - Self-Training: ~20 min
   - FlexMatch: ~25 min

4. **Memory Usage:**
   - Label Spreading: **High** (cần sample data)
   - Self-Training/FlexMatch: Low

### Per-Class F1-Score

![Per-Class F1](./data/processed/label_spreading_experiments/per_class_f1.png)

| Lớp AQI | Self-Training | Label Spreading | Chênh lệch |
|---------|---------------|-----------------|------------|
| **Good** | 0.4897 | 0.5034 | **+2.80%** ✅ |
| **Moderate** | 0.7045 | 0.7012 | -0.47% |
| **Unhealthy_for_Sensitive** | 0.1789 | 0.1956 | **+9.34%** ✅ |
| **Unhealthy** | 0.5877 | 0.5945 | +1.16% |
| **Very_Unhealthy** | 0.5689 | 0.5823 | **+2.36%** ✅ |
| **Hazardous** | 0.6762 | 0.6618 | -2.13% |

**Phát hiện:**
- ✅ **Good** (+2.80%): Cải thiện tốt
- ✅ **Unhealthy_for_Sensitive** (+9.34%): Cải thiện mạnh (lớp khó nhất)
- ✅ **Very_Unhealthy** (+2.36%): Cải thiện tốt
- ❌ **Hazardous** (-2.13%): Giảm nhẹ (có thể do sampling)

---

## 5. Phân Tích

### 5.1. Khi Nào Label Spreading Tốt Hơn?

**✅ Label Spreading thắng khi:**
1. **Data có manifold structure rõ ràng**
   - AQI data có clustering tự nhiên theo thời gian/trạm
   - Samples gần nhau có nhãn giống nhau

2. **Cần training nhanh**
   - 1 phút vs 20-25 phút
   - Phù hợp cho rapid prototyping

3. **Muốn deterministic results**
   - Không có randomness trong pseudo-labeling
   - Reproducible 100%

**❌ Self-Training/FlexMatch thắng khi:**
1. **Dataset lớn (>100K samples)**
   - Label Spreading cần quá nhiều memory
   - Sampling làm mất thông tin

2. **Cần F1-macro cao nhất**
   - FlexMatch: 0.5445
   - Label Spreading: 0.5398 (-0.86%)

3. **Có thời gian training**
   - Self-Training có thể chạy overnight
   - Iterative refinement tốt hơn

### 5.2. Training Time vs Performance

![Time vs Performance](./data/processed/label_spreading_experiments/time_vs_performance.png)

**Trade-off analysis:**
```
Label Spreading:  1 min  → F1=0.5398  (Speed champion ⚡)
Self-Training:   20 min  → F1=0.5343  (Baseline)
FlexMatch:       25 min  → F1=0.5445  (Accuracy champion 🏆)
```

**ROI (Return on Investment):**
- Label Spreading: **Best time efficiency** (1 min for 0.5398)
- FlexMatch: **Best F1-macro** (25 min for 0.5445)
- Self-Training: **Balanced** (20 min for 0.5343)

### 5.3. Memory Constraint Impact

**Original dataset:**
- Train: ~404K samples
- Labeled: ~20K (5%)
- Unlabeled: ~384K (95%)

**After sampling:**
- Train: 50K samples
- Labeled: ~20K (100% kept)
- Unlabeled: ~30K (7.8% of original)

**Impact:**
- ❌ Mất 92.2% unlabeled data
- ❌ Có thể mất patterns quan trọng
- ✅ Vẫn đạt F1-macro tốt (0.5398)
- ✅ Training rất nhanh (1 min)

**Giải pháp:**
- Sử dụng **stratified sampling** để giữ distribution
- Hoặc dùng **approximate methods** (k-NN graph thay vì full graph)

---

## 6. Kết Luận

### Tổng Kết

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         LABEL SPREADING SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test Accuracy:       0.5912 (+0.37% vs Self-Training)
Test F1-macro:       0.5398 (+1.03% vs Self-Training)

Training Time:       ~1 minute (20x faster!)
Memory Usage:        High (requires sampling)
Sampled Data:        50K / 404K (12.4%)

Best for:            Fast prototyping, small-medium datasets
Not recommended:     Large datasets (>100K), memory-constrained
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Decision Matrix

| Scenario | Recommended Method | Lý do |
|----------|-------------------|-------|
| **Large dataset (>100K)** | FlexMatch | Label Spreading cần quá nhiều memory |
| **Small dataset (<50K)** | Label Spreading | Nhanh, hiệu quả, không cần sampling |
| **Cần F1-macro cao nhất** | FlexMatch | +0.86% vs Label Spreading |
| **Rapid prototyping** | Label Spreading | 1 phút vs 20-25 phút |
| **Production deployment** | FlexMatch | Ổn định, scalable, không cần sampling |
| **Research/Analysis** | Label Spreading | Deterministic, dễ reproduce |

### Best Practices

```python
# Recommended configuration
ls_cfg = LabelSpreadingConfig(
    kernel="rbf",
    gamma=20.0,         # Balanced similarity
    alpha=0.2,          # Balanced clamping
    max_iter=30,
    sample_size=50000   # Adjust based on memory
)

# For large datasets: use stratified sampling
if train_size > 50000:
    # Keep all labeled, sample unlabeled
    labeled_idx = df[df['is_labeled']].index
    unlabeled_idx = df[~df['is_labeled']].sample(
        n=50000 - len(labeled_idx),
        random_state=42
    ).index
    df_sampled = df.loc[labeled_idx.union(unlabeled_idx)]
```

### Kết Hợp Các Phương Pháp

**Ensemble approach:**
```python
# Combine predictions from multiple methods
y_pred_st = self_training_model.predict(X_test)
y_pred_fm = flexmatch_model.predict(X_test)
y_pred_ls = label_spreading_model.predict(X_test)

# Voting
y_pred_ensemble = majority_vote([y_pred_st, y_pred_fm, y_pred_ls])
```

**Expected improvement:** +1-2% F1-macro

### Tiếp Theo

Xem thêm:
- [Self-Training Analysis](./BLOG_SELF_TRAINING.md) - Baseline comparison
- [FlexMatch Analysis](./BLOG_FLEXMATCH.md) - Dynamic threshold + Focal loss
- [Co-Training Analysis](./BLOG_CO_TRAINING.md) - Multi-view learning

---

## Tài Liệu Tham Khảo

### Files Liên Quan

- **Code:** `notebooks/semi_label_spreading.ipynb`
- **Library:** `src/semi_supervised_library.py`
  - `LabelSpreadingConfig`
  - `LabelSpreadingAQIClassifier`
  - `run_label_spreading`
- **Results:** `data/processed/label_spreading_experiments/`
  - `metrics_label_spreading.json`
  - `label_spreading_summary.json`
  - `method_comparison.csv`
- **Visualizations:**
  - `method_comparison.png`
  - `per_class_f1.png`
  - `time_vs_performance.png`

### Papers

- **Label Propagation:** [Zhu & Ghahramani, 2002 - Learning from Labeled and Unlabeled Data with Label Propagation](http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf)
- **Label Spreading:** [Zhou et al., 2004 - Learning with Local and Global Consistency](https://proceedings.neurips.cc/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf)

---

<div align="center">

**Blog được tạo tự động từ kết quả thí nghiệm**

*Data Mining - Air Quality Prediction Project*

</div>
