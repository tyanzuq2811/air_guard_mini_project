import os
import papermill as pm

# Run notebooks end-to-end (semi-supervised + supervised + regression + ARIMA)
os.makedirs("notebooks/runs", exist_ok=True)

KERNEL = "beijing_env"

# 1) Preprocessing + EDA
pm.execute_notebook(
    "notebooks/preprocessing_and_eda.ipynb",
    "notebooks/runs/preprocessing_and_eda_run.ipynb",
    parameters=dict(
        USE_UCIMLREPO=False,
        RAW_ZIP_PATH="data/raw/PRSA2017_Data_20130301-20170228.zip",
        OUTPUT_CLEANED_PATH="data/processed/cleaned.parquet",
        LAG_HOURS=[1, 3, 24],
    ),
    language="python",
    kernel_name=KERNEL,
)

# 2) NEW: Semi-supervised dataset preparation
pm.execute_notebook(
    "notebooks/semi_dataset_preparation.ipynb",
    "notebooks/runs/semi_dataset_preparation_run.ipynb",
    parameters=dict(
        CLEANED_PATH="data/processed/cleaned.parquet",
        OUTPUT_SEMI_DATASET_PATH="data/processed/dataset_for_semi.parquet",
        CUTOFF="2017-01-01",
        LABEL_MISSING_FRACTION=0.95,
        RANDOM_STATE=42,
    ),
    language="python",
    kernel_name=KERNEL,
)

# 3) Feature preparation for supervised baseline
pm.execute_notebook(
    "notebooks/feature_preparation.ipynb",
    "notebooks/runs/feature_preparation_run.ipynb",
    parameters=dict(
        CLEANED_PATH="data/processed/cleaned.parquet",
        OUTPUT_DATASET_PATH="data/processed/dataset_for_clf.parquet",
        DROP_ROWS_WITHOUT_TARGET=True,
    ),
    language="python",
    kernel_name=KERNEL,
)

# 4) Self-training (single τ = 0.90)
pm.execute_notebook(
    "notebooks/semi_self_training.ipynb",
    "notebooks/runs/semi_self_training_run.ipynb",
    parameters=dict(
        SEMI_DATASET_PATH="data/processed/dataset_for_semi.parquet",
        CUTOFF="2017-01-01",
        TAU=0.90,
        MAX_ITER=10,
        MIN_NEW_PER_ITER=20,
        VAL_FRAC=0.20,
        RANDOM_STATE=42,
        METRICS_PATH="data/processed/metrics_self_training.json",
        PRED_SAMPLE_PATH="data/processed/predictions_self_training_sample.csv",
        ALERTS_SAMPLE_PATH="data/processed/alerts_self_training_sample.csv",
        ALERT_FROM_CLASS="Unhealthy",
    ),
    language="python",
    kernel_name=KERNEL,
)

# 4b) Self-training EXPERIMENTS (multiple τ values for comparison)
print("\n" + "="*80)
print("CHẠY THÍ NGHIỆM SELF-TRAINING VỚI NHIỀU GIÁ TRỊ τ")
print("="*80)
pm.execute_notebook(
    "notebooks/semi_self_training_experiments.ipynb",
    "notebooks/runs/semi_self_training_experiments_run.ipynb",
    parameters=dict(
        SEMI_DATASET_PATH="data/processed/dataset_for_semi.parquet",
        CUTOFF="2017-01-01",
        TAU_VALUES=[0.80, 0.90, 0.95],  # Reduced to 3 for faster execution
        MAX_ITER=10,
        MIN_NEW_PER_ITER=20,
        VAL_FRAC=0.20,
        RANDOM_STATE=42,
        RESULTS_DIR="data/processed/self_training_experiments",
        ALERT_FROM_CLASS="Unhealthy",
    ),
    language="python",
    kernel_name=KERNEL,
)

# 5) Co-training
pm.execute_notebook(
    "notebooks/semi_co_training.ipynb",
    "notebooks/runs/semi_co_training_run.ipynb",
    parameters=dict(
        SEMI_DATASET_PATH="data/processed/dataset_for_semi.parquet",
        CUTOFF="2017-01-01",
        TAU=0.90,
        MAX_ITER=10,
        MAX_NEW_PER_ITER=500,
        MIN_NEW_PER_ITER=20,
        VAL_FRAC=0.20,
        RANDOM_STATE=42,
        METRICS_PATH="data/processed/metrics_co_training.json",
        PRED_SAMPLE_PATH="data/processed/predictions_co_training_sample.csv",
        ALERTS_SAMPLE_PATH="data/processed/alerts_co_training_sample.csv",
        ALERT_FROM_CLASS="Unhealthy",
        VIEW1_COLS=None,
        VIEW2_COLS=None,
    ),
    language="python",
    kernel_name=KERNEL,
)

# 6) Supervised classification baseline
pm.execute_notebook(
    "notebooks/classification_modelling.ipynb",
    "notebooks/runs/classification_modelling_run.ipynb",
    parameters=dict(
        DATASET_PATH="data/processed/dataset_for_clf.parquet",
        CUTOFF="2017-01-01",
        METRICS_PATH="data/processed/metrics.json",
        PRED_SAMPLE_PATH="data/processed/predictions_sample.csv",
    ),
    language="python",
    kernel_name=KERNEL,
)

# 7) Regression
pm.execute_notebook(
    "notebooks/regression_modelling.ipynb",
    "notebooks/runs/regression_modelling_run.ipynb",
    parameters=dict(
        USE_UCIMLREPO=False,
        RAW_ZIP_PATH="data/raw/PRSA2017_Data_20130301-20170228.zip",
        LAG_HOURS=[1, 3, 24],
        HORIZON=1,
        TARGET_COL="PM2.5",
        OUTPUT_REG_DATASET_PATH="data/processed/dataset_for_regression.parquet",
        CUTOFF="2017-01-01",
        MODEL_OUT="regressor.joblib",
        METRICS_OUT="regression_metrics.json",
        PRED_SAMPLE_OUT="regression_predictions_sample.csv",
    ),
    language="python",
    kernel_name=KERNEL,
)

# 8) ARIMA
pm.execute_notebook(
    "notebooks/arima_forecasting.ipynb",
    "notebooks/runs/arima_forecasting_run.ipynb",
    parameters=dict(
        RAW_ZIP_PATH="data/raw/PRSA2017_Data_20130301-20170228.zip",
        STATION="Aotizhongxin",
        VALUE_COL="PM2.5",
        CUTOFF="2017-01-01",
        P_MAX=3,
        Q_MAX=3,
        D_MAX=2,
        IC="aic",
        ARTIFACTS_PREFIX="arima_pm25",
    ),
    language="python",
    kernel_name=KERNEL,
)

# 9) Summary report (read metrics and make charts)
pm.execute_notebook(
    "notebooks/semi_supervised_report.ipynb",
    "notebooks/runs/semi_supervised_report_run.ipynb",
    parameters=dict(
        BASELINE_METRICS_PATH="data/processed/metrics.json",
        SELF_METRICS_PATHS=["data/processed/metrics_self_training.json"],
        CO_METRICS_PATHS=["data/processed/metrics_co_training.json"],
        BASELINE_PRED_PATH="data/processed/predictions_sample.csv",
        SELF_ALERTS_PATH="data/processed/alerts_self_training_sample.csv",
        CO_ALERTS_PATH="data/processed/alerts_co_training_sample.csv",
        STATION_TO_PLOT=None,
        MAX_ROWS_PLOT=1500,
    ),
    language="python",
    kernel_name=KERNEL,
)

print("\n" + "="*80)
print("🎉 ĐÃ CHẠY XONG TOÀN BỘ PIPELINE!")
print("="*80)
print("\n📊 KẾT QUẢ ĐÃ TẠO:")
print("✅ 1. Preprocessing & EDA")
print("✅ 2. Semi-supervised dataset")
print("✅ 3. Feature preparation")
print("✅ 4. Self-training (τ=0.90)")
print("✅ 4b. Self-training experiments (τ=0.70,0.80,0.85,0.90,0.95)")
print("✅ 5. Co-training")
print("✅ 6. Baseline supervised classification")
print("✅ 7. Regression modelling")
print("✅ 8. ARIMA forecasting")
print("✅ 9. Summary report")
print("\n📁 TẤT CẢ KẾT QUẢ TRONG:")
print("   - data/processed/")
print("   - data/processed/self_training_experiments/")
print("   - notebooks/runs/")
print("\n" + "="*80)
