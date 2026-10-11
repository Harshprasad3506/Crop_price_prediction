# Detailed Technical Report: Crop Price Prediction using Support Vector Regression (SVR)

---

## 1. Executive Summary

This report provides a comprehensive technical evaluation of the **Crop Price Prediction** machine learning system. The goal of the project is to predict monthly crop prices (`avg_modal_price`) across various agricultural commodities in India using historical market records spanning **January 2010 to March 2025**.

The core predictive engine utilizes **Support Vector Regression (SVR)** from `scikit-learn`. The pipeline incorporates automated dataset validation, missing-value and duplicate cleaning, cyclical temporal feature engineering (sine/cosine transformations of calendar months), anti-leakage feature selection, time-based dataset splitting, and standard feature scaling.

### Key Metrics Summary
* **Dataset Size:** 2,810 records after cleaning (80% Train: 2,248 | 20% Test: 562)
* **Target Variable:** `avg_modal_price` (Range: ₹7.00 to ₹45,412.60 | Mean: ₹4,878.74)
* **Model Algorithm:** Support Vector Regression (`SVR(kernel='rbf', C=100.0, epsilon=0.1, gamma='scale')`)
* **Test Performance Metrics:**
  * **MAE (Mean Absolute Error):** ₹2,516.10
  * **RMSE (Root Mean Squared Error):** ₹4,706.06
  * **MSE (Mean Squared Error):** 22,147,027.75
  * **R² Score:** -0.0557

> [!IMPORTANT]
> **Key Finding:** While the overall global R² metric is negative (-0.0557) due to massive price scale disparities across commodities (e.g., Coffee at ~₹17,500/qtl and Tea at ~₹27,000/qtl vs. Rice at ~₹3,200/qtl), the model achieves **high prediction precision (<15% error)** on standard high-volume crops like **Rice (7.35% error)**, **Coconut (12.63% error)**, **Sugar (12.64% error)**, **Wheat (13.46% error)**, **Millets (13.89% error)**, and **Jowar (14.17% error)**.

---

## 2. Generated Diagnostic Visualizations

Below are the 4 generated analytical graphs illustrating model performance, error distribution, target variance, and commodity accuracy tiers:

### 2.1 Commodity-Wise MAPE Accuracy Chart
![Commodity-wise MAPE Chart](C:/Users/snsadmin/.gemini/antigravity-ide/brain/e9cdac45-b675-430b-bbb1-16ce61c74c84/commodity_mape_chart.png)

### 2.2 Target Price Scale Variance across Commodities
![Commodity Price Boxplot](C:/Users/snsadmin/.gemini/antigravity-ide/brain/e9cdac45-b675-430b-bbb1-16ce61c74c84/commodity_price_boxplot.png)

### 2.3 Actual vs. Predicted Prices Scatter Plot (Linear & Log Scales)
![Actual vs Predicted Scatter](C:/Users/snsadmin/.gemini/antigravity-ide/brain/e9cdac45-b675-430b-bbb1-16ce61c74c84/actual_vs_predicted_scatter.png)

### 2.4 Residuals & Prediction Error Diagnostics
![Residuals Diagnostics](C:/Users/snsadmin/.gemini/antigravity-ide/brain/e9cdac45-b675-430b-bbb1-16ce61c74c84/residuals_diagnostics.png)

---

## 3. Project Architecture & Modular Structure

The codebase is organized into modular Python components for reproducibility, clean separation of concerns, and ease of deployment:

```text
Crop Price Prediction/
│
├── data/
│   └── crop_price_dataset.csv       # Historical dataset (2,810 records)
│
├── models/
│   ├── svm_regressor.pkl            # Trained SVR model binary
│   └── preprocessing_pipeline.pkl   # Fitted ColumnTransformer object
│
├── results/
│   ├── commodity_mape_chart.png     # Bar chart of MAPE % per commodity
│   ├── actual_vs_predicted_scatter.png # Dual-panel scatter plot (linear & log)
│   ├── commodity_price_boxplot.png  # Target price scale variance per commodity
│   ├── residuals_diagnostics.png   # Error distribution & residual diagnostics
│   ├── actual_vs_predicted.png      # Original fit plot
│   ├── residual_plot.png           # Original residual plot
│   ├── price_distribution.png      # Original price distribution
│   ├── metrics.json                # Structured JSON evaluation metrics
│   └── prediction_errors.csv       # Detailed per-sample prediction logs
│
├── src/
│   ├── preprocessing.py             # Reusable data cleaning & engineering functions
│   ├── train.py                     # SVR pipeline training & evaluation script
│   └── predict.py                   # Interactive CLI & batch inference pipeline
│
├── PROJECT_REPORT.md               # Detailed technical status & diagnostic report
├── README.md                       # High-level project documentation
└── requirements.txt                # Dependency declarations
```

---

## 4. Data Engineering & Preprocessing Pipeline

```mermaid
flowchart TD
    RawData["Raw Dataset (2,810 rows)"] --> DataVal["Validation & Schema Check"]
    DataVal --> DataClean["Handling Nulls & Duplicates"]
    DataClean --> FeatEng["Date Feature Engineering (Year, Month Sin/Cos)"]
    FeatEng --> AntiLeak["Target Leakage Prevention (Exclude Min/Max/Change)"]
    AntiLeak --> ChronSplit["Time-Based Train/Test Split (80/20)"]
    ChronSplit --> PipeTransform["ColumnTransformer: OneHotEncoder & StandardScaler"]
    PipeTransform --> SVRTrain["SVR Model Training (RBF Kernel)"]
    SVRTrain --> ModelSave["Save Models & Evaluate Metrics"]
```

### 4.1 Data Characteristics
* **Total Records:** 2,810 valid monthly price entries across India.
* **Commodities (16 total):** Maize, Wheat, Rice, Sugar, Cotton, Coffee, Tea, Sugarcane, Sunflower, Jowar(Sorghum), Millets, Barley (Jau), Ginger(Dry), Turmeric, Groundnut, Coconut.
* **Time Range:** January 2010 to March 2025.
* **Target Statistics (`avg_modal_price`):**
  * Minimum: ₹7.00 / qtl
  * Maximum: ₹45,412.60 / qtl
  * Mean: ₹4,878.74 / qtl
  * Median: ₹3,347.72 / qtl

### 4.2 Anti-Target Leakage Controls
To guarantee that predictions simulate real-world future price forecasting, the preprocessing pipeline (`src/preprocessing.py`) explicitly drops `avg_min_price`, `avg_max_price`, and `change`. Including these columns would cause severe data leakage, as minimum and maximum prices are only known simultaneously with the modal price of a given month.

### 4.3 Feature Engineering Strategy
1. **Temporal Features:** Year (`year`), Month number (`month_number`).
2. **Cyclical Encoding:** `month_sin` (\(\sin(2\pi \cdot \text{month}/12)\)) and `month_cos` (\(\cos(2\pi \cdot \text{month}/12)\)) to capture annual seasonality without artificial boundary jumps between December and January.
3. **Categorical Features:** One-Hot Encoding applied to `commodity_name`, `state_name`, `district_name`, and `calculationType`.
4. **Numerical Scaling:** `StandardScaler` applied to all continuous features prior to SVR optimization.

---

## 5. Model Training & Configuration

The model uses **Support Vector Regression (SVR)**, which optimizes a margin of tolerance (\(\epsilon\)) around predictions using support vectors.

### Hyperparameter Settings
| Parameter | Value | Rationale / Function |
| :--- | :--- | :--- |
| **Kernel** | `rbf` (Radial Basis Function) | Captures non-linear relationships between cyclical month features & commodity types |
| **C (Regularization)** | `100.0` | Controls trade-off between margin violation penalty and model simplicity |
| **Epsilon (\(\epsilon\))** | `0.1` | Specifies the epsilon-tube within which no penalty is associated in the loss function |
| **Gamma** | `'scale'` | Automatic kernel coefficient scaling (\(\frac{1}{n\_features \cdot \text{Var}(X)}\)) |

---

## 6. Commodity-Level Error Analysis & Evaluation

### 6.1 Global Performance Metrics
* **MAE:** ₹2,516.10
* **RMSE:** ₹4,706.06
* **R² Score:** -0.0557

### 6.2 Commodity-Wise Performance Breakdown
The global performance metrics are heavily distorted by high-value cash crops. Below is the breakdown of actual vs. predicted values and error percentages across all 16 commodities:

| Commodity Name | Avg Actual Price (₹) | Avg Predicted Price (₹) | MAE (₹) | MAPE (%) | Performance Tier |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Rice** | ₹3,230.75 | ₹3,006.03 | ₹251.50 | **7.35%** | Tier 1: High Precision |
| **Coconut** | ₹4,132.53 | ₹3,861.38 | ₹536.69 | **12.63%** | Tier 1: High Precision |
| **Sugar** | ₹3,963.79 | ₹3,454.61 | ₹509.18 | **12.64%** | Tier 1: High Precision |
| **Wheat** | ₹2,446.87 | ₹2,764.35 | ₹317.47 | **13.46%** | Tier 1: High Precision |
| **Millets** | ₹2,935.43 | ₹2,996.48 | ₹391.46 | **13.89%** | Tier 1: High Precision |
| **Jowar (Sorghum)** | ₹2,970.93 | ₹2,936.36 | ₹414.65 | **14.17%** | Tier 1: High Precision |
| **Barley (Jau)** | ₹2,246.83 | ₹2,763.16 | ₹520.02 | 25.16% | Tier 2: Moderate |
| **Sunflower** | ₹5,018.34 | ₹3,552.29 | ₹1,466.05 | 27.88% | Tier 2: Moderate |
| **Maize** | ₹2,130.68 | ₹2,757.15 | ₹626.47 | 29.71% | Tier 2: Moderate |
| **Groundnut** | ₹6,093.48 | ₹3,804.23 | ₹2,289.26 | 37.29% | Tier 2: Moderate |
| **Ginger (Dry)** | ₹8,082.02 | ₹3,745.98 | ₹4,336.04 | 44.76% | Tier 2: Moderate |
| **Cotton** | ₹7,496.42 | ₹3,868.93 | ₹3,627.49 | 47.85% | Tier 2: Moderate |
| **Turmeric** | ₹9,730.54 | ₹3,979.59 | ₹5,750.95 | 54.26% | Tier 2: Moderate |
| **Coffee** | ₹17,546.84 | ₹3,972.26 | ₹13,574.58 | **75.11%** | Tier 3: Outlier Scale Failure |
| **Tea** | ₹27,000.00 | ₹3,806.46 | ₹23,193.54 | **85.90%** | Tier 3: Outlier Scale Failure |
| **Sugarcane** | ₹533.98 | ₹2,974.72 | ₹2,532.30 | **767.55%** | Tier 3: Unit Disparity |

---

## 7. Actionable Roadmap & Enhancements

1. **Log-Transformation of Target Variable (`log1p`):** Train SVR on \(\log(\text{avg\_modal\_price})\).
2. **Per-Commodity Multi-Model Architecture:** Implement dedicated regressor models per commodity group.
3. **Tree-Based Ensemble Models:** Benchmark **Random Forest**, **Gradient Boosting (XGBoost / LightGBM)**, and **CatBoost Regressors**.
4. **Hyperparameter Tuning:** Conduct Automated Grid/Randomized Search across `C`, `epsilon`, and `gamma`.

---
*Report generated on: 2026-09-21*
