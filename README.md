# Crop Price Prediction using Support Vector Machine (SVM)

## Project Overview
This project is an ML mini-project that aims to predict crop prices using **Support Vector Regression (SVR)** from `sklearn.svm.SVR`.

- **Project Title:** Crop Price Prediction using Support Vector Machine (SVM)
- **Problem Type:** Regression
- **Algorithm:** Support Vector Regression (SVR)
- **Dataset:** Crop Price Dataset (`crop_price_dataset.csv`)
- **Target Variable:** `avg_modal_price` (Continuous Numerical Value)

## Directory Structure
```text
Crop Price Prediction/
│
├── data/
│   └── crop_price_dataset.csv
│
├── notebooks/
│   └── crop_price_svm.ipynb
│
├── models/
│   ├── svm_regressor.pkl
│   └── preprocessing_pipeline.pkl
│
├── results/
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   ├── price_distribution.png
│   └── metrics.json
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore
```
