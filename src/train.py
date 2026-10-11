"""
train.py
========

Crop Price Prediction using Support Vector Regression (SVR).

Project:
    Crop Price Prediction

Machine Learning Algorithm:
    Support Vector Machine - Support Vector Regression (SVR)

Problem Type:
    Regression

Target Variable:
    avg_modal_price

Dataset:
    data/crop_price_dataset.csv

Pipeline:
    Dataset
        ↓
    Data Validation
        ↓
    Data Cleaning
        ↓
    Date Feature Engineering
        ↓
    Feature Selection
        ↓
    Time-based Train/Test Split
        ↓
    Categorical Encoding
        ↓
    Feature Scaling
        ↓
    SVR Training
        ↓
    Prediction
        ↓
    Evaluation
        ↓
    Visualization
        ↓
    Model Saving


Important:
    This script intentionally does not use avg_min_price,
    avg_max_price, or change as direct input features because
    these variables can be derived from the same price record
    and may introduce target leakage for future-price prediction.

The trained model and preprocessing transformer are saved
separately so that predict.py can use the exact same
preprocessing during inference.
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import json
import logging
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Percentage of latest observations used for testing.
TEST_SIZE = 0.20

# Target column.
TARGET_COLUMN = "avg_modal_price"

# Dataset filename.
DATASET_FILENAME = "crop_price_dataset.csv"

# SVR configuration.
SVR_KERNEL = "rbf"
SVR_C = 100.0
SVR_EPSILON = 0.1
SVR_GAMMA = "scale"

# Required dataset columns.
REQUIRED_COLUMNS = [
    "month",
    "commodity_name",
    "avg_modal_price",
    "avg_min_price",
    "avg_max_price",
    "state_name",
    "district_name",
    "calculationType",
    "change",
]


# ============================================================
# 3. PROJECT PATHS
# ============================================================

# train.py is located inside:
# Crop Price Prediction/src/train.py
#
# Therefore parent.parent gives:
# Crop Price Prediction/

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_PATH = DATA_DIR / DATASET_FILENAME

MODEL_PATH = MODELS_DIR / "svm_regressor.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessing_pipeline.pkl"

METRICS_PATH = RESULTS_DIR / "metrics.json"

PRICE_DISTRIBUTION_PATH = RESULTS_DIR / "price_distribution.png"
ACTUAL_VS_PREDICTED_PATH = RESULTS_DIR / "actual_vs_predicted.png"
RESIDUAL_PLOT_PATH = RESULTS_DIR / "residual_plot.png"

MISCLASSIFIED_PATH = RESULTS_DIR / "prediction_errors.csv"


# ============================================================
# 4. LOGGING
# ============================================================

logger = logging.getLogger("crop_price_training")


def setup_logging() -> None:
    """
    Configure logging for the training pipeline.

    The logging format is intentionally simple so that
    the training process is easy to understand during
    a college project demonstration.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )


# ============================================================
# 5. RANDOM SEED
# ============================================================

def set_random_seed(seed: int = RANDOM_STATE) -> None:
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    """

    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# 6. DIRECTORY MANAGEMENT
# ============================================================

def create_project_directories() -> None:
    """
    Create required project directories if they do not exist.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Project directories verified.")


# ============================================================
# 7. DATASET VALIDATION
# ============================================================

def validate_dataset_file() -> None:
    """
    Verify that the dataset exists.

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be found.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found.\n"
            f"Expected location:\n"
            f"{DATASET_PATH}\n\n"
            f"Please place '{DATASET_FILENAME}' inside the data folder."
        )

    logger.info("Dataset found: %s", DATASET_PATH)


# ============================================================
# 8. LOAD DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the crop price dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    logger.info("Loading dataset...")

    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read dataset: {exc}"
        ) from exc

    logger.info(
        "Dataset loaded successfully: %d rows × %d columns",
        df.shape[0],
        df.shape[1],
    )

    return df


# ============================================================
# 9. DATASET COLUMN VALIDATION
# ============================================================

def validate_dataset_columns(df: pd.DataFrame) -> None:
    """
    Verify that required dataset columns are present.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Raises
    ------
    ValueError
        If required columns are missing.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing:\n"
            + "\n".join(f"- {column}" for column in missing_columns)
        )

    logger.info("All required dataset columns are present.")


# ============================================================
# 10. DISPLAY DATASET OVERVIEW
# ============================================================

def display_dataset_overview(df: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.
    """

    print("\n" + "=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)

    print(f"Rows       : {df.shape[0]}")
    print(f"Columns    : {df.shape[1]}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData Types:")
    print(df.dtypes.to_string())

    print("\nFirst 5 Records:")
    print(df.head().to_string())

    print("=" * 70)


# ============================================================
# 11. MISSING VALUE ANALYSIS
# ============================================================

def analyze_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Analyze missing values in every column.

    Returns
    -------
    pd.Series
        Missing-value counts.
    """

    missing_counts = df.isnull().sum()

    print("\n" + "=" * 70)
    print("MISSING VALUE ANALYSIS")
    print("=" * 70)

    print(missing_counts.to_string())

    total_missing = int(missing_counts.sum())

    logger.info("Total missing values: %d", total_missing)

    return missing_counts


# ============================================================
# 12. DUPLICATE ANALYSIS
# ============================================================

def analyze_duplicates(df: pd.DataFrame) -> int:
    """
    Count duplicate rows.

    Returns
    -------
    int
        Number of duplicate rows.
    """

    duplicate_count = int(df.duplicated().sum())

    print("\n" + "=" * 70)
    print("DUPLICATE ANALYSIS")
    print("=" * 70)
    print(f"Duplicate rows: {duplicate_count}")

    logger.info("Duplicate rows detected: %d", duplicate_count)

    return duplicate_count


# ============================================================
# 13. BASIC DATA CLEANING
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic dataset cleaning.

    Cleaning operations:
        - Copy dataframe.
        - Convert target to numeric.
        - Convert month to datetime.
        - Remove rows with invalid target/date.
        - Remove exact duplicate rows.
        - Clean categorical text values.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """

    logger.info("Starting data cleaning...")

    cleaned = df.copy()

    # Remove exact duplicate records.
    before_duplicates = len(cleaned)

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    removed_duplicates = before_duplicates - len(cleaned)

    if removed_duplicates > 0:
        logger.info(
            "Removed %d duplicate records.",
            removed_duplicates,
        )

    # Convert target to numeric.
    cleaned[TARGET_COLUMN] = pd.to_numeric(
        cleaned[TARGET_COLUMN],
        errors="coerce",
    )

    # Convert date.
    cleaned["month"] = pd.to_datetime(
        cleaned["month"],
        errors="coerce",
    )

    # Remove rows where target/date cannot be interpreted.
    before_invalid = len(cleaned)

    cleaned = cleaned.dropna(
        subset=[
            "month",
            TARGET_COLUMN,
        ]
    ).reset_index(drop=True)

    removed_invalid = before_invalid - len(cleaned)

    if removed_invalid > 0:
        logger.info(
            "Removed %d records with invalid date/target values.",
            removed_invalid,
        )

    # Clean categorical values.
    categorical_columns = [
        "commodity_name",
        "state_name",
        "district_name",
        "calculationType",
    ]

    for column in categorical_columns:
        cleaned[column] = (
            cleaned[column]
            .astype(str)
            .str.strip()
        )

        # Convert empty strings into missing values.
        cleaned[column] = cleaned[column].replace(
            "",
            np.nan,
        )

    # Remove rows with missing commodity.
    cleaned = cleaned.dropna(
        subset=["commodity_name"]
    ).reset_index(drop=True)

    logger.info(
        "Data cleaning completed. Remaining rows: %d",
        len(cleaned),
    )

    return cleaned


# ============================================================
# 14. DATE FEATURE ENGINEERING
# ============================================================

def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create numerical time-related features.

    Features:
        year
        month_number
        month_sin
        month_cos
        time_index

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with engineered time features.
    """

    result = df.copy()

    result["year"] = result["month"].dt.year
    result["month_number"] = result["month"].dt.month

    # Cyclic representation of month.
    result["month_sin"] = np.sin(
        2 * np.pi * result["month_number"] / 12
    )

    result["month_cos"] = np.cos(
        2 * np.pi * result["month_number"] / 12
    )

    # Number of months from the first available month.
    minimum_date = result["month"].min()

    result["time_index"] = (
        (result["month"].dt.year - minimum_date.year) * 12
        + (
            result["month"].dt.month
            - minimum_date.month
        )
    )

    logger.info("Date features created successfully.")

    return result


# ============================================================
# 15. TARGET ANALYSIS
# ============================================================

def display_target_statistics(df: pd.DataFrame) -> None:
    """
    Display descriptive statistics for the target variable.
    """

    target = df[TARGET_COLUMN]

    print("\n" + "=" * 70)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 70)

    print(f"Target variable: {TARGET_COLUMN}")
    print(f"Minimum price  : {target.min():.2f}")
    print(f"Maximum price  : {target.max():.2f}")
    print(f"Mean price     : {target.mean():.2f}")
    print(f"Median price   : {target.median():.2f}")
    print(f"Std deviation  : {target.std():.2f}")

    print("\nDetailed Statistics:")
    print(target.describe().to_string())

    print("=" * 70)


# ============================================================
# 16. CATEGORICAL ANALYSIS
# ============================================================

def display_categorical_statistics(df: pd.DataFrame) -> None:
    """
    Display unique values for important categorical columns.
    """

    categorical_columns = [
        "commodity_name",
        "state_name",
        "district_name",
        "calculationType",
    ]

    print("\n" + "=" * 70)
    print("CATEGORICAL FEATURE ANALYSIS")
    print("=" * 70)

    for column in categorical_columns:
        unique_count = df[column].nunique()

        print(
            f"{column}: "
            f"{unique_count} unique value(s)"
        )

        if unique_count <= 20:
            values = sorted(
                df[column].dropna().astype(str).unique()
            )

            print("  Values:", ", ".join(values))

    print("=" * 70)


# ============================================================
# 17. PRICE DISTRIBUTION PLOT
# ============================================================

def create_price_distribution_plot(
    df: pd.DataFrame,
) -> None:
    """
    Create and save the target price distribution plot.
    """

    logger.info("Creating price distribution plot...")

    plt.figure(figsize=(10, 6))

    plt.hist(
        df[TARGET_COLUMN],
        bins=40,
        edgecolor="black",
    )

    plt.title(
        "Distribution of Average Modal Crop Price"
    )

    plt.xlabel(
        "Average Modal Price"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        PRICE_DISTRIBUTION_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Saved: %s",
        PRICE_DISTRIBUTION_PATH,
    )


# ============================================================
# 18. FEATURE SELECTION
# ============================================================

def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare model input features and target.

    The following features are intentionally NOT used
    as direct predictors:

        avg_modal_price
            Target variable.

        avg_min_price
        avg_max_price
        change

    These price-derived variables may introduce leakage
    when the objective is to predict a future modal price.

    State and district are checked for variation. If they
    contain only one unique value, they are excluded because
    constant columns provide no predictive information.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature dataframe X and target series y.
    """

    logger.info("Preparing features and target...")

    working = df.copy()

    # Features intentionally selected for the model.
    candidate_features = [
        "commodity_name",
        "year",
        "month_number",
        "month_sin",
        "month_cos",
        "time_index",
    ]

    # State/district/calculationType may be useful in a
    # more geographically diverse dataset. Add them only
    # when they contain actual variation.
    optional_categorical = [
        "state_name",
        "district_name",
        "calculationType",
    ]

    selected_features = candidate_features.copy()

    for column in optional_categorical:
        if column in working.columns:
            unique_count = working[column].nunique(
                dropna=True
            )

            if unique_count > 1:
                selected_features.append(column)
            else:
                logger.info(
                    "Skipping constant feature '%s'.",
                    column,
                )

    X = working[selected_features].copy()
    y = working[TARGET_COLUMN].copy()

    logger.info(
        "Selected features: %s",
        ", ".join(selected_features),
    )

    logger.info(
        "Number of input features before encoding: %d",
        len(selected_features),
    )

    return X, y


# ============================================================
# 19. TIME-BASED TRAIN/TEST SPLIT
# ============================================================

def time_based_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset chronologically.

    Earlier observations are used for training and the
    latest observations are used for testing.

    This is more suitable for price prediction than a
    completely random split because future records should
    not be used to train a model that predicts earlier data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Training and testing dataframes.
    """

    logger.info("Creating chronological train/test split...")

    sorted_df = (
        df.sort_values("month")
        .reset_index(drop=True)
    )

    split_index = int(
        len(sorted_df) * (1 - TEST_SIZE)
    )

    if split_index <= 0 or split_index >= len(sorted_df):
        raise ValueError(
            "Invalid train/test split. "
            "Check dataset size and TEST_SIZE."
        )

    train_df = sorted_df.iloc[
        :split_index
    ].copy()

    test_df = sorted_df.iloc[
        split_index:
    ].copy()

    logger.info(
        "Training records: %d",
        len(train_df),
    )

    logger.info(
        "Testing records: %d",
        len(test_df),
    )

    logger.info(
        "Training period: %s → %s",
        train_df["month"].min().date(),
        train_df["month"].max().date(),
    )

    logger.info(
        "Testing period: %s → %s",
        test_df["month"].min().date(),
        test_df["month"].max().date(),
    )

    return train_df, test_df


# ============================================================
# 20. CREATE PREPROCESSOR
# ============================================================

def create_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """
    Create the preprocessing transformer.

    Numerical features:
        StandardScaler

    Categorical features:
        OneHotEncoder

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing transformer.
    """

    numeric_features = [
        column
        for column in X.columns
        if pd.api.types.is_numeric_dtype(
            X[column]
        )
    ]

    categorical_features = [
        column
        for column in X.columns
        if not pd.api.types.is_numeric_dtype(
            X[column]
        )
    ]

    logger.info(
        "Numerical features: %s",
        ", ".join(numeric_features)
        if numeric_features
        else "None",
    )

    logger.info(
        "Categorical features: %s",
        ", ".join(categorical_features)
        if categorical_features
        else "None",
    )

    numeric_transformer = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            )
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                numeric_transformer,
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError(
            "No valid features were found for preprocessing."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor


# ============================================================
# 21. CREATE SVR MODEL
# ============================================================

def create_svr_model() -> SVR:
    """
    Create the Support Vector Regression model.

    Returns
    -------
    SVR
        Configured SVR estimator.
    """

    model = SVR(
        kernel=SVR_KERNEL,
        C=SVR_C,
        epsilon=SVR_EPSILON,
        gamma=SVR_GAMMA,
    )

    logger.info(
        "SVR model created."
    )

    logger.info(
        "Kernel   : %s",
        SVR_KERNEL,
    )

    logger.info(
        "C        : %s",
        SVR_C,
    )

    logger.info(
        "Epsilon  : %s",
        SVR_EPSILON,
    )

    logger.info(
        "Gamma    : %s",
        SVR_GAMMA,
    )

    return model


# ============================================================
# 22. PREPARE TRAINING DATA
# ============================================================

def prepare_train_test_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Convert train/test dataframes into X and y.

    Returns
    -------
    Tuple
        X_train, X_test, y_train, y_test
    """

    X_train, y_train = prepare_features_and_target(
        train_df
    )

    X_test, y_test = prepare_features_and_target(
        test_df
    )

    # Ensure identical feature order.
    X_test = X_test[X_train.columns]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# 23. TRAIN MODEL
# ============================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[SVR, ColumnTransformer]:
    """
    Fit preprocessing and train the SVR model.

    Preprocessing is fitted ONLY on training data.

    Returns
    -------
    Tuple[SVR, ColumnTransformer]
        Trained SVR model and fitted preprocessor.
    """

    logger.info(
        "Starting model training..."
    )

    preprocessor = create_preprocessor(
        X_train
    )

    # Fit preprocessing ONLY on training data.
    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    logger.info(
        "Training data transformed successfully."
    )

    logger.info(
        "Transformed training shape: %s",
        X_train_transformed.shape,
    )

    model = create_svr_model()

    model.fit(
        X_train_transformed,
        y_train,
    )

    logger.info(
        "SVR model training completed."
    )

    return model, preprocessor


# ============================================================
# 24. GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    model: SVR,
    preprocessor: ColumnTransformer,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Transform test data using the fitted preprocessor
    and generate SVR predictions.

    Important:
        No fitting occurs on test data.

    Returns
    -------
    np.ndarray
        Predicted target values.
    """

    logger.info(
        "Transforming test data..."
    )

    X_test_transformed = preprocessor.transform(
        X_test
    )

    logger.info(
        "Generating predictions..."
    )

    predictions = model.predict(
        X_test_transformed
    )

    logger.info(
        "Prediction completed."
    )

    return predictions


# ============================================================
# 25. MODEL EVALUATION
# ============================================================

def evaluate_model(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics.

    Metrics:
        MAE
        MSE
        RMSE
        R2

    Returns
    -------
    Dict[str, float]
        Evaluation metrics.
    """

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions,
    )

    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
    }

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(
        f"Mean Absolute Error (MAE)  : {mae:.4f}"
    )

    print(
        f"Mean Squared Error (MSE)   : {mse:.4f}"
    )

    print(
        f"Root Mean Squared Error    : {rmse:.4f}"
    )

    print(
        f"R² Score                   : {r2:.4f}"
    )

    print("=" * 70)

    logger.info(
        "Model evaluation completed."
    )

    return metrics


# ============================================================
# 26. ACTUAL VS PREDICTED PLOT
# ============================================================

def create_actual_vs_predicted_plot(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> None:
    """
    Create Actual vs Predicted price plot.
    """

    logger.info(
        "Creating Actual vs Predicted plot..."
    )

    plt.figure(figsize=(9, 7))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.65,
        edgecolors="black",
        linewidths=0.4,
    )

    # Perfect prediction reference line.
    minimum = min(
        float(y_test.min()),
        float(predictions.min()),
    )

    maximum = max(
        float(y_test.max()),
        float(predictions.max()),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=2,
    )

    plt.title(
        "Actual vs Predicted Crop Price"
    )

    plt.xlabel(
        "Actual Average Modal Price"
    )

    plt.ylabel(
        "Predicted Average Modal Price"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        ACTUAL_VS_PREDICTED_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Saved: %s",
        ACTUAL_VS_PREDICTED_PATH,
    )


# ============================================================
# 27. RESIDUAL PLOT
# ============================================================

def create_residual_plot(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> None:
    """
    Create a residual plot.

    Residual:
        Actual Price - Predicted Price
    """

    logger.info(
        "Creating residual plot..."
    )

    residuals = (
        np.asarray(y_test)
        - np.asarray(predictions)
    )

    plt.figure(figsize=(9, 6))

    plt.scatter(
        predictions,
        residuals,
        alpha=0.65,
        edgecolors="black",
        linewidths=0.4,
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=2,
    )

    plt.title(
        "Residual Plot"
    )

    plt.xlabel(
        "Predicted Average Modal Price"
    )

    plt.ylabel(
        "Residual (Actual - Predicted)"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        RESIDUAL_PLOT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Saved: %s",
        RESIDUAL_PLOT_PATH,
    )


# ============================================================
# 28. SAVE PREDICTION ERRORS
# ============================================================

def save_prediction_errors(
    test_df: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
) -> None:
    """
    Save actual prices, predictions and errors.

    This file is useful for analyzing where the model
    makes larger prediction errors.
    """

    output = pd.DataFrame(
        {
            "month": test_df["month"].values,
            "commodity_name": test_df[
                "commodity_name"
            ].values,
            "actual_price": np.asarray(y_test),
            "predicted_price": np.asarray(
                predictions
            ),
        }
    )

    output["absolute_error"] = (
        output["actual_price"]
        - output["predicted_price"]
    ).abs()

    output["percentage_error"] = np.where(
        output["actual_price"] != 0,
        (
            output["absolute_error"]
            / output["actual_price"]
        )
        * 100,
        np.nan,
    )

    output = output.sort_values(
        "absolute_error",
        ascending=False,
    )

    output.to_csv(
        MISCLASSIFIED_PATH,
        index=False,
    )

    logger.info(
        "Saved prediction error analysis: %s",
        MISCLASSIFIED_PATH,
    )


# ============================================================
# 29. SAVE METRICS
# ============================================================

def save_metrics(
    metrics: Dict[str, float],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Save evaluation metrics and experiment information
    to a JSON file.
    """

    output = {
        "project": "Crop Price Prediction",
        "algorithm": "Support Vector Regression (SVR)",
        "problem_type": "Regression",
        "target": TARGET_COLUMN,
        "dataset": DATASET_FILENAME,
        "dataset_rows_after_cleaning": int(
            len(train_df) + len(test_df)
        ),
        "training_rows": int(
            len(train_df)
        ),
        "testing_rows": int(
            len(test_df)
        ),
        "test_size": TEST_SIZE,
        "svr_parameters": {
            "kernel": SVR_KERNEL,
            "C": SVR_C,
            "epsilon": SVR_EPSILON,
            "gamma": SVR_GAMMA,
        },
        "metrics": metrics,
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=4,
        )

    logger.info(
        "Saved metrics: %s",
        METRICS_PATH,
    )


# ============================================================
# 30. SAVE MODEL
# ============================================================

def save_model(
    model: SVR,
    preprocessor: ColumnTransformer,
) -> None:
    """
    Save the trained SVR model and fitted preprocessing
    transformer using pickle.
    """

    logger.info(
        "Saving trained model..."
    )

    with open(
        MODEL_PATH,
        "wb",
    ) as file:
        pickle.dump(
            model,
            file,
        )

    logger.info(
        "Saved model: %s",
        MODEL_PATH,
    )

    logger.info(
        "Saving preprocessing pipeline..."
    )

    with open(
        PREPROCESSOR_PATH,
        "wb",
    ) as file:
        pickle.dump(
            preprocessor,
            file,
        )

    logger.info(
        "Saved preprocessing pipeline: %s",
        PREPROCESSOR_PATH,
    )


# ============================================================
# 31. VALIDATE SAVED ARTIFACTS
# ============================================================

def validate_saved_artifacts() -> None:
    """
    Verify that model and preprocessing files were saved.
    """

    print("\n" + "=" * 70)
    print("SAVED ARTIFACT VALIDATION")
    print("=" * 70)

    files_to_check = [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        METRICS_PATH,
        PRICE_DISTRIBUTION_PATH,
        ACTUAL_VS_PREDICTED_PATH,
        RESIDUAL_PLOT_PATH,
        MISCLASSIFIED_PATH,
    ]

    all_valid = True

    for file_path in files_to_check:
        exists = file_path.exists()

        status = "PASS" if exists else "FAIL"

        print(
            f"{status:<6} {file_path.relative_to(PROJECT_ROOT)}"
        )

        if not exists:
            all_valid = False

    print("=" * 70)

    if not all_valid:
        raise RuntimeError(
            "One or more expected output files were not created."
        )

    logger.info(
        "All expected artifacts validated successfully."
    )


# ============================================================
# 32. DISPLAY FINAL SUMMARY
# ============================================================

def display_final_summary(
    metrics: Dict[str, float],
) -> None:
    """
    Display a concise final training summary.
    """

    print("\n")
    print("=" * 70)
    print("CROP PRICE PREDICTION - TRAINING COMPLETED")
    print("=" * 70)

    print(f"Algorithm       : Support Vector Regression (SVR)")
    print(f"Kernel          : {SVR_KERNEL}")
    print(f"Target          : {TARGET_COLUMN}")
    print(f"MAE             : {metrics['MAE']:.4f}")
    print(f"MSE             : {metrics['MSE']:.4f}")
    print(f"RMSE            : {metrics['RMSE']:.4f}")
    print(f"R² Score        : {metrics['R2']:.4f}")

    print("\nGenerated Files:")
    print(f"  Model         : {MODEL_PATH}")
    print(f"  Preprocessor  : {PREPROCESSOR_PATH}")
    print(f"  Metrics       : {METRICS_PATH}")
    print(f"  Price Plot    : {PRICE_DISTRIBUTION_PATH}")
    print(f"  Actual/Pred   : {ACTUAL_VS_PREDICTED_PATH}")
    print(f"  Residual Plot : {RESIDUAL_PLOT_PATH}")
    print(f"  Error Analysis: {MISCLASSIFIED_PATH}")

    print("\nImportant:")
    print(
        "  The model uses chronological train/test splitting."
    )
    print(
        "  Preprocessing was fitted only on training data."
    )
    print(
        "  avg_min_price, avg_max_price and change were"
    )
    print(
        "  excluded to reduce potential target leakage."
    )

    print("=" * 70)


# ============================================================
# 33. MAIN TRAINING PIPELINE
# ============================================================

def main() -> None:
    """
    Execute the complete crop price prediction pipeline.
    """

    setup_logging()

    print("\n")
    print("=" * 70)
    print("CROP PRICE PREDICTION USING SVM")
    print("Support Vector Regression (SVR)")
    print("=" * 70)

    try:
        # ----------------------------------------------------
        # STEP 1: Reproducibility
        # ----------------------------------------------------
        set_random_seed()

        # ----------------------------------------------------
        # STEP 2: Directory setup
        # ----------------------------------------------------
        create_project_directories()

        # ----------------------------------------------------
        # STEP 3: Dataset validation
        # ----------------------------------------------------
        validate_dataset_file()

        # ----------------------------------------------------
        # STEP 4: Load dataset
        # ----------------------------------------------------
        df = load_dataset()

        # ----------------------------------------------------
        # STEP 5: Validate columns
        # ----------------------------------------------------
        validate_dataset_columns(df)

        # ----------------------------------------------------
        # STEP 6: Display overview
        # ----------------------------------------------------
        display_dataset_overview(df)

        # ----------------------------------------------------
        # STEP 7: Analyze missing values
        # ----------------------------------------------------
        analyze_missing_values(df)

        # ----------------------------------------------------
        # STEP 8: Analyze duplicates
        # ----------------------------------------------------
        analyze_duplicates(df)

        # ----------------------------------------------------
        # STEP 9: Clean dataset
        # ----------------------------------------------------
        df = clean_dataset(df)

        # ----------------------------------------------------
        # STEP 10: Date feature engineering
        # ----------------------------------------------------
        df = create_date_features(df)

        # ----------------------------------------------------
        # STEP 11: Target analysis
        # ----------------------------------------------------
        display_target_statistics(df)

        # ----------------------------------------------------
        # STEP 12: Categorical analysis
        # ----------------------------------------------------
        display_categorical_statistics(df)

        # ----------------------------------------------------
        # STEP 13: Price distribution visualization
        # ----------------------------------------------------
        create_price_distribution_plot(df)

        # ----------------------------------------------------
        # STEP 14: Chronological train/test split
        # ----------------------------------------------------
        train_df, test_df = time_based_split(df)

        # ----------------------------------------------------
        # STEP 15: Prepare X and y
        # ----------------------------------------------------
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = prepare_train_test_data(
            train_df,
            test_df,
        )

        # ----------------------------------------------------
        # STEP 16: Train SVR
        # ----------------------------------------------------
        model, preprocessor = train_model(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # STEP 17: Generate predictions
        # ----------------------------------------------------
        predictions = generate_predictions(
            model,
            preprocessor,
            X_test,
        )

        # ----------------------------------------------------
        # STEP 18: Evaluate model
        # ----------------------------------------------------
        metrics = evaluate_model(
            y_test,
            predictions,
        )

        # ----------------------------------------------------
        # STEP 19: Actual vs predicted plot
        # ----------------------------------------------------
        create_actual_vs_predicted_plot(
            y_test,
            predictions,
        )

        # ----------------------------------------------------
        # STEP 20: Residual plot
        # ----------------------------------------------------
        create_residual_plot(
            y_test,
            predictions,
        )

        # ----------------------------------------------------
        # STEP 21: Prediction error analysis
        # ----------------------------------------------------
        save_prediction_errors(
            test_df,
            y_test,
            predictions,
        )

        # ----------------------------------------------------
        # STEP 22: Save metrics
        # ----------------------------------------------------
        save_metrics(
            metrics,
            train_df,
            test_df,
        )

        # ----------------------------------------------------
        # STEP 23: Save model and preprocessor
        # ----------------------------------------------------
        save_model(
            model,
            preprocessor,
        )

        # ----------------------------------------------------
        # STEP 24: Validate generated files
        # ----------------------------------------------------
        validate_saved_artifacts()

        # ----------------------------------------------------
        # STEP 25: Final summary
        # ----------------------------------------------------
        display_final_summary(
            metrics
        )

    except FileNotFoundError as exc:
        logger.error(str(exc))
        print("\nTRAINING STOPPED")
        print(str(exc))
        sys.exit(1)

    except ValueError as exc:
        logger.error(
            "Data validation error: %s",
            exc,
        )
        print("\nTRAINING STOPPED")
        print(f"Error: {exc}")
        sys.exit(1)

    except Exception as exc:
        logger.exception(
            "Unexpected training error."
        )
        print("\nTRAINING FAILED")
        print(
            f"Unexpected error: {exc}"
        )
        print(
            "Check the error message above and verify the dataset,"
            " dependencies and project structure."
        )
        sys.exit(1)


# ============================================================
# 34. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
