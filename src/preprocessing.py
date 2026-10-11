"""
preprocessing.py
================

Reusable preprocessing utilities for the Crop Price Prediction
machine learning project.

Project:
    Crop Price Prediction

Machine Learning Algorithm:
    Support Vector Machine - Support Vector Regression (SVR)

Problem Type:
    Regression

Target Variable:
    avg_modal_price

Purpose:
    This module contains reusable functions for:

    1. Dataset validation
    2. Missing-value handling
    3. Duplicate removal
    4. Data type conversion
    5. Categorical data cleaning
    6. Date conversion
    7. Date feature engineering
    8. Feature selection
    9. Input validation for prediction
   10. Preparation of new input data

Important:
    The target variable and price-derived columns such as
    avg_min_price, avg_max_price and change are not used as
    prediction features because they may cause target leakage
    when predicting a future crop price.

This module contains NO machine learning algorithm.
The SVR model is implemented separately in train.py.
"""

# ============================================================
# 1. IMPORTS
# ============================================================

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Target variable used by the regression model.
TARGET_COLUMN = "avg_modal_price"


# Required columns expected in the original dataset.
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


# Columns containing categorical/text information.
CATEGORICAL_COLUMNS = [
    "commodity_name",
    "state_name",
    "district_name",
    "calculationType",
]


# Price-related columns that should not be directly used
# as predictors for future price prediction.
LEAKAGE_COLUMNS = [
    "avg_modal_price",
    "avg_min_price",
    "avg_max_price",
    "change",
]


# Basic model input features.
BASE_FEATURE_COLUMNS = [
    "commodity_name",
    "year",
    "month_number",
    "month_sin",
    "month_cos",
    "time_index",
]


# Optional categorical features.
OPTIONAL_CATEGORICAL_COLUMNS = [
    "state_name",
    "district_name",
    "calculationType",
]


# ============================================================
# 3. DATASET VALIDATION
# ============================================================

def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
) -> None:
    """
    Validate that all required columns exist in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    required_columns : list, optional
        List of required columns. If not provided,
        REQUIRED_COLUMNS is used.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.

    ValueError
        If one or more required columns are missing.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    columns_to_check = (
        required_columns
        if required_columns is not None
        else REQUIRED_COLUMNS
    )

    missing_columns = [
        column
        for column in columns_to_check
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )


# ============================================================
# 4. DATAFRAME COPY
# ============================================================

def copy_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a safe copy of the input dataframe.

    This prevents preprocessing functions from accidentally
    modifying the original dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Independent dataframe copy.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    return df.copy()


# ============================================================
# 5. CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean dataframe column names.

    Operations:
        - Convert column names to strings.
        - Remove leading/trailing spaces.
        - Replace spaces with underscores.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with cleaned column names.
    """

    result = copy_dataframe(df)

    result.columns = (
        result.columns
        .astype(str)
        .str.strip()
        .str.replace(
            " ",
            "_",
            regex=False,
        )
    )

    return result


# ============================================================
# 6. ANALYZE MISSING VALUES
# ============================================================

def get_missing_value_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a missing-value report.

    The report contains:
        - column name
        - missing count
        - missing percentage

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Missing-value report.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    total_rows = len(df)

    missing_count = df.isnull().sum()

    if total_rows > 0:
        missing_percentage = (
            missing_count / total_rows
        ) * 100
    else:
        missing_percentage = missing_count

    report = pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_percentage": (
                missing_percentage.values
            ),
        }
    )

    report = report.sort_values(
        "missing_count",
        ascending=False,
    ).reset_index(drop=True)

    return report


# ============================================================
# 7. REMOVE DUPLICATES
# ============================================================

def remove_duplicate_rows(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    Tuple[pd.DataFrame, int]
        Cleaned dataframe and number of removed rows.
    """

    result = copy_dataframe(df)

    duplicate_count = int(
        result.duplicated().sum()
    )

    result = (
        result
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return result, duplicate_count


# ============================================================
# 8. CLEAN CATEGORICAL VALUES
# ============================================================

def clean_categorical_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Clean categorical/text columns.

    Operations:
        - Convert values to strings.
        - Remove leading/trailing spaces.
        - Replace empty strings with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    columns : list, optional
        Categorical columns to clean.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """

    result = copy_dataframe(df)

    columns_to_clean = (
        columns
        if columns is not None
        else CATEGORICAL_COLUMNS
    )

    for column in columns_to_clean:

        if column not in result.columns:
            continue

        # Preserve missing values before converting
        # everything to string.
        result[column] = result[column].where(
            result[column].notna(),
            np.nan,
        )

        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
        )

        result[column] = result[column].replace(
            "",
            np.nan,
        )

    return result


# ============================================================
# 9. CONVERT TARGET TO NUMERIC
# ============================================================

def convert_target_to_numeric(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Convert the target column into numeric format.

    Invalid values are converted to NaN so that they can
    be handled during the cleaning stage.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    target_column : str
        Name of target column.

    Returns
    -------
    pd.DataFrame
        Dataframe with numeric target.
    """

    result = copy_dataframe(df)

    if target_column not in result.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "was not found."
        )

    result[target_column] = pd.to_numeric(
        result[target_column],
        errors="coerce",
    )

    return result


# ============================================================
# 10. CONVERT DATE COLUMN
# ============================================================

def convert_month_to_datetime(
    df: pd.DataFrame,
    date_column: str = "month",
) -> pd.DataFrame:
    """
    Convert the month column into pandas datetime format.

    Invalid date values are converted to NaT.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    date_column : str
        Date/month column.

    Returns
    -------
    pd.DataFrame
        Dataframe with datetime month column.
    """

    result = copy_dataframe(df)

    if date_column not in result.columns:
        raise ValueError(
            f"Date column '{date_column}' "
            "was not found."
        )

    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )

    return result


# ============================================================
# 11. REMOVE INVALID TARGET/DATE ROWS
# ============================================================

def remove_invalid_rows(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    date_column: str = "month",
) -> Tuple[pd.DataFrame, int]:
    """
    Remove rows with invalid target or date values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    target_column : str
        Target variable.

    date_column : str
        Date column.

    Returns
    -------
    Tuple[pd.DataFrame, int]
        Cleaned dataframe and number of removed rows.
    """

    result = copy_dataframe(df)

    before = len(result)

    result = result.dropna(
        subset=[
            target_column,
            date_column,
        ]
    )

    result = result.reset_index(
        drop=True
    )

    removed = before - len(result)

    return result, removed


# ============================================================
# 12. REMOVE MISSING CATEGORICAL ROWS
# ============================================================

def remove_missing_categorical_rows(
    df: pd.DataFrame,
    required_categories: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Remove rows where important categorical fields are missing.

    By default, commodity_name is required because it is a
    core prediction feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    required_categories : list, optional
        Categorical columns that must have values.

    Returns
    -------
    Tuple[pd.DataFrame, int]
        Cleaned dataframe and number of removed rows.
    """

    result = copy_dataframe(df)

    if required_categories is None:
        required_categories = [
            "commodity_name"
        ]

    available_columns = [
        column
        for column in required_categories
        if column in result.columns
    ]

    if not available_columns:
        return result, 0

    before = len(result)

    result = result.dropna(
        subset=available_columns
    )

    result = result.reset_index(
        drop=True
    )

    removed = before - len(result)

    return result, removed


# ============================================================
# 13. REMOVE INVALID PRICES
# ============================================================

def remove_invalid_target_values(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Remove records with zero or negative target prices.

    Crop prices should normally be positive. This function
    removes invalid target values from the training dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    target_column : str
        Target price column.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """

    result = copy_dataframe(df)

    if target_column not in result.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "was not found."
        )

    result = result[
        result[target_column] > 0
    ].copy()

    result = result.reset_index(
        drop=True
    )

    return result


# ============================================================
# 14. COMPLETE DATASET CLEANING
# ============================================================

def clean_dataset(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Perform the complete dataset cleaning pipeline.

    Steps:
        1. Copy dataframe
        2. Clean column names
        3. Validate required columns
        4. Remove duplicate rows
        5. Clean categorical columns
        6. Convert target to numeric
        7. Convert month to datetime
        8. Remove invalid target/date rows
        9. Remove missing commodity names
       10. Remove non-positive target prices

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    target_column : str
        Target variable.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    result = copy_dataframe(df)

    # --------------------------------------------------------
    # Step 1: Clean column names
    # --------------------------------------------------------
    result = clean_column_names(
        result
    )

    # --------------------------------------------------------
    # Step 2: Validate required columns
    # --------------------------------------------------------
    validate_required_columns(
        result
    )

    # --------------------------------------------------------
    # Step 3: Remove duplicates
    # --------------------------------------------------------
    result, _ = remove_duplicate_rows(
        result
    )

    # --------------------------------------------------------
    # Step 4: Clean categorical fields
    # --------------------------------------------------------
    result = clean_categorical_columns(
        result
    )

    # --------------------------------------------------------
    # Step 5: Convert target to numeric
    # --------------------------------------------------------
    result = convert_target_to_numeric(
        result,
        target_column,
    )

    # --------------------------------------------------------
    # Step 6: Convert month to datetime
    # --------------------------------------------------------
    result = convert_month_to_datetime(
        result
    )

    # --------------------------------------------------------
    # Step 7: Remove invalid date/target rows
    # --------------------------------------------------------
    result, _ = remove_invalid_rows(
        result,
        target_column,
        "month",
    )

    # --------------------------------------------------------
    # Step 8: Remove missing commodity names
    # --------------------------------------------------------
    result, _ = remove_missing_categorical_rows(
        result,
        ["commodity_name"],
    )

    # --------------------------------------------------------
    # Step 9: Remove invalid prices
    # --------------------------------------------------------
    result = remove_invalid_target_values(
        result,
        target_column,
    )

    return result


# ============================================================
# 15. DATE FEATURE ENGINEERING
# ============================================================

def create_date_features(
    df: pd.DataFrame,
    date_column: str = "month",
) -> pd.DataFrame:
    """
    Create useful numerical features from the month column.

    Generated features:

        year
            Calendar year.

        month_number
            Month number from 1 to 12.

        month_sin
            Cyclic representation of month.

        month_cos
            Cyclic representation of month.

        time_index
            Number of months from the earliest date.

    Cyclic month features allow the model to understand that
    December and January are close to each other in the yearly
    cycle.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.

    date_column : str
        Date column.

    Returns
    -------
    pd.DataFrame
        Dataframe with date-derived features.
    """

    result = copy_dataframe(df)

    if date_column not in result.columns:
        raise ValueError(
            f"Date column '{date_column}' "
            "was not found."
        )

    # Make sure the column is datetime.
    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )

    if result[date_column].isna().all():
        raise ValueError(
            "No valid dates were found."
        )

    # --------------------------------------------------------
    # Year
    # --------------------------------------------------------
    result["year"] = (
        result[date_column].dt.year
    )

    # --------------------------------------------------------
    # Month number
    # --------------------------------------------------------
    result["month_number"] = (
        result[date_column].dt.month
    )

    # --------------------------------------------------------
    # Cyclic month representation
    # --------------------------------------------------------
    result["month_sin"] = np.sin(
        2
        * np.pi
        * result["month_number"]
        / 12
    )

    result["month_cos"] = np.cos(
        2
        * np.pi
        * result["month_number"]
        / 12
    )

    # --------------------------------------------------------
    # Time index
    # --------------------------------------------------------
    minimum_date = result[
        date_column
    ].min()

    result["time_index"] = (
        (
            result[date_column].dt.year
            - minimum_date.year
        )
        * 12
        + (
            result[date_column].dt.month
            - minimum_date.month
        )
    )

    return result


# ============================================================
# 16. GET UNIQUE VALUE COUNTS
# ============================================================

def get_unique_value_counts(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Get the number of unique values for selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    columns : list, optional
        Columns to analyze.

    Returns
    -------
    Dict[str, int]
        Dictionary containing unique-value counts.
    """

    if columns is None:
        columns = [
            column
            for column in CATEGORICAL_COLUMNS
            if column in df.columns
        ]

    result = {}

    for column in columns:
        if column in df.columns:
            result[column] = int(
                df[column].nunique(
                    dropna=True
                )
            )

    return result


# ============================================================
# 17. SELECT MODEL FEATURES
# ============================================================

def get_model_feature_columns(
    df: pd.DataFrame,
) -> List[str]:
    """
    Determine which columns should be used as model inputs.

    Core features:
        commodity_name
        year
        month_number
        month_sin
        month_cos
        time_index

    Optional categorical features are included only when
    they contain more than one unique value.

    Price-derived columns are intentionally excluded.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataframe.

    Returns
    -------
    List[str]
        Selected feature column names.
    """

    selected_features = []

    # --------------------------------------------------------
    # Core features
    # --------------------------------------------------------
    for column in BASE_FEATURE_COLUMNS:

        if column not in df.columns:
            raise ValueError(
                f"Required model feature "
                f"'{column}' is missing."
            )

        selected_features.append(
            column
        )

    # --------------------------------------------------------
    # Optional categorical features
    # --------------------------------------------------------
    for column in OPTIONAL_CATEGORICAL_COLUMNS:

        if column not in df.columns:
            continue

        unique_count = df[column].nunique(
            dropna=True
        )

        # Constant columns provide no useful information.
        if unique_count > 1:
            selected_features.append(
                column
            )

    return selected_features


# ============================================================
# 18. PREPARE FEATURES AND TARGET
# ============================================================

def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X and y for model training.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and feature-engineered dataset.

    target_column : str
        Target variable.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X and y.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "was not found."
        )

    feature_columns = get_model_feature_columns(
        df
    )

    X = df[
        feature_columns
    ].copy()

    y = df[
        target_column
    ].copy()

    return X, y


# ============================================================
# 19. PREPARE DATASET FOR TRAINING
# ============================================================

def prepare_training_dataset(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Execute the complete preprocessing workflow for training.

    Workflow:

        Raw Data
            ↓
        Cleaning
            ↓
        Date Features
            ↓
        Feature Selection
            ↓
        X / y

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        X, y, processed dataframe
    """

    # Clean the raw data.
    cleaned = clean_dataset(
        df,
        target_column,
    )

    # Create time-related features.
    processed = create_date_features(
        cleaned
    )

    # Prepare model features and target.
    X, y = prepare_features_and_target(
        processed,
        target_column,
    )

    return X, y, processed


# ============================================================
# 20. PREPARE NEW PREDICTION INPUT
# ============================================================

def prepare_prediction_input(
    commodity_name: str,
    month: str,
    state_name: Optional[str] = None,
    district_name: Optional[str] = None,
    calculation_type: Optional[str] = None,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Prepare a single new record for prediction.

    This function is designed for predict.py.

    Example
    -------
    input_data = prepare_prediction_input(
        commodity_name="Wheat",
        month="2025-06-01",
        state_name="Maharashtra",
        district_name="Pune",
        calculation_type="Average",
    )

    Parameters
    ----------
    commodity_name : str
        Name of crop/commodity.

    month : str
        Prediction month.

    state_name : str, optional
        State name.

    district_name : str, optional
        District name.

    calculation_type : str, optional
        Calculation type.

    reference_date : pd.Timestamp, optional
        Reference date used to calculate time_index.

    Returns
    -------
    pd.DataFrame
        Prepared feature dataframe.
    """

    # --------------------------------------------------------
    # Validate commodity
    # --------------------------------------------------------
    if commodity_name is None:
        raise ValueError(
            "commodity_name cannot be None."
        )

    commodity_name = str(
        commodity_name
    ).strip()

    if not commodity_name:
        raise ValueError(
            "commodity_name cannot be empty."
        )

    # --------------------------------------------------------
    # Convert month
    # --------------------------------------------------------
    prediction_date = pd.to_datetime(
        month,
        errors="coerce",
    )

    if pd.isna(prediction_date):
        raise ValueError(
            f"Invalid month value: {month}"
        )

    # --------------------------------------------------------
    # Reference date
    # --------------------------------------------------------
    if reference_date is None:
        reference_date = prediction_date

    reference_date = pd.to_datetime(
        reference_date,
        errors="coerce",
    )

    if pd.isna(reference_date):
        raise ValueError(
            "Invalid reference_date."
        )

    # --------------------------------------------------------
    # Create basic record
    # --------------------------------------------------------
    data = {
        "commodity_name": [
            commodity_name
        ],
        "month": [
            prediction_date
        ],
    }

    # Add optional categorical values.
    if state_name is not None:
        data["state_name"] = [
            str(state_name).strip()
        ]

    if district_name is not None:
        data["district_name"] = [
            str(district_name).strip()
        ]

    if calculation_type is not None:
        data["calculationType"] = [
            str(calculation_type).strip()
        ]

    result = pd.DataFrame(data)

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------
    result["year"] = (
        result["month"].dt.year
    )

    result["month_number"] = (
        result["month"].dt.month
    )

    result["month_sin"] = np.sin(
        2
        * np.pi
        * result["month_number"]
        / 12
    )

    result["month_cos"] = np.cos(
        2
        * np.pi
        * result["month_number"]
        / 12
    )

    # --------------------------------------------------------
    # Time index
    # --------------------------------------------------------
    result["time_index"] = (
        (
            result["month"].dt.year
            - reference_date.year
        )
        * 12
        + (
            result["month"].dt.month
            - reference_date.month
        )
    )

    # Remove original date column because the model
    # uses engineered date features instead.
    result = result.drop(
        columns=["month"]
    )

    return result


# ============================================================
# 21. ALIGN PREDICTION FEATURES
# ============================================================

def align_prediction_features(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Align prediction input with the exact feature columns
    expected by the trained preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        New prediction data.

    feature_columns : list
        Feature columns expected by the model.

    Returns
    -------
    pd.DataFrame
        Feature-aligned dataframe.

    Raises
    ------
    ValueError
        If required features are missing.
    """

    result = copy_dataframe(df)

    missing_columns = [
        column
        for column in feature_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            "Prediction input is missing required "
            "features: "
            + ", ".join(missing_columns)
        )

    return result[
        feature_columns
    ].copy()


# ============================================================
# 22. GET PREPROCESSING SUMMARY
# ============================================================

def get_preprocessing_summary(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Generate a summary of preprocessing changes.

    Parameters
    ----------
    original_df : pd.DataFrame
        Dataset before preprocessing.

    processed_df : pd.DataFrame
        Dataset after preprocessing.

    Returns
    -------
    Dict[str, object]
        Preprocessing summary.
    """

    original_rows = len(
        original_df
    )

    processed_rows = len(
        processed_df
    )

    removed_rows = (
        original_rows
        - processed_rows
    )

    summary = {
        "original_rows": int(
            original_rows
        ),
        "processed_rows": int(
            processed_rows
        ),
        "removed_rows": int(
            removed_rows
        ),
        "original_columns": int(
            original_df.shape[1]
        ),
        "processed_columns": int(
            processed_df.shape[1]
        ),
        "target_column": TARGET_COLUMN,
        "leakage_columns_excluded": (
            LEAKAGE_COLUMNS.copy()
        ),
        "model_features": (
            get_model_feature_columns(
                processed_df
            )
            if all(
                column in processed_df.columns
                for column in BASE_FEATURE_COLUMNS
            )
            else []
        ),
    }

    return summary


# ============================================================
# 23. DISPLAY PREPROCESSING SUMMARY
# ============================================================

def print_preprocessing_summary(
    summary: Dict[str, object],
) -> None:
    """
    Print preprocessing summary in a readable format.

    Parameters
    ----------
    summary : dict
        Summary returned by
        get_preprocessing_summary().
    """

    print("\n")
    print("=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)

    print(
        f"Original rows      : "
        f"{summary['original_rows']}"
    )

    print(
        f"Processed rows     : "
        f"{summary['processed_rows']}"
    )

    print(
        f"Removed rows       : "
        f"{summary['removed_rows']}"
    )

    print(
        f"Original columns   : "
        f"{summary['original_columns']}"
    )

    print(
        f"Processed columns  : "
        f"{summary['processed_columns']}"
    )

    print(
        f"Target variable    : "
        f"{summary['target_column']}"
    )

    print("\nExcluded price-derived columns:")

    for column in summary[
        "leakage_columns_excluded"
    ]:
        print(
            f"  - {column}"
        )

    print("\nModel features:")

    for column in summary[
        "model_features"
    ]:
        print(
            f"  - {column}"
        )

    print("=" * 70)


# ============================================================
# 24. FULL PREPROCESSING PIPELINE
# ============================================================

def preprocess_for_training(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    High-level preprocessing function.

    This is the main reusable function for training.

    Workflow:

        Raw Dataset
              ↓
        Column Cleaning
              ↓
        Validation
              ↓
        Duplicate Removal
              ↓
        Missing Value Handling
              ↓
        Numeric Conversion
              ↓
        Date Conversion
              ↓
        Invalid Record Removal
              ↓
        Date Feature Engineering
              ↓
        Feature Selection
              ↓
        X / y

    Parameters
    ----------
    df : pd.DataFrame
        Raw crop price dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        X, y, processed dataframe.
    """

    original_df = copy_dataframe(
        df
    )

    X, y, processed_df = (
        prepare_training_dataset(
            df
        )
    )

    summary = (
        get_preprocessing_summary(
            original_df,
            processed_df,
        )
    )

    print_preprocessing_summary(
        summary
    )

    return (
        X,
        y,
        processed_df,
    )


# ============================================================
# 25. MODULE SELF-TEST
# ============================================================

def run_preprocessing_self_test() -> None:
    """
    Run a lightweight self-test.

    This test does not require the actual dataset.

    It verifies:
        - Date conversion
        - Date feature generation
        - Prediction input preparation
        - Feature alignment
    """

    print("\n")
    print("=" * 70)
    print("PREPROCESSING MODULE SELF-TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Test 1: Date features
    # --------------------------------------------------------
    sample_df = pd.DataFrame(
        {
            "month": [
                "2024-01-01",
                "2024-06-01",
                "2024-12-01",
            ],
            "commodity_name": [
                "Wheat",
                "Rice",
                "Maize",
            ],
        }
    )

    sample_df = convert_month_to_datetime(
        sample_df
    )

    sample_df = create_date_features(
        sample_df
    )

    required_test_features = [
        "year",
        "month_number",
        "month_sin",
        "month_cos",
        "time_index",
    ]

    for feature in required_test_features:
        assert feature in sample_df.columns, (
            f"Missing generated feature: {feature}"
        )

    print(
        "[PASS] Date feature engineering"
    )

    # --------------------------------------------------------
    # Test 2: Prediction input
    # --------------------------------------------------------
    prediction_input = (
        prepare_prediction_input(
            commodity_name="Wheat",
            month="2025-01-01",
            state_name="Maharashtra",
            district_name="Pune",
            calculation_type="Average",
        )
    )

    assert (
        "commodity_name"
        in prediction_input.columns
    )

    assert (
        "year"
        in prediction_input.columns
    )

    assert (
        "month_number"
        in prediction_input.columns
    )

    print(
        "[PASS] Prediction input preparation"
    )

    # --------------------------------------------------------
    # Test 3: Feature alignment
    # --------------------------------------------------------
    expected_features = [
        "commodity_name",
        "year",
        "month_number",
        "month_sin",
        "month_cos",
        "time_index",
    ]

    aligned = align_prediction_features(
        prediction_input,
        expected_features,
    )

    assert list(
        aligned.columns
    ) == expected_features

    print(
        "[PASS] Prediction feature alignment"
    )

    print("=" * 70)
    print(
        "All preprocessing self-tests passed."
    )
    print("=" * 70)


# ============================================================
# 26. MAIN
# ============================================================

if __name__ == "__main__":
    run_preprocessing_self_test()