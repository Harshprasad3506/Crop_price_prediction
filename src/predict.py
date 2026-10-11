"""
predict.py
==========

Prediction module for the Crop Price Prediction project.

Machine Learning Algorithm:
    Support Vector Machine - Support Vector Regression (SVR)

Purpose:
    This script loads the trained SVR model and allows the user
    to enter crop-related information from the terminal.

Workflow:

    User Input
        ↓
    Input Validation
        ↓
    Feature Engineering
        ↓
    Saved Preprocessing Objects
        ↓
    TF-IDF / One-Hot / Feature Transformation
        ↓
    Trained SVR Model
        ↓
    Predicted Crop Price

Usage:
    python src/predict.py

The script provides both:
    1. Interactive terminal prediction
    2. Reusable predict_price() function
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import os
import sys
import pickle
import json
from datetime import datetime
from typing import Any, Dict, Optional


import numpy as np
import pandas as pd


# ============================================================
# 2. PROJECT PATH CONFIGURATION
# ============================================================

# Current file:
# Crop Price Prediction/src/predict.py
#
# PROJECT_ROOT therefore points to:
# Crop Price Prediction/

CURRENT_FILE = os.path.abspath(__file__)

SRC_DIR = os.path.dirname(
    CURRENT_FILE
)

PROJECT_ROOT = os.path.dirname(
    SRC_DIR
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
)


# ============================================================
# 3. MODEL FILE CONFIGURATION
# ============================================================

# Main trained model file.
MODEL_PATH = os.path.join(
    MODELS_DIR,
    "svm_regressor.pkl",
)


# Optional saved preprocessing object.
#
# If train.py saves a complete preprocessing pipeline,
# this file can be loaded automatically.
PREPROCESSOR_PATH = os.path.join(
    MODELS_DIR,
    "preprocessing_pipeline.pkl",
)


# Optional metadata generated during training.
METADATA_PATH = os.path.join(
    MODELS_DIR,
    "model_metadata.json",
)


# ============================================================
# 4. DEFAULT FEATURE CONFIGURATION
# ============================================================

# These are the core features created by preprocessing.py.

BASE_FEATURE_COLUMNS = [
    "commodity_name",
    "year",
    "month_number",
    "month_sin",
    "month_cos",
    "time_index",
]


# Optional categorical columns.
OPTIONAL_CATEGORICAL_COLUMNS = [
    "state_name",
    "district_name",
    "calculationType",
]


# ============================================================
# 5. DISPLAY CONFIGURATION
# ============================================================

CURRENCY_SYMBOL = "₹"

PRICE_DECIMAL_PLACES = 2


# ============================================================
# 6. UTILITY FUNCTIONS
# ============================================================

def print_separator(
    character: str = "=",
    length: int = 70,
) -> None:
    """
    Print a visual separator in the terminal.

    Parameters
    ----------
    character : str
        Character used for separator.

    length : int
        Number of characters.
    """

    print(
        character * length
    )


def print_header(
    title: str,
) -> None:
    """
    Print a formatted section header.

    Parameters
    ----------
    title : str
        Header text.
    """

    print()
    print_separator()
    print(title)
    print_separator()


# ============================================================
# 7. MODEL FILE VALIDATION
# ============================================================

def check_model_file(
    model_path: str = MODEL_PATH,
) -> None:
    """
    Check whether the trained model exists.

    Parameters
    ----------
    model_path : str
        Path to trained model.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """

    if not os.path.exists(
        model_path
    ):
        raise FileNotFoundError(
            "\nTrained model was not found.\n\n"
            f"Expected location:\n"
            f"{model_path}\n\n"
            "Please run train.py first:\n"
            "python src/train.py"
        )


# ============================================================
# 8. LOAD TRAINED MODEL
# ============================================================

def load_model(
    model_path: str = MODEL_PATH,
) -> Any:
    """
    Load the trained SVR model from disk.

    Parameters
    ----------
    model_path : str
        Path to saved model.

    Returns
    -------
    Any
        Loaded machine learning model.
    """

    check_model_file(
        model_path
    )

    try:
        with open(
            model_path,
            "rb",
        ) as model_file:

            model = pickle.load(
                model_file
            )

    except Exception as exc:

        raise RuntimeError(
            "Unable to load the trained model.\n"
            f"Model path: {model_path}\n"
            f"Error: {exc}"
        ) from exc

    return model


# ============================================================
# 9. LOAD PREPROCESSOR
# ============================================================

def load_preprocessor(
    preprocessor_path: str = PREPROCESSOR_PATH,
) -> Optional[Any]:
    """
    Load the saved preprocessing pipeline if available.

    This function is optional because some training pipelines
    save preprocessing directly inside the model pipeline.

    Parameters
    ----------
    preprocessor_path : str
        Path to saved preprocessor.

    Returns
    -------
    Optional[Any]
        Loaded preprocessor or None.
    """

    if not os.path.exists(
        preprocessor_path
    ):
        return None

    try:

        with open(
            preprocessor_path,
            "rb",
        ) as file:

            return pickle.load(
                file
            )

    except Exception as exc:

        print(
            "\nWarning: preprocessing pipeline "
            "could not be loaded."
        )

        print(
            f"Reason: {exc}"
        )

        return None


# ============================================================
# 10. LOAD MODEL METADATA
# ============================================================

def load_metadata(
    metadata_path: str = METADATA_PATH,
) -> Dict[str, Any]:
    """
    Load model metadata if available.

    Metadata can contain information such as:
        - model type
        - training date
        - feature names
        - training configuration
        - target variable

    Parameters
    ----------
    metadata_path : str
        Path to metadata JSON file.

    Returns
    -------
    dict
        Metadata dictionary.
    """

    if not os.path.exists(
        metadata_path
    ):
        return {}

    try:

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(
                file
            )

        return metadata

    except Exception as exc:

        print(
            "\nWarning: model metadata "
            "could not be loaded."
        )

        print(
            f"Reason: {exc}"
        )

        return {}


# ============================================================
# 11. INPUT CLEANING
# ============================================================

def clean_input_text(
    value: Any,
) -> str:
    """
    Clean a user-provided categorical value.

    Parameters
    ----------
    value : Any
        Input value.

    Returns
    -------
    str
        Cleaned string.
    """

    if value is None:
        return ""

    return str(
        value
    ).strip()


# ============================================================
# 12. VALIDATE CROP NAME
# ============================================================

def validate_crop_name(
    crop_name: str,
) -> str:
    """
    Validate crop/commodity name.

    Parameters
    ----------
    crop_name : str
        Crop name.

    Returns
    -------
    str
        Cleaned crop name.

    Raises
    ------
    ValueError
        If crop name is empty.
    """

    crop_name = clean_input_text(
        crop_name
    )

    if not crop_name:
        raise ValueError(
            "Crop name cannot be empty."
        )

    return crop_name


# ============================================================
# 13. PARSE MONTH INPUT
# ============================================================

def parse_month_input(
    month_input: str,
) -> pd.Timestamp:
    """
    Convert user month input into pandas Timestamp.

    Accepted examples:

        2025-01
        2025-01-01
        January 2025
        Jan 2025

    Parameters
    ----------
    month_input : str
        User-provided month.

    Returns
    -------
    pd.Timestamp
        Parsed date.

    Raises
    ------
    ValueError
        If input cannot be converted to a date.
    """

    month_input = clean_input_text(
        month_input
    )

    if not month_input:
        raise ValueError(
            "Month cannot be empty."
        )

    parsed_date = pd.to_datetime(
        month_input,
        errors="coerce",
    )

    if pd.isna(
        parsed_date
    ):
        raise ValueError(
            f"Invalid month: {month_input}\n"
            "Example format: 2025-06"
        )

    # Normalize to first day of month.
    parsed_date = pd.Timestamp(
        year=parsed_date.year,
        month=parsed_date.month,
        day=1,
    )

    return parsed_date


# ============================================================
# 14. VALIDATE OPTIONAL INPUT
# ============================================================

def validate_optional_input(
    value: Optional[str],
) -> Optional[str]:
    """
    Clean an optional text input.

    Parameters
    ----------
    value : str or None
        Optional value.

    Returns
    -------
    str or None
        Cleaned value.
    """

    if value is None:
        return None

    value = clean_input_text(
        value
    )

    if not value:
        return None

    return value


# ============================================================
# 15. CREATE DATE FEATURES
# ============================================================

def create_prediction_date_features(
    prediction_date: pd.Timestamp,
    reference_date: Optional[pd.Timestamp] = None,
) -> Dict[str, float]:
    """
    Create the same date-based features used during training.

    Features:
        year
        month_number
        month_sin
        month_cos
        time_index

    Parameters
    ----------
    prediction_date : pd.Timestamp
        Requested prediction month.

    reference_date : pd.Timestamp, optional
        Reference date used for time_index.

    Returns
    -------
    dict
        Date feature dictionary.
    """

    prediction_date = pd.Timestamp(
        prediction_date
    )

    if reference_date is None:
        # When no reference date is available,
        # use the prediction date as reference.
        reference_date = prediction_date

    reference_date = pd.Timestamp(
        reference_date
    )

    year = int(
        prediction_date.year
    )

    month_number = int(
        prediction_date.month
    )

    month_sin = np.sin(
        2
        * np.pi
        * month_number
        / 12
    )

    month_cos = np.cos(
        2
        * np.pi
        * month_number
        / 12
    )

    time_index = (
        (
            prediction_date.year
            - reference_date.year
        )
        * 12
        + (
            prediction_date.month
            - reference_date.month
        )
    )

    return {
        "year": year,
        "month_number": month_number,
        "month_sin": float(
            month_sin
        ),
        "month_cos": float(
            month_cos
        ),
        "time_index": int(
            time_index
        ),
    }


# ============================================================
# 16. CREATE RAW PREDICTION DATA
# ============================================================

def create_prediction_dataframe(
    crop_name: str,
    month: str,
    state_name: Optional[str] = None,
    district_name: Optional[str] = None,
    calculation_type: Optional[str] = None,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Create a single-row dataframe for prediction.

    Parameters
    ----------
    crop_name : str
        Crop/commodity name.

    month : str
        Prediction month.

    state_name : str, optional
        State.

    district_name : str, optional
        District.

    calculation_type : str, optional
        Calculation type.

    reference_date : pd.Timestamp, optional
        Reference date for time index.

    Returns
    -------
    pd.DataFrame
        Prediction feature dataframe.
    """

    # Validate crop.
    crop_name = validate_crop_name(
        crop_name
    )

    # Parse date.
    prediction_date = parse_month_input(
        month
    )

    # Clean optional values.
    state_name = validate_optional_input(
        state_name
    )

    district_name = validate_optional_input(
        district_name
    )

    calculation_type = validate_optional_input(
        calculation_type
    )

    # Create date features.
    date_features = (
        create_prediction_date_features(
            prediction_date,
            reference_date,
        )
    )

    # Build row.
    row = {
        "commodity_name": crop_name,
        **date_features,
    }

    # Add optional categorical fields.
    if state_name is not None:
        row["state_name"] = (
            state_name
        )

    if district_name is not None:
        row["district_name"] = (
            district_name
        )

    if calculation_type is not None:
        row["calculationType"] = (
            calculation_type
        )

    return pd.DataFrame(
        [row]
    )


# ============================================================
# 17. DETERMINE MODEL FEATURES
# ============================================================

def get_expected_features(
    model: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[list]:
    """
    Try to determine the exact feature columns expected
    by the trained model.

    Sources checked:
        1. Model feature_names_in_
        2. Metadata
        3. None

    Parameters
    ----------
    model : Any
        Loaded model.

    metadata : dict, optional
        Model metadata.

    Returns
    -------
    list or None
        Expected feature names.
    """

    # --------------------------------------------------------
    # Try sklearn feature_names_in_
    # --------------------------------------------------------
    if hasattr(
        model,
        "feature_names_in_",
    ):

        try:

            return list(
                model.feature_names_in_
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # Try metadata
    # --------------------------------------------------------
    if metadata:

        for key in [
            "feature_columns",
            "features",
            "model_features",
        ]:

            if key in metadata:

                features = metadata[key]

                if isinstance(
                    features,
                    list,
                ):
                    return features

    return None


# ============================================================
# 18. ALIGN FEATURES
# ============================================================

def align_features_for_model(
    prediction_df: pd.DataFrame,
    expected_features: Optional[list],
) -> pd.DataFrame:
    """
    Align prediction dataframe with model input features.

    Parameters
    ----------
    prediction_df : pd.DataFrame
        Prepared prediction dataframe.

    expected_features : list or None
        Features expected by model.

    Returns
    -------
    pd.DataFrame
        Aligned dataframe.
    """

    if expected_features is None:

        # If the model does not expose feature names,
        # use the dataframe as prepared.
        return prediction_df.copy()

    missing_features = [
        feature
        for feature in expected_features
        if feature not in prediction_df.columns
    ]

    if missing_features:

        raise ValueError(
            "\nPrediction feature mismatch.\n\n"
            "Missing features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
            + "\n\nAvailable features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in prediction_df.columns
            )
        )

    return prediction_df[
        expected_features
    ].copy()


# ============================================================
# 19. APPLY PREPROCESSOR
# ============================================================

def transform_features(
    prediction_df: pd.DataFrame,
    model: Any,
    preprocessor: Optional[Any] = None,
) -> Any:
    """
    Transform prediction features using the saved preprocessor.

    The function supports common sklearn preprocessing
    objects such as ColumnTransformer, Pipeline and
    other objects implementing transform().

    Parameters
    ----------
    prediction_df : pd.DataFrame
        Raw prediction features.

    model : Any
        Loaded model.

    preprocessor : Any, optional
        Saved preprocessing object.

    Returns
    -------
    Any
        Transformed features ready for model prediction.
    """

    # --------------------------------------------------------
    # Case 1: Separate preprocessor exists
    # --------------------------------------------------------
    if preprocessor is not None:

        if not hasattr(
            preprocessor,
            "transform",
        ):
            raise TypeError(
                "Saved preprocessor does not "
                "provide a transform() method."
            )

        return preprocessor.transform(
            prediction_df
        )

    # --------------------------------------------------------
    # Case 2: Model itself is a Pipeline
    # --------------------------------------------------------
    if hasattr(
        model,
        "named_steps",
    ):

        # A sklearn Pipeline normally handles
        # preprocessing internally.
        return prediction_df

    # --------------------------------------------------------
    # Case 3: Model has transform()
    # --------------------------------------------------------
    if hasattr(
        model,
        "transform",
    ):

        return model.transform(
            prediction_df
        )

    # --------------------------------------------------------
    # Case 4: No preprocessing object
    # --------------------------------------------------------
    return prediction_df


# ============================================================
# 20. RUN MODEL PREDICTION
# ============================================================

def run_model_prediction(
    model: Any,
    features: Any,
) -> float:
    """
    Generate a prediction using the trained model.

    Parameters
    ----------
    model : Any
        Trained regression model.

    features : Any
        Model-ready feature data.

    Returns
    -------
    float
        Predicted crop price.
    """

    if not hasattr(
        model,
        "predict",
    ):
        raise TypeError(
            "Loaded object does not contain "
            "a predict() method."
        )

    try:

        prediction = model.predict(
            features
        )

    except Exception as exc:

        raise RuntimeError(
            "\nModel prediction failed.\n"
            f"Error: {exc}\n\n"
            "This usually means that the prediction "
            "features do not match the features used "
            "during training."
        ) from exc

    if len(prediction) == 0:
        raise RuntimeError(
            "Model returned an empty prediction."
        )

    predicted_price = float(
        prediction[0]
    )

    if not np.isfinite(
        predicted_price
    ):
        raise ValueError(
            "Model returned an invalid "
            "prediction value."
        )

    return predicted_price


# ============================================================
# 21. FORMAT PRICE
# ============================================================

def format_price(
    price: float,
) -> str:
    """
    Format predicted price for terminal display.

    Parameters
    ----------
    price : float
        Predicted price.

    Returns
    -------
    str
        Formatted price.
    """

    return (
        f"{CURRENCY_SYMBOL}"
        f"{price:,.{PRICE_DECIMAL_PLACES}f}"
    )


# ============================================================
# 22. MAIN PREDICTION FUNCTION
# ============================================================

def predict_price(
    crop_name: str,
    month: str,
    state_name: Optional[str] = None,
    district_name: Optional[str] = None,
    calculation_type: Optional[str] = None,
    model: Optional[Any] = None,
    preprocessor: Optional[Any] = None,
    reference_date: Optional[pd.Timestamp] = None,
) -> float:
    """
    Predict crop price for the supplied input.

    Parameters
    ----------
    crop_name : str
        Crop/commodity name.

    month : str
        Month for which price should be predicted.

    state_name : str, optional
        State name.

    district_name : str, optional
        District name.

    calculation_type : str, optional
        Calculation type.

    model : Any, optional
        Already-loaded model.

    preprocessor : Any, optional
        Already-loaded preprocessor.

    reference_date : pd.Timestamp, optional
        Reference date used for time_index.

    Returns
    -------
    float
        Predicted crop price.
    """

    # --------------------------------------------------------
    # Load model if not supplied.
    # --------------------------------------------------------
    if model is None:
        model = load_model()

    # --------------------------------------------------------
    # Prepare prediction dataframe.
    # --------------------------------------------------------
    prediction_df = (
        create_prediction_dataframe(
            crop_name=crop_name,
            month=month,
            state_name=state_name,
            district_name=district_name,
            calculation_type=calculation_type,
            reference_date=reference_date,
        )
    )

    # --------------------------------------------------------
    # Load metadata.
    # --------------------------------------------------------
    metadata = load_metadata()

    # --------------------------------------------------------
    # Determine expected features.
    # --------------------------------------------------------
    expected_features = (
        get_expected_features(
            model,
            metadata,
        )
    )

    # --------------------------------------------------------
    # Align feature columns.
    # --------------------------------------------------------
    prediction_df = (
        align_features_for_model(
            prediction_df,
            expected_features,
        )
    )

    # --------------------------------------------------------
    # Apply preprocessing.
    # --------------------------------------------------------
    model_features = (
        transform_features(
            prediction_df,
            model,
            preprocessor,
        )
    )

    # --------------------------------------------------------
    # Generate prediction.
    # --------------------------------------------------------
    predicted_price = (
        run_model_prediction(
            model,
            model_features,
        )
    )

    return predicted_price


# ============================================================
# 23. DISPLAY PREDICTION RESULT
# ============================================================

def display_prediction_result(
    crop_name: str,
    month: str,
    predicted_price: float,
    state_name: Optional[str] = None,
    district_name: Optional[str] = None,
) -> None:
    """
    Display prediction result in a clean format.

    Parameters
    ----------
    crop_name : str
        Crop name.

    month : str
        Prediction month.

    predicted_price : float
        Predicted price.

    state_name : str, optional
        State.

    district_name : str, optional
        District.
    """

    print()
    print_separator()
    print("CROP PRICE PREDICTION RESULT")
    print_separator()

    print(
        f"Crop             : {crop_name}"
    )

    print(
        f"Prediction Month : {month}"
    )

    if state_name:
        print(
            f"State            : {state_name}"
        )

    if district_name:
        print(
            f"District         : {district_name}"
        )

    print_separator()

    print(
        "Predicted Price  : "
        f"{format_price(predicted_price)}"
    )

    print_separator()

    print(
        "\nNote: This is a machine-learning prediction "
        "based on the patterns learned from the training "
        "dataset. It should not be treated as a guaranteed "
        "market price."
    )


# ============================================================
# 24. GET USER INPUT
# ============================================================

def get_user_input() -> Dict[str, Optional[str]]:
    """
    Collect prediction information from the user.

    Returns
    -------
    dict
        User input dictionary.
    """

    print_header(
        "CROP PRICE PREDICTION"
    )

    print(
        "Enter the following information."
    )

    print(
        "Press Enter for optional fields."
    )

    print()

    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------
    crop_name = input(
        "Enter crop name: "
    ).strip()

    # --------------------------------------------------------
    # Month
    # --------------------------------------------------------
    month = input(
        "Enter prediction month "
        "(example: 2025-06): "
    ).strip()

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------
    state_name = input(
        "Enter state "
        "(optional): "
    ).strip()

    # --------------------------------------------------------
    # District
    # --------------------------------------------------------
    district_name = input(
        "Enter district "
        "(optional): "
    ).strip()

    # --------------------------------------------------------
    # Calculation type
    # --------------------------------------------------------
    calculation_type = input(
        "Enter calculation type "
        "(optional): "
    ).strip()

    return {
        "crop_name": crop_name,
        "month": month,
        "state_name": (
            state_name
            if state_name
            else None
        ),
        "district_name": (
            district_name
            if district_name
            else None
        ),
        "calculation_type": (
            calculation_type
            if calculation_type
            else None
        ),
    }


# ============================================================
# 25. INTERACTIVE PREDICTION
# ============================================================

def interactive_prediction() -> None:
    """
    Run the interactive terminal prediction system.

    The user enters crop information and receives
    a predicted price.
    """

    try:

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------
        print()
        print(
            "Loading trained SVM/SVR model..."
        )

        model = load_model()

        print(
            "Model loaded successfully."
        )

        # ----------------------------------------------------
        # Load optional preprocessor
        # ----------------------------------------------------
        preprocessor = (
            load_preprocessor()
        )

        if preprocessor is not None:

            print(
                "Preprocessing pipeline loaded."
            )

        else:

            print(
                "No separate preprocessing "
                "pipeline found."
            )

        # ----------------------------------------------------
        # Get user input
        # ----------------------------------------------------
        user_input = (
            get_user_input()
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------
        predicted_price = (
            predict_price(
                crop_name=user_input[
                    "crop_name"
                ],
                month=user_input[
                    "month"
                ],
                state_name=user_input[
                    "state_name"
                ],
                district_name=user_input[
                    "district_name"
                ],
                calculation_type=user_input[
                    "calculation_type"
                ],
                model=model,
                preprocessor=preprocessor,
            )
        )

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------
        display_prediction_result(
            crop_name=user_input[
                "crop_name"
            ],
            month=user_input[
                "month"
            ],
            predicted_price=predicted_price,
            state_name=user_input[
                "state_name"
            ],
            district_name=user_input[
                "district_name"
            ],
        )

    except KeyboardInterrupt:

        print(
            "\n\nPrediction cancelled by user."
        )

    except Exception as exc:

        print()
        print_separator()
        print("PREDICTION ERROR")
        print_separator()

        print(
            str(exc)
        )

        print_separator()


# ============================================================
# 26. MODEL INFORMATION
# ============================================================

def display_model_information(
    model: Any,
    metadata: Dict[str, Any],
) -> None:
    """
    Display basic information about the trained model.

    Parameters
    ----------
    model : Any
        Loaded model.

    metadata : dict
        Model metadata.
    """

    print_header(
        "MODEL INFORMATION"
    )

    model_name = type(
        model
    ).__name__

    print(
        f"Model class : {model_name}"
    )

    if metadata:

        if "algorithm" in metadata:
            print(
                f"Algorithm   : "
                f"{metadata['algorithm']}"
            )

        if "target_column" in metadata:
            print(
                f"Target      : "
                f"{metadata['target_column']}"
            )

        if "training_date" in metadata:
            print(
                f"Training    : "
                f"{metadata['training_date']}"
            )

    if hasattr(
        model,
        "C",
    ):

        print(
            f"SVM C       : "
            f"{model.C}"
        )

    if hasattr(
        model,
        "epsilon",
    ):

        print(
            f"Epsilon     : "
            f"{model.epsilon}"
        )

    if hasattr(
        model,
        "gamma",
    ):

        print(
            f"Gamma       : "
            f"{model.gamma}"
        )


# ============================================================
# 27. RUN MODEL CHECK
# ============================================================

def run_model_check() -> bool:
    """
    Check whether the model can be loaded successfully.

    Returns
    -------
    bool
        True if model loads successfully.
    """

    try:

        model = load_model()

        print(
            "\n[PASS] Trained model loaded."
        )

        print(
            f"[INFO] Model type: "
            f"{type(model).__name__}"
        )

        return True

    except Exception as exc:

        print(
            "\n[FAIL] Model loading failed."
        )

        print(
            f"Reason: {exc}"
        )

        return False


# ============================================================
# 28. MAIN FUNCTION
# ============================================================

def main() -> None:
    """
    Main entry point of predict.py.
    """

    print_header(
        "CROP PRICE PREDICTION - SVM/SVR"
    )

    print(
        "Machine Learning Based Crop Price Prediction"
    )

    print(
        f"Project directory: {PROJECT_ROOT}"
    )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------
    if not run_model_check():

        print(
            "\nPlease train the model first."
        )

        print(
            "Run:"
        )

        print(
            "python src/train.py"
        )

        return

    # --------------------------------------------------------
    # Load model and metadata
    # --------------------------------------------------------
    model = load_model()

    metadata = load_metadata()

    # --------------------------------------------------------
    # Display information
    # --------------------------------------------------------
    display_model_information(
        model,
        metadata,
    )

    # --------------------------------------------------------
    # Start prediction
    # --------------------------------------------------------
    interactive_prediction()


# ============================================================
# 29. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
