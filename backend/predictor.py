import os
import re

import pandas as pd
import streamlit as st
import dagshub

from joblib import load


# ============================================================
# Model configuration
# ============================================================

DAGSHUB_REPO = "/price-range-classification-mlflow"

# File inside DagsHub Storage
REMOTE_MODEL_PATH = "models/model_data.joblib"

# Current file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Local model location
LOCAL_ARTIFACT_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "model_data.joblib"
)


# ============================================================
# Load trained model and preprocessing objects
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model_artifact():
    """
    Load the trained model artifact.

    Local development:
        Load backend/artifacts/model_data.joblib

    Streamlit Cloud:
        Download model_data.joblib from private DagsHub Storage
        and then load it.
    """

    # --------------------------------------------------------
    # 1. Local development
    # --------------------------------------------------------

    if os.path.exists(LOCAL_ARTIFACT_PATH):

        print("Loading model from local artifact.")

        return load(LOCAL_ARTIFACT_PATH)

    # --------------------------------------------------------
    # 2. Streamlit Cloud
    # --------------------------------------------------------

    if "DAGSHUB_USER_TOKEN" not in st.secrets:

        raise RuntimeError(
            "DAGSHUB_USER_TOKEN is missing from "
            "Streamlit Secrets."
        )

    # --------------------------------------------------------
    # Set DagsHub authentication
    # --------------------------------------------------------

    os.environ["DAGSHUB_USER_TOKEN"] = (
        st.secrets["DAGSHUB_USER_TOKEN"]
    )

    # --------------------------------------------------------
    # Create DagsHub Storage client
    # --------------------------------------------------------

    client = dagshub.get_repo_bucket_client(
        DAGSHUB_REPO,
        flavor="boto"
    )

    # --------------------------------------------------------
    # Create local artifacts directory
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(LOCAL_ARTIFACT_PATH),
        exist_ok=True
    )

    # --------------------------------------------------------
    # Download private model from DagsHub Storage
    # --------------------------------------------------------

    print("Downloading model from DagsHub Storage...")

    client.download_file(
        Bucket="price-range-classification-mlflow",
        Key=REMOTE_MODEL_PATH,
        Filename=LOCAL_ARTIFACT_PATH
    )

    print("Model downloaded successfully.")

    # --------------------------------------------------------
    # Load downloaded artifact
    # --------------------------------------------------------

    return load(LOCAL_ARTIFACT_PATH)


# ============================================================
# Load model artifact
# ============================================================

model_data = load_model_artifact()

model = model_data["model"]
model_name = model_data["model_name"]

label_encoders = model_data["label_encoders"]
ohe = model_data["ohe"]
target_encoder = model_data["target_encoder"]

features = model_data["features"]


# ============================================================
# Original mappings used during training
# ============================================================

ZONE_MAPPING = {
    "Rural": 1,
    "Semi-Urban": 2,
    "Urban": 3,
    "Metro": 4
}


INCOME_MAPPING = {
    "<10L": 1,
    "10L - 15L": 2,
    "16L - 25L": 3,
    "26L - 35L": 4,

    # Support both the Streamlit value and
    # the original training-data spelling.
    ">35L": 5,
    "> 35L": 5,

    "Not Reported": 0
}


CONSUME_FREQUENCY_MAPPING = {
    "0-2 times": 1,
    "3-4 times": 2,
    "5-7 times": 3
}


AWARENESS_MAPPING = {
    "0 to 1": 1,
    "2 to 4": 2,
    "above 4": 3
}


# ============================================================
# Age Group
# ============================================================

def get_age_group(age):

    if age <= 25:
        return "18-25"

    elif age <= 35:
        return "26-35"

    elif age <= 45:
        return "36-45"

    elif age <= 55:
        return "46-55"

    elif age <= 70:
        return "56-70"

    else:
        return "70+"


# ============================================================
# Feature Engineering
# ============================================================

def prepare_raw_features(input_data):
    """
    Convert raw Streamlit input into the feature-engineered
    dataframe used before model encoding.
    """

    df = pd.DataFrame([input_data]).copy()

    # --------------------------------------------------------
    # Fix categorical spelling
    # --------------------------------------------------------

    df["current_brand"] = df["current_brand"].replace({
        "newcomer": "Newcomer",
        "Establishd": "Established"
    })

    df["zone"] = df["zone"].replace({
        "urbna": "Urban",
        "Metor": "Metro"
    })

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    df["income_levels"] = (
        df["income_levels"]
        .fillna("Not Reported")
    )

    # Training notebook used:
    # mode = "3-4 times"

    df["consume_frequency(weekly)"] = (
        df["consume_frequency(weekly)"]
        .fillna("3-4 times")
    )

    # --------------------------------------------------------
    # Apply mappings
    # --------------------------------------------------------

    df["zone"] = df["zone"].map(
        ZONE_MAPPING
    )

    df["income_levels"] = df["income_levels"].map(
        INCOME_MAPPING
    )

    df["consume_frequency(weekly)"] = (
        df["consume_frequency(weekly)"]
        .map(CONSUME_FREQUENCY_MAPPING)
    )

    df["awareness_of_other_brands"] = (
        df["awareness_of_other_brands"]
        .map(AWARENESS_MAPPING)
    )

    # --------------------------------------------------------
    # Age Group
    # --------------------------------------------------------

    df["age_group"] = df["age"].apply(
        get_age_group
    )

    # --------------------------------------------------------
    # CF-AB Score
    #
    # consume frequency /
    # (awareness + consume frequency)
    # --------------------------------------------------------

    df["cf_ab_score"] = df.apply(
        lambda row:
            (
                row["consume_frequency(weekly)"]
                /
                (
                    row["awareness_of_other_brands"]
                    +
                    row["consume_frequency(weekly)"]
                )
            )
            if (
                row["awareness_of_other_brands"]
                +
                row["consume_frequency(weekly)"]
            ) != 0
            else 0,
        axis=1
    ).round(2)

    # --------------------------------------------------------
    # ZAS Score
    #
    # zone * income level
    # --------------------------------------------------------

    df["zas_score"] = (
        df["zone"]
        *
        df["income_levels"]
    ).round()

    # --------------------------------------------------------
    # BSI
    #
    # current_brand != Established
    # AND reason is Price or Quality
    # --------------------------------------------------------

    df["bsi"] = (
        (
            (df["current_brand"] != "Established")
            &
            (
                df["reasons_for_choosing_brands"].isin(
                    ["Price", "Quality"]
                )
            )
        )
        .astype(int)
    )

    return df


# ============================================================
# Label Encoding
# ============================================================

def apply_label_encoding(df):
    """
    Apply the exact LabelEncoders fitted during training.
    """

    df = df.copy()

    label_cols = [
        "age_group",
        "income_levels",
        "health_concerns",
        "consume_frequency(weekly)",
        "preferable_consumption_size"
    ]

    for col in label_cols:

        if col not in label_encoders:

            raise KeyError(
                f"Label encoder for '{col}' "
                f"not found in artifact."
            )

        encoder = label_encoders[col]

        value = df[col].iloc[0]

        # LabelEncoder does not support unknown values
        if value not in encoder.classes_:

            raise ValueError(
                f"Unknown value '{value}' "
                f"for feature '{col}'. "
                f"Expected one of: "
                f"{list(encoder.classes_)}"
            )

        df[col] = encoder.transform(
            df[col]
        )

    return df


# ============================================================
# One-Hot Encoding
# ============================================================

def apply_one_hot_encoding(df):
    """
    Apply the fitted OneHotEncoder used during training.
    """

    df = df.copy()

    # After Label Encoding, remaining object/string
    # columns are one-hot encoded.

    categorical_cols = (
        df.select_dtypes(
            include=["object", "str"]
        )
        .columns
        .tolist()
    )

    encoded_array = ohe.transform(
        df[categorical_cols]
    )

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=ohe.get_feature_names_out(
            categorical_cols
        ),
        index=df.index
    )

    # Remove original categorical columns
    df = df.drop(
        columns=categorical_cols
    )

    # Add one-hot encoded columns
    df = pd.concat(
        [df, encoded_df],
        axis=1
    )

    return df


# ============================================================
# Clean Feature Names
# ============================================================

def clean_feature_names(df):
    """
    Apply the same feature-name cleaning used during training.
    """

    df = df.copy()

    df.columns = [
        re.sub(
            r"[^A-Za-z0-9_]+",
            "_",
            col
        ).strip("_")
        for col in df.columns
    ]

    return df


# ============================================================
# Final Feature Preparation
# ============================================================

def prepare_features(input_data):
    """
    Main preprocessing function.

    Input:
        Dictionary containing raw Streamlit inputs.

    Output:
        DataFrame containing exactly the same features
        and feature order used during model training.
    """

    # --------------------------------------------------------
    # 1. Feature Engineering
    # --------------------------------------------------------

    df = prepare_raw_features(
        input_data
    )

    # --------------------------------------------------------
    # 2. Label Encoding
    # --------------------------------------------------------

    df = apply_label_encoding(
        df
    )

    # --------------------------------------------------------
    # 3. One-Hot Encoding
    # --------------------------------------------------------

    df = apply_one_hot_encoding(
        df
    )

    # --------------------------------------------------------
    # 4. Clean Feature Names
    # --------------------------------------------------------

    df = clean_feature_names(
        df
    )

    # --------------------------------------------------------
    # 5. Ensure all training features exist
    # --------------------------------------------------------

    for feature in features:

        if feature not in df.columns:

            df[feature] = 0

    # --------------------------------------------------------
    # 6. Keep exactly the training feature order
    # --------------------------------------------------------

    df = df[features]

    return df


# ============================================================
# Prediction
# ============================================================

def predict_price_range(input_data):
    """
    Predict the price range from raw Streamlit input.
    """

    # Prepare raw input
    X_input = prepare_features(
        input_data
    )

    # Model prediction
    prediction = model.predict(
        X_input
    )

    # Convert encoded target back to
    # original price range
    predicted_price_range = (
        target_encoder
        .inverse_transform(prediction)[0]
    )

    return predicted_price_range