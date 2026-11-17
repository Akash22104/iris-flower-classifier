
import json
import pickle
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# DEFAULT CONFIGURATION
# --------------------------------------------------

DEFAULT_FEATURE_RANGES = {
    "sepal_length": {"min": 4.0, "max": 8.0, "default": 5.8},
    "sepal_width": {"min": 2.0, "max": 4.5, "default": 3.0},
    "petal_length": {"min": 1.0, "max": 7.0, "default": 4.0},
    "petal_width": {"min": 0.1, "max": 2.5, "default": 1.2},
}

DEFAULT_FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

DEFAULT_TARGET_NAMES = [
    "Setosa",
    "Versicolor",
    "Virginica",
]


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(format_type="joblib"):

    try:
        if format_type == "joblib":
            model_path = MODEL_DIR / "iris_model.joblib"

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {model_path}"
                )

            return joblib.load(model_path)

        elif format_type == "pickle":
            model_path = MODEL_DIR / "iris_model.pickle"

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {model_path}"
                )

            with open(model_path, "rb") as model_file:
                return pickle.load(model_file)

    except Exception as error:
        st.error(f"Error loading model: {error}")
        return None


# --------------------------------------------------
# LOAD JSON FILES
# --------------------------------------------------

@st.cache_data
def load_json_file(filename):

    try:
        file_path = MODEL_DIR / filename

        if not file_path.exists():
            return None

        with open(file_path, "r") as file:
            return json.load(file)

    except Exception as error:
        st.error(f"Error loading {filename}: {error}")
        return None


model_info = load_json_file("model_info.json") or {}

feature_ranges = (
    load_json_file("feature_ranges.json")
    or DEFAULT_FEATURE_RANGES
)

target_names = model_info.get(
    "target_names",
    DEFAULT_TARGET_NAMES
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("⚙️ Settings")

    model_format = st.radio(
        "Model Format",
        ["joblib", "pickle"]
    )

    model = load_model(model_format)

    if st.button("🔄 Reload Model", width="stretch"):

        load_model.clear()
        model = load_model(model_format)

        if model is not None:
            st.success("Model loaded successfully!")

    st.divider()

    st.subheader("📊 Model Information")

    if model_info:

        st.write(
            f"**Type:** {model_info.get('model_type', 'RandomForest')}"
        )

        accuracy = model_info.get("accuracy")

        if accuracy is not None:
            st.write(f"**Accuracy:** {accuracy:.1%}")

        st.write(
            f"**Features:** {len(model_info.get('feature_names', DEFAULT_FEATURE_NAMES))}"
        )

        st.write(
            f"**Classes:** {len(target_names)}"
        )

    else:
        st.info("Using default model information.")


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

st.title("🌸 Iris Flower Classification")

st.write(
    "Adjust the sliders to enter flower measurements "
    "and predict the Iris flower species using Machine Learning."
)

col1, col2 = st.columns([2, 1])


# --------------------------------------------------
# INPUT FEATURES
# --------------------------------------------------

with col1:

    st.header("🌱 Input Features")

    sepal_length = st.slider(
        "Sepal Length (cm)",
        min_value=float(feature_ranges["sepal_length"]["min"]),
        max_value=float(feature_ranges["sepal_length"]["max"]),
        value=float(feature_ranges["sepal_length"]["default"]),
        step=0.1,
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        min_value=float(feature_ranges["sepal_width"]["min"]),
        max_value=float(feature_ranges["sepal_width"]["max"]),
        value=float(feature_ranges["sepal_width"]["default"]),
        step=0.1,
    )

    petal_length = st.slider(
        "Petal Length (cm)",
        min_value=float(feature_ranges["petal_length"]["min"]),
        max_value=float(feature_ranges["petal_length"]["max"]),
        value=float(feature_ranges["petal_length"]["default"]),
        step=0.1,
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        min_value=float(feature_ranges["petal_width"]["min"]),
        max_value=float(feature_ranges["petal_width"]["max"]),
        value=float(feature_ranges["petal_width"]["default"]),
        step=0.1,
    )


# --------------------------------------------------
# DISPLAY CURRENT VALUES
# --------------------------------------------------

with col2:

    st.header("📋 Current Values")

    features_df = pd.DataFrame({
        "Feature": [
            "Sepal Length",
            "Sepal Width",
            "Petal Length",
            "Petal Width",
        ],
        "Value (cm)": [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
        ],
    })

    st.dataframe(
        features_df,
        hide_index=True,
        width="stretch"
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "🔍 Predict Species",
    type="primary",
    width="stretch"
):

    if model is None:

        st.error(
            "Model could not be loaded. "
            "Please check your models directory."
        )

    else:

        try:

            # Input feature values
            input_values = [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width,
            ]

            # Use the exact feature names from training
            if hasattr(model, "feature_names_in_"):
                feature_names = list(model.feature_names_in_)

            else:
                feature_names = model_info.get(
                    "feature_names",
                    DEFAULT_FEATURE_NAMES
                )

            if len(feature_names) != len(input_values):
                raise ValueError(
                    "The number of model features does not match "
                    "the input measurements."
                )

            # Create DataFrame instead of NumPy array
            input_features = pd.DataFrame(
                [input_values],
                columns=feature_names
            )

            # Prediction
            prediction = model.predict(input_features)[0]

            probabilities = model.predict_proba(
                input_features
            )[0]

            # Map model classes to species names
            model_classes = list(model.classes_)

            class_name_map = {
                class_value: target_names[index]
                if index < len(target_names)
                else str(class_value)
                for index, class_value in enumerate(model_classes)
            }

            predicted_species = class_name_map.get(
                prediction,
                str(prediction)
            )

            # Display result
            st.divider()

            st.success(
                f"🌸 Predicted Species: {predicted_species}"
            )

            st.subheader("📊 Prediction Confidence")

            for class_value, probability in zip(
                model_classes,
                probabilities
            ):

                species = class_name_map.get(
                    class_value,
                    str(class_value)
                )

                st.write(
                    f"**{species}: {probability:.2%}**"
                )

                st.progress(float(probability))

        except Exception as error:

            st.error(
                f"Error making prediction: {error}"
            )


# --------------------------------------------------
# ABOUT DATASET
# --------------------------------------------------

with st.expander("ℹ️ About the Iris Dataset"):

    st.write(
        """
        The Iris dataset is a popular Machine Learning dataset
        containing 150 samples and three flower species.

        **Flower species:**
        - Setosa
        - Versicolor
        - Virginica

        **Input measurements:**
        - Sepal Length
        - Sepal Width
        - Petal Length
        - Petal Width

        This application uses a trained Machine Learning
        classification model to predict the flower species.
        """
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Python, Streamlit, Pandas and Scikit-learn 🚀"
)
