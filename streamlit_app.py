import json
import pickle

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="Iris",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model(format_type="joblib"):
    try:
        if format_type == "joblib":
            return joblib.load("models/iris_model.joblib")
        with open("models/iris_model.pickle", "rb") as model_file:
            return pickle.load(model_file)
    except Exception as error:
        st.error(f"Error loading model: {error}")
        return None


@st.cache_data
def load_json_file(path):
    try:
        with open(path, "r") as data_file:
            return json.load(data_file)
    except Exception as error:
        st.error(f"Error loading {path}: {error}")
        return None


model_info = load_json_file("models/model_info.json")
feature_ranges = load_json_file("models/feature_ranges.json") or {
    "sepal_length": {"min": 4.0, "max": 8.0, "default": 5.8},
    "sepal_width": {"min": 2.0, "max": 4.5, "default": 3.0},
    "petal_length": {"min": 1.0, "max": 7.0, "default": 4.0},
    "petal_width": {"min": 0.1, "max": 2.5, "default": 1.2},
}

with st.sidebar:
    st.title("Settings")
    model_format = st.radio("Model Format", ["joblib", "pickle"])
    model = load_model(model_format)
    if st.button("Reload Model"):
        load_model.clear()
        model = load_model(model_format)
        if model is not None:
            st.success(f"Model loaded from {model_format} format.")

    st.divider()
    st.subheader("Model Information")
    if model_info:
        st.write(f"**Type:** {model_info.get('model_type', 'RandomForest')}")
        st.write(f"**Accuracy:** {model_info.get('accuracy', 0):.1%}")
        st.write(f"**Features:** {len(model_info.get('feature_names', []))}")
        st.write(f"**Classes:** {len(model_info.get('target_names', []))}")

st.title("Iris Flower Classification")
st.write("Adjust the sliders to enter flower measurements and make a prediction.")

col1, col2 = st.columns([2, 1])
with col1:
    st.header("Input Features")
    sepal_length_range = feature_ranges["sepal_length"]
    sepal_length = st.slider(
        "Sepal Length (cm)",
        min_value=float(sepal_length_range["min"]),
        max_value=float(sepal_length_range["max"]),
        value=float(sepal_length_range["default"]),
        step=0.1,
    )
    sepal_width_range = feature_ranges["sepal_width"]
    sepal_width = st.slider(
        "Sepal Width (cm)",
        min_value=float(sepal_width_range["min"]),
        max_value=float(sepal_width_range["max"]),
        value=float(sepal_width_range["default"]),
        step=0.1,
    )
    petal_length_range = feature_ranges["petal_length"]
    petal_length = st.slider(
        "Petal Length (cm)",
        min_value=float(petal_length_range["min"]),
        max_value=float(petal_length_range["max"]),
        value=float(petal_length_range["default"]),
        step=0.1,
    )
    petal_width_range = feature_ranges["petal_width"]
    petal_width = st.slider(
        "Petal Width (cm)",
        min_value=float(petal_width_range["min"]),
        max_value=float(petal_width_range["max"]),
        value=float(petal_width_range["default"]),
        step=0.1,
    )

with col2:
    st.header("Current Values")
    features_df = pd.DataFrame({
        "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
        "Value (cm)": [sepal_length, sepal_width, petal_length, petal_width],
    })
    st.dataframe(features_df, hide_index=True, use_container_width=True)

if st.button("Predict Species", type="primary", use_container_width=True):
    if model is None or model_info is None:
        st.error("Model could not be loaded. Check the files in the models directory.")
    else:
        try:
            input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(input_features)[0]
            probabilities = model.predict_proba(input_features)[0]
            target_names = model_info["target_names"]
            st.subheader(f"Predicted Species: {target_names[prediction]}")
            st.write("Confidence scores")
            for species, probability in zip(target_names, probabilities):
                st.write(f"{species}: {probability:.1%}")
                st.progress(float(probability))
        except Exception as error:
            st.error(f"Error making prediction: {error}")

with st.expander("About the Iris Dataset"):
    st.write("The Iris dataset contains 150 samples, four measurements, and three flower species.")

st.markdown("---")
st.caption("Built with Streamlit and Scikit-learn")
