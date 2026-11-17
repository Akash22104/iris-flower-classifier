import json
import pickle
import random
from pathlib import Path
from typing import Any

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris

import ui_art

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Iris flower classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

FEATURES = {
    # session-state key: (label, short label)
    "sepal_length": ("Sepal length (cm)", "Sepal L"),
    "sepal_width": ("Sepal width (cm)", "Sepal W"),
    "petal_length": ("Petal length (cm)", "Petal L"),
    "petal_width": ("Petal width (cm)", "Petal W"),
}

DEFAULT_FEATURE_RANGES = {
    "sepal_length": {"min": 4.0, "max": 8.0, "default": 5.8},
    "sepal_width": {"min": 2.0, "max": 4.5, "default": 3.0},
    "petal_length": {"min": 1.0, "max": 7.0, "default": 4.0},
    "petal_width": {"min": 0.1, "max": 2.5, "default": 1.2},
}

DEFAULT_MODEL_FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

DEFAULT_TARGET_NAMES = ["setosa", "versicolor", "virginica"]

PRESETS = {
    "setosa": [5.0, 3.4, 1.5, 0.2],
    "versicolor": [5.9, 2.8, 4.3, 1.3],
    "virginica": [6.6, 3.0, 5.6, 2.1],
}


# --------------------------------------------------
# LOADERS
# --------------------------------------------------

@st.cache_resource
def load_model(format_type="joblib"):
    try:
        if format_type == "joblib":
            model_path = MODEL_DIR / "iris_model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            return joblib.load(model_path)

        model_path = MODEL_DIR / "iris_model.pickle"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with open(model_path, "rb") as model_file:
            return pickle.load(model_file)

    except Exception as error:
        st.error(f"Error loading model: {error}")
        return None


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


@st.cache_data
def load_dataset():
    """The 150-sample Iris dataset (bundled with scikit-learn, no download)."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = list(FEATURES) + ["target"]
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    df["sample_id"] = range(len(df))
    return df.drop(columns="target")


@st.cache_data
def load_css():
    return ui_art.base_css()


@st.cache_data
def load_flower_svg(species):
    return ui_art.flower_svg(species)


model_info = load_json_file("model_info.json") or {}
feature_ranges = load_json_file("feature_ranges.json") or DEFAULT_FEATURE_RANGES
target_names = model_info.get("target_names", DEFAULT_TARGET_NAMES)
dataset = load_dataset()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

for feature_key in FEATURES:
    st.session_state.setdefault(
        feature_key, round(float(feature_ranges[feature_key]["default"]), 1)
    )
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_species", None)


def set_measurements(values):
    for feature_key, value in zip(FEATURES, values):
        low = float(feature_ranges[feature_key]["min"])
        high = float(feature_ranges[feature_key]["max"])
        st.session_state[feature_key] = round(min(max(float(value), low), high), 1)


def apply_preset(species):
    set_measurements(PRESETS[species])


def apply_random_sample():
    row = dataset.sample(1, random_state=random.randint(0, 10_000)).iloc[0]
    set_measurements([row[key] for key in FEATURES])
    st.toast(
        f"Loaded sample #{row['sample_id']} (a real {row['species']})",
        icon=":material/casino:",
    )


def reset_measurements():
    set_measurements(
        [feature_ranges[key]["default"] for key in FEATURES]
    )


def load_clicked_flower():
    """Copy a flower clicked in the explorer chart into the sliders."""
    event = st.session_state.get("explore_chart")
    if not event:
        return
    picked = event["selection"].get("pick", [])
    if not picked:
        return
    sample_id = picked[0].get("sample_id")
    if sample_id is None:
        return
    row = dataset.loc[dataset["sample_id"] == sample_id].iloc[0]
    set_measurements([row[key] for key in FEATURES])
    st.toast(
        f"Loaded sample #{sample_id}, a real {row['species']}",
        icon=":material/ads_click:",
    )


# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.html(load_css())


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header(":material/tune: Controls")

    model_format = st.segmented_control(
        "Model format",
        ["joblib", "pickle"],
        default="joblib",
        required=True,
        width="stretch",
    )
    model = load_model(model_format)

    if model is not None:
        st.badge("Model loaded", icon=":material/check_circle:", color="green")
    else:
        st.badge("Model unavailable", icon=":material/error:", color="red")

    st.subheader("Quick fill")
    st.caption("Jump to a typical flower of each species.")
    for species in PRESETS:
        st.button(
            species.title(),
            key=f"preset_{species}",
            icon=":material/local_florist:",
            on_click=apply_preset,
            args=(species,),
            width="stretch",
        )

    with st.container(horizontal=True):
        st.button(
            "Surprise me",
            key="surprise",
            icon=":material/casino:",
            on_click=apply_random_sample,
            width="stretch",
        )
        st.button(
            "Reset",
            key="reset",
            icon=":material/restart_alt:",
            on_click=reset_measurements,
            width="stretch",
        )

    st.divider()
    st.caption(
        f"{model_info.get('model_type', 'Model')} · "
        f"test accuracy {model_info.get('accuracy', 0):.0%}"
    )


# --------------------------------------------------
# HERO BANNER
# --------------------------------------------------

with st.container(key="hero"):
    st.title("Iris flower classifier")
    st.markdown(
        "Move the sliders and watch the model guess the species in real "
        "time. Then explore the data it learned from."
    )


# --------------------------------------------------
# PREDICTION (runs live on every slider change)
# --------------------------------------------------

def model_feature_names():
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return model_info.get("feature_names", DEFAULT_MODEL_FEATURES)


def predict_frame(frame):
    """Return class probabilities for a frame with FEATURES columns."""
    renamed = frame[list(FEATURES)].copy()
    renamed.columns = model_feature_names()
    return model.predict_proba(renamed)


def class_names():
    return [
        target_names[i] if i < len(target_names) else str(value)
        for i, value in enumerate(model.classes_)
    ]


measurements = {key: st.session_state[key] for key in FEATURES}

prediction: dict[str, Any] | None = None
if model is not None:
    try:
        probs = predict_frame(pd.DataFrame([measurements]))[0]
        names = class_names()
        order = np.argsort(probs)[::-1]
        prediction = {
            "species": names[order[0]],
            "confidence": float(probs[order[0]]),
            "runner_up": names[order[1]],
            "margin": float(probs[order[0]] - probs[order[1]]),
            "probs": pd.DataFrame(
                {"species": names, "probability": probs.astype(float)}
            ),
        }
    except Exception as error:
        st.error(f"Error making prediction: {error}")

if prediction:
    if (
        st.session_state.last_species is not None
        and st.session_state.last_species != prediction["species"]
    ):
        st.toast(
            f"Now looks like **{prediction['species'].title()}**",
            icon=":material/auto_awesome:",
        )
    st.session_state.last_species = prediction["species"]
    st.html(ui_art.result_card_css(prediction["species"]))


def save_to_history():
    if not prediction:
        return
    st.session_state.history.append(
        {
            **{FEATURES[key][0]: value for key, value in measurements.items()},
            "Predicted species": prediction["species"].title(),
            "Confidence": prediction["confidence"],
        }
    )
    st.toast("Saved to history", icon=":material/bookmark_added:")


# --------------------------------------------------
# TABS
# --------------------------------------------------

predict_tab, explore_tab, insights_tab, history_tab = st.tabs(
    [
        ":material/local_florist: Predict",
        ":material/scatter_plot: Explore the data",
        ":material/insights: Model insights",
        f":material/history: History ({len(st.session_state.history)})",
    ]
)

species_scale = alt.Scale(
    domain=DEFAULT_TARGET_NAMES,
    range=[ui_art.species_color(name) for name in DEFAULT_TARGET_NAMES],
)

# ---------------- Predict ----------------
with predict_tab:
    inputs_col, result_col = st.columns([1, 1], gap="large")

    with inputs_col:
        with st.container(key="panel_inputs"):
            st.subheader(":material/straighten: Measurements")
            for feature_key, (label, _) in FEATURES.items():
                st.slider(
                    label,
                    min_value=float(feature_ranges[feature_key]["min"]),
                    max_value=float(feature_ranges[feature_key]["max"]),
                    step=0.1,
                    key=feature_key,
                )

            ratio = measurements["petal_length"] / max(
                measurements["petal_width"], 0.1
            )
            with st.container(horizontal=True):
                st.metric(
                    "Petal area",
                    f"{measurements['petal_length'] * measurements['petal_width']:.1f} cm²",
                    border=True,
                )
                st.metric("Petal shape", f"{ratio:.1f} : 1", border=True)

    with result_col:
        with st.container(key="panel_result"):
            if prediction is None:
                st.error(
                    "The model could not be loaded. Check the models folder.",
                    title="No model",
                )
            else:
                species = prediction["species"]
                style = ui_art.species_style(species)

                art_col, text_col = st.columns([2, 3], vertical_alignment="center")
                with art_col:
                    st.image(
                        load_flower_svg(species),
                        width="stretch",
                        alt=f"Illustration of an Iris {species} flower",
                    )
                with text_col:
                    st.caption("The model thinks this is")
                    st.header(f"Iris {species}")
                    st.markdown(style["blurb"])
                    st.metric(
                        "Confidence",
                        f"{prediction['confidence']:.0%}",
                        delta=f"{prediction['margin']:.0%} ahead of "
                        f"{prediction['runner_up']}",
                        delta_arrow="off",
                        delta_color="off",
                    )

                bars = (
                    alt.Chart(prediction["probs"])
                    .mark_bar(cornerRadiusEnd=4, size=22)
                    .encode(
                        x=alt.X(
                            "probability:Q",
                            scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(format="%", title=None),
                        ),
                        y=alt.Y("species:N", title=None, sort=DEFAULT_TARGET_NAMES),
                        color=alt.Color("species:N", scale=species_scale, legend=None),
                        tooltip=[
                            alt.Tooltip("species:N", title="Species"),
                            alt.Tooltip("probability:Q", format=".1%", title="Probability"),
                        ],
                    )
                    .properties(height=130)
                )
                labels = bars.mark_text(align="left", dx=6, fontWeight="bold").encode(
                    text=alt.Text("probability:Q", format=".0%"),
                    color=alt.value("#e2e8f0"),
                )
                st.altair_chart(
                    bars + labels,
                    alt="Bar chart of predicted probability for each iris species",
                )

                st.button(
                    "Save this prediction",
                    key="save_prediction",
                    icon=":material/bookmark_add:",
                    type="primary",
                    on_click=save_to_history,
                    width="stretch",
                )

# ---------------- Explore ----------------
with explore_tab:
    with st.container(key="panel_explore"):
        st.subheader(":material/scatter_plot: Where does your flower sit?")
        st.caption(
            "Each dot is one of the 150 flowers the model was trained on. "
            "The star is your flower. Click any dot to load its measurements."
        )

        with st.container(horizontal=True, vertical_alignment="bottom"):
            x_feature = st.segmented_control(
                "X axis",
                list(FEATURES),
                default="petal_length",
                required=True,
                format_func=lambda key: FEATURES[key][1],
                key="x_feature",
            )
            y_feature = st.segmented_control(
                "Y axis",
                list(FEATURES),
                default="petal_width",
                required=True,
                format_func=lambda key: FEATURES[key][1],
                key="y_feature",
            )
            shown_species = st.pills(
                "Species",
                DEFAULT_TARGET_NAMES,
                selection_mode="multi",
                default=DEFAULT_TARGET_NAMES,
                format_func=str.title,
                key="shown_species",
            )

        visible = dataset[dataset["species"].isin(shown_species or [])]
        pick = alt.selection_point(name="pick", fields=["sample_id"], on="click")

        dots = (
            alt.Chart(visible)
            .mark_circle(size=90, stroke="#0b1220", strokeWidth=1)
            .encode(
                x=alt.X(f"{x_feature}:Q", title=FEATURES[x_feature][0],
                        scale=alt.Scale(zero=False)),
                y=alt.Y(f"{y_feature}:Q", title=FEATURES[y_feature][0],
                        scale=alt.Scale(zero=False)),
                color=alt.Color("species:N", scale=species_scale, title="Species"),
                opacity=alt.condition(pick, alt.value(0.9), alt.value(0.35)),
                tooltip=["sample_id", "species", *FEATURES],
            )
            .add_params(pick)
        )
        you = (
            alt.Chart(pd.DataFrame([measurements]))
            .mark_point(shape="diamond", size=420, filled=True,
                        color="#f8fafc", stroke="#0b1220", strokeWidth=2)
            .encode(x=f"{x_feature}:Q", y=f"{y_feature}:Q",
                    tooltip=[alt.Tooltip(key, title=FEATURES[key][0]) for key in FEATURES])
        )
        st.altair_chart(
            dots + you,
            key="explore_chart",
            on_select=load_clicked_flower,
            selection_mode="pick",
            height=440,
            alt="Scatter plot of the iris dataset with your flower highlighted",
        )

        with st.container(horizontal=True):
            for name in DEFAULT_TARGET_NAMES:
                subset = dataset[dataset["species"] == name]
                st.metric(
                    f"{name.title()} · avg {FEATURES[x_feature][1]}",
                    f"{subset[x_feature].mean():.2f} cm",
                    delta=f"Yours: {measurements[x_feature] - subset[x_feature].mean():+.2f} cm",
                    delta_color="off",
                    delta_arrow="off",
                    border=True,
                )

# ---------------- Insights ----------------
with insights_tab:
    if model is None:
        st.info("Load a model to see insights.", icon=":material/info:")
    else:
        importance_col, sensitivity_col = st.columns([2, 3], gap="large")

        with importance_col, st.container(key="panel_importance"):
            st.subheader(":material/leaderboard: What the model looks at")
            if hasattr(model, "feature_importances_"):
                importance = pd.DataFrame(
                    {
                        "feature": [FEATURES[key][0] for key in FEATURES],
                        "importance": model.feature_importances_,
                    }
                )
                st.altair_chart(
                    alt.Chart(importance)
                    .mark_bar(cornerRadiusEnd=4, color="#3b82f6")
                    .encode(
                        x=alt.X("importance:Q", axis=alt.Axis(format="%"), title=None),
                        y=alt.Y("feature:N", sort="-x", title=None),
                        tooltip=[alt.Tooltip("importance:Q", format=".1%")],
                    )
                    .properties(height=220),
                    alt="Bar chart of feature importance in the random forest",
                )
                st.caption(
                    "Petal measurements do most of the work, which is why "
                    "the petal sliders change the prediction fastest."
                )
            else:
                st.caption("This model type doesn't report feature importances.")

        with sensitivity_col, st.container(key="panel_sensitivity"):
            st.subheader(":material/swap_horiz: What if…?")
            vary = st.segmented_control(
                "Vary one measurement, keep the others as they are",
                list(FEATURES),
                default="petal_length",
                required=True,
                format_func=lambda key: FEATURES[key][0],
                key="vary_feature",
            )
            low = float(feature_ranges[vary]["min"])
            high = float(feature_ranges[vary]["max"])
            sweep = pd.DataFrame([measurements] * 60)
            sweep[vary] = np.linspace(low, high, 60)
            sweep_probs = predict_frame(sweep)
            curves = pd.DataFrame(sweep_probs, columns=class_names())
            curves[vary] = sweep[vary].values
            curves = curves.melt(id_vars=vary, var_name="species", value_name="probability")

            area = (
                alt.Chart(curves)
                .mark_area(opacity=0.85, interpolate="step-after")
                .encode(
                    x=alt.X(f"{vary}:Q", title=FEATURES[vary][0],
                            scale=alt.Scale(domain=[low, high])),
                    y=alt.Y("probability:Q", stack="normalize",
                            axis=alt.Axis(format="%"), title="Probability"),
                    color=alt.Color("species:N", scale=species_scale, title="Species"),
                    tooltip=["species", alt.Tooltip("probability:Q", format=".0%")],
                )
            )
            marker = (
                alt.Chart(pd.DataFrame({vary: [measurements[vary]]}))
                .mark_rule(color="#f8fafc", strokeWidth=3, strokeDash=[6, 4])
                .encode(x=f"{vary}:Q")
            )
            st.altair_chart(
                (area + marker).properties(height=280),
                alt="Stacked area chart of how species probability changes "
                "as one measurement varies",
            )
            st.caption("The dashed line marks your current value.")

# ---------------- History ----------------
with history_tab, st.container(key="panel_history"):
    st.subheader(":material/history: Saved predictions")
    if not st.session_state.history:
        st.caption(
            "Nothing saved yet. Use **Save this prediction** on the Predict tab."
        )
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(
            history_df,
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", format="percent", min_value=0, max_value=1
                ),
            },
            alt="Table of saved predictions",
        )
        with st.container(horizontal=True):
            st.download_button(
                "Download CSV",
                history_df.to_csv(index=False),
                file_name="iris_predictions.csv",
                mime="text/csv",
                icon=":material/download:",
            )
            st.button(
                "Clear history",
                key="clear_history",
                icon=":material/delete:",
                on_click=lambda: st.session_state.history.clear(),
            )


# --------------------------------------------------
# ABOUT
# --------------------------------------------------

with st.expander("About the Iris dataset", icon=":material/menu_book:"):
    st.markdown(
        "The Iris dataset is a classic machine learning dataset with 150 "
        "flowers from three species, measured by the botanist Edgar Anderson "
        "and made famous by Ronald Fisher in 1936."
    )
    with st.container(horizontal=True):
        for name in DEFAULT_TARGET_NAMES:
            with st.container(border=True):
                st.image(
                    load_flower_svg(name),
                    width=120,
                    alt=f"Illustration of an Iris {name} flower",
                )
                st.markdown(f"**Iris {name}**")
                st.caption(ui_art.species_style(name)["blurb"])

st.caption("Built with Python, Streamlit, pandas and scikit-learn")
