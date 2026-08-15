
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AQUA-TWIN-AI",
    page_icon="🌊",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load(
    "aqua_twin_models/best_model.pkl"
)

scaler = joblib.load(
    "aqua_twin_models/scaler.pkl"
)

features = joblib.load(
    "aqua_twin_models/features.pkl"
)

results = pd.read_csv(
    "aqua_twin_models/model_comparison.csv"
)

# Environmental dataset
environmental_df = pd.read_csv(
    "environmental_assessment.csv"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🌊 AQUA-TWIN-AI")

st.subheader(
    "AI-Driven Digital Twin-Based Mucilage Monitoring "
    "and Environmental Decision Support"
)

st.markdown(
    """
    **AQUA-TWIN-AI** integrates AI-based mucilage detection
    with environmental monitoring to support coastal
    environmental assessment and decision support.
    """
)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("🔬 Mucilage Prediction")

st.sidebar.write(
    "Enter Sentinel-1 VV/VH observations."
)

vv = st.sidebar.number_input(
    features[0],
    value=-25.0,
    step=0.01
)

vh = st.sidebar.number_input(
    features[1],
    value=-32.0,
    step=0.01
)

predict_button = st.sidebar.button(
    "🚀 Predict Mucilage Risk",
    use_container_width=True
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_button:

    input_data = pd.DataFrame(
        [[vv, vh]],
        columns=features
    )

    # Determine whether scaler is required
    model_name = results.iloc[0]["Model"]

    if model_name in [
        "Logistic Regression",
        "SVM"
    ]:

        input_model = scaler.transform(input_data)

    else:

        input_model = input_data

    prediction = model.predict(input_model)[0]

    probability = model.predict_proba(
        input_model
    )[0][1]

    # Risk interpretation
    if probability < 0.33:
        risk = "LOW"
    elif probability < 0.66:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    # --------------------------------------------------
    # RESULT CARDS
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Mucilage Prediction",
            "MUCILAGE" if prediction == 1 else "CLEAN"
        )

    with col2:

        st.metric(
            "Mucilage Probability",
            f"{probability*100:.2f}%"
        )

    with col3:

        st.metric(
            "Risk Level",
            risk
        )

    st.divider()

    # --------------------------------------------------
    # PROBABILITY CHART
    # --------------------------------------------------

    st.subheader("Prediction Probability")

    probability_df = pd.DataFrame({
        "Class": ["Clean", "Mucilage"],
        "Probability": [
            1-probability,
            probability
        ]
    })

    st.bar_chart(
        probability_df.set_index("Class")
    )

# --------------------------------------------------
# ENVIRONMENTAL MONITORING
# --------------------------------------------------

st.divider()

st.header("🌍 Environmental Monitoring")

st.write(
    "Environmental observations from the integrated "
    "Marmara Sea monitoring dataset."
)

# Find important environmental variables

temp_col = next(
    (
        c for c in environmental_df.columns
        if "Temp" in c and "CTD" in c
    ),
    None
)

do_col = next(
    (
        c for c in environmental_df.columns
        if c.startswith("DO [mg/l]")
    ),
    None
)

turbidity_col = next(
    (
        c for c in environmental_df.columns
        if "Turbidity" in c
    ),
    None
)

chl_col = next(
    (
        c for c in environmental_df.columns
        if "Chl a" in c
    ),
    None
)

# Metrics

m1, m2, m3, m4 = st.columns(4)

with m1:
    if temp_col:
        st.metric(
            "Mean Temperature",
            f"{environmental_df[temp_col].mean():.2f} °C"
        )

with m2:
    if do_col:
        st.metric(
            "Mean Dissolved Oxygen",
            f"{environmental_df[do_col].mean():.2f} mg/L"
        )

with m3:
    if turbidity_col:
        st.metric(
            "Mean Turbidity",
            f"{environmental_df[turbidity_col].mean():.2f}"
        )

with m4:
    if chl_col:
        st.metric(
            "Mean Chlorophyll-a",
            f"{environmental_df[chl_col].mean():.2f}"
        )

# --------------------------------------------------
# ENVIRONMENTAL CHARTS
# --------------------------------------------------

st.subheader("Environmental Profiles")

chart_columns = {}

if temp_col:
    chart_columns["Temperature"] = temp_col

if do_col:
    chart_columns["Dissolved Oxygen"] = do_col

if turbidity_col:
    chart_columns["Turbidity"] = turbidity_col

if chl_col:
    chart_columns["Chlorophyll-a"] = chl_col

for label, column in chart_columns.items():

    chart_df = environmental_df[
        [column]
    ].dropna().reset_index(drop=True)

    chart_df.columns = [label]

    st.line_chart(chart_df)

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.divider()

st.header("🤖 AI Model Performance")

st.dataframe(
    results.style.format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1-Score": "{:.4f}",
        "ROC-AUC": "{:.4f}"
    }),
    use_container_width=True
)

# --------------------------------------------------
# BEST MODEL
# --------------------------------------------------

best = results.iloc[0]

st.success(
    f"Best performing model: {best['Model']} "
    f"with F1-Score = {best['F1-Score']:.4f}"
)

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------

with st.expander("📊 Environmental Dataset Preview"):

    st.dataframe(
        environmental_df.head(100),
        use_container_width=True
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AQUA-TWIN-AI | AI-driven environmental monitoring "
    "and mucilage decision-support prototype"
)
