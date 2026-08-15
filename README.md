# AQUA-TWIN-AI

**AI-Driven Digital Twin-Based Mucilage Monitoring and Environmental Decision Support**

AQUA-TWIN-AI is an Artificial Intelligence-driven Digital Twin framework developed for mucilage-risk assessment and environmental monitoring in the Marmara Basin. The system combines AI-based mucilage prediction with environmental monitoring to support coastal environmental assessment and decision support.

## Key Capabilities

- AI-based mucilage prediction and risk classification
- Environmental monitoring using integrated Marmara Sea observations
- Digital Twin-based representation of environmental conditions
- Visualization of temperature, dissolved oxygen, turbidity, and chlorophyll-a
- Risk-level assessment for environmental decision support

## Project Structure

```text
AQUA-TWIN-AI/
├── app.py
├── environmental_assessment.csv
├── mucilage_prediction.csv
│
└── aqua_twin_models/
    ├── best_model.pkl
    ├── features.pkl
    ├── scaler.pkl
    └── model_comparison.csv
```

## Main Files

### `app.py`

Main Streamlit application containing the AQUA-TWIN-AI dashboard, environmental monitoring interface, and mucilage-risk prediction functionality.

### `mucilage_prediction.csv`

Dataset containing the mucilage prediction data used in the AI-based risk assessment workflow.

### `environmental_assessment.csv`

Environmental monitoring dataset containing observations used for environmental assessment and dashboard visualization.

### `aqua_twin_models/best_model.pkl`

Saved trained machine-learning model used by the application for mucilage prediction.

### `aqua_twin_models/features.pkl`

Stores the feature information required by the trained model during prediction.

### `aqua_twin_models/scaler.pkl`

Saved preprocessing scaler used to transform prediction inputs consistently with the model training process.

### `aqua_twin_models/model_comparison.csv`

Contains the performance comparison of the evaluated machine-learning models.

## Dashboard

The Streamlit dashboard provides mucilage prediction together with environmental monitoring information, including temperature, dissolved oxygen, turbidity, and chlorophyll-a observations.

## Model Performance

The final evaluated model achieved **98% accuracy** for the implemented mucilage classification task.

## Technology

- Python
- Streamlit
- Pandas
- Scikit-learn
- Machine Learning
- Environmental Data Analytics

## Application

AQUA-TWIN-AI is designed to support integrated environmental monitoring, mucilage-risk assessment, and evidence-based environmental decision support for the Marmara Basin.
