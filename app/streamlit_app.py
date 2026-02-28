"""HeartHealthML - Patient Risk Assessment Dashboard.

A Streamlit application for heart disease risk prediction.
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.features.build_features import engineer_features
from src.models.predict import load_production_model, predict_single

# Page configuration
st.set_page_config(
    page_title="HeartHealthML",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for branding
st.markdown(
    """
    <style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }

    /* Risk level cards */
    .risk-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .risk-medium {
        background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%);
        color: white;
    }
    .risk-high {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }

    /* Metric styling */
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }

    /* Sidebar styling */
    .sidebar-header {
        color: #667eea;
        font-weight: bold;
        margin-bottom: 1rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #888;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the production model (cached)."""
    try:
        model, preprocessor = load_production_model()
        return model, preprocessor, None
    except Exception as e:
        return None, None, str(e)


def get_risk_color(risk_level: str) -> str:
    """Get color based on risk level."""
    colors = {
        "low": "#38ef7d",
        "medium": "#F2994A",
        "high": "#eb3349",
    }
    return colors.get(risk_level, "#888")


def render_header():
    """Render the main header."""
    st.markdown(
        """
        <div class="main-header">
            <h1>❤️ HeartHealthML</h1>
            <p>AI-Powered Cardiac Risk Assessment Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render sidebar with patient input form."""
    st.sidebar.markdown("## 📋 Patient Information")
    st.sidebar.markdown("---")

    # Demographics
    st.sidebar.markdown("### Demographics")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
    with col2:
        sex = st.selectbox("Sex", options=["Female", "Male"])

    # Vital Signs
    st.sidebar.markdown("### Vital Signs")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        trestbps = st.number_input(
            "Resting BP (mmHg)", min_value=80, max_value=220, value=140
        )
    with col2:
        thalach = st.number_input(
            "Max Heart Rate", min_value=60, max_value=220, value=150
        )

    # Lab Results
    st.sidebar.markdown("### Lab Results")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        chol = st.number_input(
            "Cholesterol (mg/dl)", min_value=100, max_value=600, value=250
        )
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120", options=["No", "Yes"])

    # Clinical Findings
    st.sidebar.markdown("### Clinical Findings")

    cp = st.selectbox(
        "Chest Pain Type",
        options=[
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic",
        ],
    )

    restecg = st.selectbox(
        "Resting ECG",
        options=[
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy",
        ],
    )

    exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])

    oldpeak = st.slider(
        "ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.5, step=0.1
    )

    slope = st.selectbox(
        "ST Slope",
        options=["Upsloping", "Flat", "Downsloping"],
    )

    # Additional Tests
    st.sidebar.markdown("### Additional Tests")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        ca = st.selectbox("Vessels Colored (0-4)", options=[0, 1, 2, 3, 4])
    with col2:
        thal = st.selectbox(
            "Thalassemia",
            options=["Unknown", "Normal", "Fixed Defect", "Reversible Defect"],
        )

    # Convert to numeric values
    patient_data = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic",
        ].index(cp),
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "restecg": [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy",
        ].index(restecg),
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": ["Upsloping", "Flat", "Downsloping"].index(slope),
        "ca": ca,
        "thal": ["Unknown", "Normal", "Fixed Defect", "Reversible Defect"].index(thal),
    }

    return patient_data


def render_prediction_result(result: dict):
    """Render the prediction result."""
    risk_level = result["risk_level"]
    probability = result["probability"]

    # Risk level card
    risk_class = f"risk-{risk_level}"
    risk_emoji = {"low": "✅", "medium": "⚠️", "high": "🚨"}[risk_level]

    st.markdown(
        f"""
        <div class="risk-card {risk_class}">
            <h1>{risk_emoji} {risk_level.upper()} RISK</h1>
            <h2>{probability:.1%} probability of heart disease</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Risk Level",
            value=risk_level.capitalize(),
        )

    with col2:
        st.metric(
            label="Probability",
            value=f"{probability:.1%}",
        )

    with col3:
        st.metric(
            label="Prediction",
            value=result["prediction_label"].replace("_", " ").title(),
        )

    with col4:
        st.metric(
            label="Confidence",
            value=f"{result['confidence']:.1%}",
        )


def render_risk_factors(patient_data: dict):
    """Render risk factor analysis."""
    st.markdown("### 📊 Risk Factor Analysis")

    factors = []

    # Age risk
    if patient_data["age"] > 55:
        factors.append(("🔴", "Age > 55 years", "Higher risk with advancing age"))
    else:
        factors.append(("🟢", "Age ≤ 55 years", "Lower age-related risk"))

    # Blood pressure
    if patient_data["trestbps"] >= 140:
        factors.append(
            (
                "🔴",
                "High Blood Pressure",
                f"BP: {patient_data['trestbps']} mmHg (Stage 2 Hypertension)",
            )
        )
    elif patient_data["trestbps"] >= 130:
        factors.append(
            (
                "🟡",
                "Elevated Blood Pressure",
                f"BP: {patient_data['trestbps']} mmHg (Stage 1 Hypertension)",
            )
        )
    else:
        factors.append(
            ("🟢", "Normal Blood Pressure", f"BP: {patient_data['trestbps']} mmHg")
        )

    # Cholesterol
    if patient_data["chol"] >= 240:
        factors.append(
            ("🔴", "High Cholesterol", f"Cholesterol: {patient_data['chol']} mg/dl")
        )
    elif patient_data["chol"] >= 200:
        factors.append(
            (
                "🟡",
                "Borderline Cholesterol",
                f"Cholesterol: {patient_data['chol']} mg/dl",
            )
        )
    else:
        factors.append(
            ("🟢", "Healthy Cholesterol", f"Cholesterol: {patient_data['chol']} mg/dl")
        )

    # Exercise angina
    if patient_data["exang"] == 1:
        factors.append(
            ("🔴", "Exercise-Induced Angina", "Chest pain during physical activity")
        )
    else:
        factors.append(("🟢", "No Exercise Angina", "No chest pain during activity"))

    # ST Depression
    if patient_data["oldpeak"] > 2:
        factors.append(
            ("🔴", "Significant ST Depression", f"Oldpeak: {patient_data['oldpeak']}")
        )
    elif patient_data["oldpeak"] > 1:
        factors.append(
            ("🟡", "Moderate ST Depression", f"Oldpeak: {patient_data['oldpeak']}")
        )
    else:
        factors.append(
            ("🟢", "Minimal ST Depression", f"Oldpeak: {patient_data['oldpeak']}")
        )

    # Display factors
    for icon, title, detail in factors:
        st.markdown(f"{icon} **{title}** - {detail}")


def render_model_info(model, preprocessor):
    """Render model information."""
    with st.expander("ℹ️ Model Information"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Model Type:** Logistic Regression")
            st.markdown("**Version:** 1.0.3")
            st.markdown("**Features:** 21 (including engineered)")

        with col2:
            st.markdown("**ROC-AUC:** 0.878")
            st.markdown("**Recall:** 100%")
            st.markdown("**Threshold:** 0.5")

        st.markdown("---")
        st.markdown(
            """
            **Disclaimer:** This tool is for educational and research purposes only.
            It should not be used as a substitute for professional medical advice,
            diagnosis, or treatment. Always consult with a qualified healthcare provider.
            """
        )


def render_footer():
    """Render the footer."""
    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
            <p>❤️ HeartHealthML | AI-Powered Cardiac Risk Assessment</p>
            <p>© 2024 HeartHealthML. For research and educational purposes only.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Main application."""
    # Load model
    model, preprocessor, error = load_model()

    # Render header
    render_header()

    # Check model status
    if error:
        st.error(f"⚠️ Model not loaded: {error}")
        st.info(
            "Please ensure the model is trained and available in the models/ directory."
        )
        return

    # Render sidebar and get patient data
    patient_data = render_sidebar()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## 🩺 Risk Assessment")

        # Predict button
        if st.button("🔍 Analyze Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing patient data..."):
                try:
                    # Apply feature engineering
                    df = pd.DataFrame([patient_data])
                    df = engineer_features(df)
                    engineered_data = df.iloc[0].to_dict()

                    # Make prediction
                    result = predict_single(
                        model=model,
                        preprocessor=preprocessor,
                        patient_data=engineered_data,
                        threshold=0.5,
                    )

                    # Add confidence
                    result["confidence"] = (
                        result["probability"]
                        if result["prediction"] == 1
                        else 1 - result["probability"]
                    )

                    # Store result in session state
                    st.session_state["result"] = result
                    st.session_state["patient_data"] = patient_data

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

        # Display result if available
        if "result" in st.session_state:
            render_prediction_result(st.session_state["result"])

            st.markdown("---")

            # Risk factors
            render_risk_factors(st.session_state["patient_data"])

    with col2:
        st.markdown("## 📈 Quick Stats")

        # Patient summary
        st.markdown("### Patient Summary")
        st.markdown(f"**Age:** {patient_data['age']} years")
        st.markdown(f"**Sex:** {'Male' if patient_data['sex'] == 1 else 'Female'}")
        st.markdown(f"**Blood Pressure:** {patient_data['trestbps']} mmHg")
        st.markdown(f"**Cholesterol:** {patient_data['chol']} mg/dl")
        st.markdown(f"**Max Heart Rate:** {patient_data['thalach']} bpm")

        # Heart rate reserve
        hr_reserve = (220 - patient_data["age"]) - patient_data["thalach"]
        st.markdown(f"**Heart Rate Reserve:** {hr_reserve} bpm")

        # Model info
        st.markdown("---")
        render_model_info(model, preprocessor)

    # Footer
    render_footer()


if __name__ == "__main__":
    main()
