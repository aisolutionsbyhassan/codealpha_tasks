import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# ------------------ Page Setup ------------------
st.set_page_config(
    page_title="Heart Disease Predictor 💓",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
body {background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);}
h1 {color: #ff4b5c; text-align: center; font-size: 3em; margin-bottom: 0.3em;}
h2, h3 {color: #00d4ff;}
.stButton>button {
    background: linear-gradient(90deg, #ff4b5c, #ff6b81);
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 12px 25px;
    border-radius: 30px;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {transform: scale(1.05);}
.stMetric > div {background: #1f1f1f; color: #00ffcc; padding: 15px; border-radius: 12px; font-weight:bold; text-align:center;}
</style>
""", unsafe_allow_html=True)

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("heart_model.pkl")
    except:
        st.error("Model file not found!")
        return None

model = load_model()

# ------------------ App Header ------------------
st.markdown("<h1>💓 Heart Disease Risk Checker</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#00d4ff; font-size:1.2em;'>Interactive AI-based predictor using optimized SVM</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("### 🤖 About This Tool")
    st.info("Predict your heart disease risk using 10 important health indicators.")
    st.markdown("---")
    st.markdown("### 📌 Quick Instructions")
    st.write("1. Fill in your health details below")
    st.write("2. Click 'Predict Risk'")
    st.write("3. See your personalized result and guidance")

# ------------------ Input Form ------------------
st.markdown("### 📝 Enter Your Health Data")
col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    trestbps = st.number_input("Resting BP (mmHg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    oldpeak = st.number_input("ST Depression", 0.0, 7.0, 1.0, 0.1)
    slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])

with col3:
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: ["Typical", "Atypical", "Non-anginal", "Asymptomatic"][x])
    ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])

with col4:
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible", "Unknown"][x])
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: ["Female", "Male"][x])

st.markdown("<br>", unsafe_allow_html=True)

# ------------------ Prediction ------------------
if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):
    if model:
        input_df = pd.DataFrame([[cp, ca, age, chol, thal, oldpeak, trestbps, thalach, slope, sex]],
                                columns=['cp', 'ca', 'age', 'chol', 'thal', 'oldpeak', 'trestbps', 'thalach', 'slope', 'sex'])

        # Optional scaling
        scaler = StandardScaler()
        scaler.mean_ = np.array([1, 1, 54, 246, 0.5, 1, 131, 149, 1, 0.5])
        scaler.scale_ = np.array([1, 1.2, 9, 51, 1, 1.2, 17, 23, 0.6, 0.5])
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]

        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")

        col1, col2 = st.columns([1,1])

        # Result Card
        with col1:
            if prediction == 1:
                st.markdown("""
                    <div style='background:#ff4b5c; padding:40px; border-radius:20px; text-align:center;'>
                        <h2 style='color:white; margin:0;'>⚠️ High Risk</h2>
                        <p style='color:white; font-size:1.2em;'>Indicators for heart disease detected</p>
                    </div>
                """, unsafe_allow_html=True)
                risk_value = 85
                st.warning("**Recommendation:** Consult a cardiologist for detailed evaluation.")
            else:
                st.markdown("""
                    <div style='background:#06ffa5; padding:40px; border-radius:20px; text-align:center;'>
                        <h2 style='color:#1a1a2e; margin:0;'>✅ Low Risk</h2>
                        <p style='color:#1a1a2e; font-size:1.2em;'>No significant indicators found</p>
                    </div>
                """, unsafe_allow_html=True)
                risk_value = 15
                st.success("**Great news!** Keep maintaining a healthy lifestyle.")

        # Risk Gauge
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_value,
                title={'text': "Risk Level", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ff4b5c" if prediction == 1 else "#06ffa5"},
                    'steps': [
                        {'range': [0, 30], 'color': "#d4edda"},
                        {'range': [30, 70], 'color': "#fff3cd"},
                        {'range': [70, 100], 'color': "#f8d7da"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': risk_value}
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # User Input Summary with Improved Colors
        st.markdown("### 📊 Your Health Parameters")
        params = {
            'Age': age,
            'Resting BP': trestbps,
            'Cholesterol': chol,
            'Max Heart Rate': thalach,
            'ST Depression': oldpeak,
            'Chest Pain Type': ["Typical", "Atypical", "Non-anginal", "Asymptomatic"][cp],
            'ST Slope': ["Upsloping", "Flat", "Downsloping"][slope],
            'Thalassemia': ["Normal", "Fixed Defect", "Reversible", "Unknown"][thal],
            'Sex': ["Female", "Male"][sex],
            'Major Vessels': ca
        }

        cols = st.columns(5)
        colors = ['#ff6b81', '#06ffa5', '#ffd93d', '#6a4c93', '#00d4ff']  # vibrant metric backgrounds
        for i, (key, value) in enumerate(params.items()):
            with cols[i % 5]:
                st.markdown(f"""
                    <div style='background:{colors[i % 5]}; padding:15px; border-radius:15px; color:white; font-weight:bold; text-align:center;'>
                        <h4 style='margin:0'>{key}</h4>
                        <p style='margin:0; font-size:1.1em'>{value}</p>
                    </div>
                """, unsafe_allow_html=True)

# ------------------ Footer ------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<p style='text-align: center; color:#00d4ff;'>⚕️ This tool provides predictions based on AI models. Always consult a healthcare professional for medical advice.</p>", unsafe_allow_html=True)
