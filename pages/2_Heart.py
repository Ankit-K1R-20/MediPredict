import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MediPredict - Heart Disease",
    page_icon="❤️",
    layout="centered"
)

# Red theme CSS
st.markdown("""
    <style>
    .main {background-color: #fff5f5;}
    div.stButton > button {
        width: 100%;
        background-color: #e53935;
        color: white;
        padding: 12px;
        border-radius: 8px;
        font-size: 16px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #b71c1c;
        color: white;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #ffcccc;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 1px 1px 4px rgba(229,57,53,0.15);
    }
    h1 {color: #c0392b;}
    h2 {color: #c0392b;}
    h3 {color: #c0392b;}
    hr {border-color: #ffcccc;}
    </style>
""", unsafe_allow_html=True)

# Load heart model and scaler using absolute path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = pickle.load(open(os.path.join(base_dir, 'heart_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(base_dir, 'heart_scaler.pkl'), 'rb'))

# Title
st.markdown("# ❤️ Heart Disease Prediction")
st.markdown("<p style='color:#aaa;'>Enter the patient's medical information below to predict heart disease risk.</p>",
            unsafe_allow_html=True)
st.markdown("---")

# Back button
if st.button("🏠 Back to Home"):
    st.switch_page("home.py")

st.markdown("---")

# Input fields
st.subheader("Patient Medical Information")
st.write("")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age",
                           min_value=1, max_value=120, value=45)
    sex = st.selectbox("👤 Sex",
                        options=[0, 1],
                        format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("💔 Chest Pain Type",
                       options=[0, 1, 2, 3],
                       format_func=lambda x: {
                           0: "Typical Angina",
                           1: "Atypical Angina",
                           2: "Non-anginal Pain",
                           3: "Asymptomatic"
                       }[x])
    trestbps = st.number_input("🩺 Resting Blood Pressure",
                                min_value=80, max_value=200, value=120)
    chol = st.number_input("🧪 Cholesterol",
                            min_value=100, max_value=600, value=200)
    fbs = st.selectbox("🍬 Fasting Blood Sugar > 120mg/dl",
                        options=[0, 1],
                        format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("📈 Resting ECG",
                            options=[0, 1, 2],
                            format_func=lambda x: {
                                0: "Normal",
                                1: "ST-T Abnormality",
                                2: "Left Ventricular Hypertrophy"
                            }[x])

with col2:
    thalach = st.number_input("💓 Maximum Heart Rate",
                               min_value=60, max_value=220, value=150)
    exang = st.selectbox("🏃 Exercise Induced Angina",
                          options=[0, 1],
                          format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("📉 ST Depression",
                               min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("📊 Slope of Peak Exercise",
                          options=[0, 1, 2],
                          format_func=lambda x: {
                              0: "Downsloping",
                              1: "Flat",
                              2: "Upsloping"
                          }[x])
    ca = st.selectbox("🫀 Number of Major Vessels",
                       options=[0, 1, 2, 3, 4])
    thal = st.selectbox("🧬 Thalassemia",
                         options=[0, 1, 2, 3],
                         format_func=lambda x: {
                             0: "Normal",
                             1: "Fixed Defect",
                             2: "Reversible Defect",
                             3: "Unknown"
                         }[x])

st.markdown("---")

# Predict button
if st.button("🔍 Predict Heart Disease Risk"):

    # Prepare input
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1] * 100
    healthy_prob = 100 - probability

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease — {probability:.1f}% probability")
        st.write("Please consult a cardiologist for proper medical advice.")
    else:
        st.success(f"✅ Low Risk of Heart Disease — {probability:.1f}% probability")
        st.write("Keep maintaining a heart healthy lifestyle!")

    st.write("")

    # 3 metric cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("🔴 Heart Disease Risk", f"{probability:.1f}%")
    with m2:
        st.metric("🟢 Healthy Probability", f"{healthy_prob:.1f}%")
    with m3:
        risk_level = "High 🔴" if probability > 60 else \
                     "Medium 🟡" if probability > 40 else "Low 🟢"
        st.metric("⚠️ Risk Level", risk_level)

    st.write("")

    # Red themed chart
    st.write("#### Risk Breakdown")
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#fff5f5')
    ax.set_facecolor('#fff5f5')
    bars = ax.barh(['Healthy', 'Heart Disease Risk'],
                   [healthy_prob, probability],
                   color=['#74c69d', '#e53935'],
                   height=0.4)
    ax.set_xlim([0, 100])
    ax.set_xlabel('Probability (%)', color='#888')
    for bar, val in zip(bars, [healthy_prob, probability]):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center',
                fontweight='bold', color='#c0392b')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ffcccc')
    ax.spines['bottom'].set_color('#ffcccc')
    ax.tick_params(colors='#888')
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")
st.caption("⚠️ This app is for educational purposes only, not medical advice.")