import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MediPredict - Diabetes",
    page_icon="🩺",
    layout="centered"
)

# Light blue theme CSS
st.markdown("""
    <style>
    .main {background-color: #f0f7ff;}
    div.stButton > button {
        width: 100%;
        background-color: #4a90d9;
        color: white;
        padding: 12px;
        border-radius: 8px;
        font-size: 16px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #357abd;
        color: white;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e8f5;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 1px 1px 4px rgba(74,144,217,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler using absolute path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = pickle.load(open(os.path.join(base_dir, 'diabetes_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(base_dir, 'diabetes_scaler.pkl'), 'rb'))

# Title
st.markdown("# 🩺 Diabetes Prediction")
st.markdown("<p style='color:#aaa;'>Enter the patient's medical information below to predict diabetes risk.</p>",
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
    pregnancies = st.number_input("🤰 Pregnancies",
                                   min_value=0, max_value=20, value=1)
    glucose = st.number_input("🩸 Glucose Level",
                               min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("💉 Blood Pressure",
                                      min_value=0, max_value=150, value=70)

with col2:
    bmi = st.number_input("⚖️ BMI",
                           min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree = st.number_input("🧬 Diabetes Pedigree Function",
                                         min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("🎂 Age",
                           min_value=1, max_value=120, value=25)

st.markdown("---")

# Predict button
if st.button("🔍 Predict Diabetes Risk"):

    # Prepare input
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            bmi, diabetes_pedigree, age]])
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1] * 100
    healthy_prob = 100 - probability

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Diabetes — {probability:.1f}% probability")
        st.write("Please consult a doctor for proper medical advice.")
    else:
        st.success(f"✅ Low Risk of Diabetes — {probability:.1f}% probability")
        st.write("Keep maintaining a healthy lifestyle!")

    st.write("")

    # 3 metric cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("🔴 Diabetes Risk", f"{probability:.1f}%")
    with m2:
        st.metric("🟢 Healthy Probability", f"{healthy_prob:.1f}%")
    with m3:
        risk_level = "High 🔴" if probability > 60 else \
                     "Medium 🟡" if probability > 40 else "Low 🟢"
        st.metric("⚠️ Risk Level", risk_level)

    st.write("")

    # Chart
    st.write("#### Risk Breakdown")
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#f0f7ff')
    ax.set_facecolor('#f0f7ff')
    bars = ax.barh(['Healthy', 'Diabetes Risk'],
                   [healthy_prob, probability],
                   color=['#74c69d', '#4a90d9'],
                   height=0.4)
    ax.set_xlim([0, 100])
    ax.set_xlabel('Probability (%)', color='#888')
    for bar, val in zip(bars, [healthy_prob, probability]):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center',
                fontweight='bold', color='#2c3e50')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dee2e6')
    ax.spines['bottom'].set_color('#dee2e6')
    ax.tick_params(colors='#888')
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")
st.caption("⚠️ This app is for educational purposes only, not medical advice.")