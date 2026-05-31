import streamlit as st

st.set_page_config(
    page_title="MediPredict",
    page_icon="🏥",
    layout="centered"
)

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin: 10px;
    }
    div.stButton > button {
        width: 100%;
        padding: 12px;
        border-radius: 10px;
        font-size: 16px;
        border: none;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center; color:#2c3e50;'>🏥 MediPredict</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#aaa; font-size:18px;'>AI Powered Disease Prediction Platform</p>",
            unsafe_allow_html=True)
st.markdown("---")

st.write("")
st.markdown("<h3 style='text-align:center; color:#2c3e50;'>Select a Prediction Tool</h3>",
            unsafe_allow_html=True)
st.write("")

# Two cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="card">
            <h2>🩺</h2>
            <h3 style='color:#4a90d9;'>Diabetes</h3>
            <p style='color:#888;'>Predict diabetes risk using 
            key medical indicators like 
            glucose, BMI and age</p>
        </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("Open Diabetes Predictor →",
                  use_container_width=True):
        st.switch_page("pages/1_Diabetes.py")

with col2:
    st.markdown("""
        <div class="card">
            <h2>❤️</h2>
            <h3 style='color:#e53935;'>Heart Disease</h3>
            <p style='color:#888;'>Predict heart disease risk 
            using cardiac indicators like 
            cholesterol and heart rate</p>
        </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("Open Heart Disease Predictor →",
                  use_container_width=True):
        st.switch_page("pages/2_Heart.py")

st.markdown("---")

# Stats row
s1, s2, s3 = st.columns(3)
with s1:
    st.metric("🧠 ML Models", "2 Trained")
with s2:
    st.metric("📊 Diseases", "2 Covered")
with s3:
    st.metric("🎯 Best Accuracy", "98.54%")

st.markdown("---")
st.caption("⚠️ This app is for educational purposes only, not medical advice.")