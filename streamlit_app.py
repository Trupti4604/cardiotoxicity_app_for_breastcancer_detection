import streamlit as st
import requests
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cardiotoxicity Prediction",
    layout="centered"
)

API_URL = "http://127.0.0.1:5000/predict"

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* ---------- GLOBAL BACKGROUND ---------- */
.stApp {
    background: linear-gradient(135deg, #fff5f8, #f3f6fb);
    animation: fadeIn 1s ease-in;
}

/* ---------- CARDS ---------- */
.card {
    background-color: white;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 22px;
    animation: slideUp 0.8s ease forwards;
}

/* ---------- HEADINGS ---------- */
h1, h2, h3 {
    color: #c2185b;
    animation: fadeInDown 0.8s ease-in-out;
}

/* ---------- RESULT BOX ---------- */
.result-box {
    padding: 24px;
    border-radius: 14px;
    text-align: center;
    font-size: 18px;
    animation: popIn 0.6s ease-out;
}

/* ---------- RISK COLORS ---------- */
.low { background-color: #e8f5e9; color: #2e7d32; }
.moderate { background-color: #fff8e1; color: #f9a825; }
.high { background-color: #ffebee; color: #c62828; }

/* ---------- BUTTON ANIMATION ---------- */
button[kind="primary"] {
    background: linear-gradient(90deg, #c2185b, #e91e63);
    color: white;
    border-radius: 12px;
    font-weight: bold;
    transition: all 0.3s ease-in-out;
}

button[kind="primary"]:hover {
    transform: scale(1.03);
    box-shadow: 0px 6px 18px rgba(194,24,91,0.4);
}

/* ---------- KEYFRAMES ---------- */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-15px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes popIn {
    0% { transform: scale(0.9); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div style="text-align: center;">
    <h1>Breast Cancer Cardiotoxicity Prediction</h1>
    <p style="color:gray; font-size:18px;">
        ML-based multimodal risk assessment using clinical parameters and echocardiography images.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- CLINICAL INPUTS ----------------
st.subheader("🧑‍⚕️ Clinical Parameters")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", 0, 120, 50)
    hr = st.number_input("Heart Rate (bpm)", 40, 180, 70)
    lvef = st.number_input("LVEF (%)", 20, 80, 55)

with col2:
    weight = st.number_input("Weight (kg)", 30, 150, 70)
    height = st.number_input("Height (cm)", 130, 200, 165)
    time = st.number_input("Time since therapy start (months)", 0, 120, 0)

# BMI Calculation
bmi = round(weight / ((height / 100) ** 2), 2)
st.info(f"📊 **Calculated BMI:** {bmi}")

st.markdown("</div>", unsafe_allow_html=True)

clinical_data = {
    "age": age,
    "hr": hr,
    "lvef": lvef,
    "weight": weight,
    "height": height,
    "time": time,
    "bmi": bmi
}

# ---------------- IMAGE UPLOAD ----------------
st.subheader("🫀 Echocardiography Image")

uploaded_file = st.file_uploader(
    "Upload Echo Image (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Echo Image", use_column_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Cardiotoxicity", use_container_width=True):

    if not uploaded_file:
        st.warning("⚠️ Please upload an echocardiography image.")
    else:
        with st.spinner("Analyzing data and image…"):
            image_bytes = uploaded_file.read()
            image_base64 = base64.b64encode(image_bytes).decode()

            payload = {
                "clinical": clinical_data,
                "image_base64": image_base64
            }

            response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            res = response.json()
            score = res["cardiotoxicity_score"]
            risk = res["risk"].lower()

            risk_class = "low"
            if "moderate" in risk:
                risk_class = "moderate"
            elif "high" in risk:
                risk_class = "high"

            st.markdown(f"""
            <div class="result-box {risk_class}">
                <h2>🧪 Prediction Result</h2>
                <p><b>Cardiotoxicity Score:</b> {score:.2f}</p>
                <p><b>Risk Level:</b> {res["risk"]}</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.error("❌ Prediction failed. Please check API connection.")
