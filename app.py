import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os

# -------------------------------
# Load trained ML model
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(
    page_title="Smart Loan Approval System",
    layout="centered"
)

st.title("🏦 Smart Loan Approval System")
st.write("Real-time loan approval using Machine Learning")

# ===============================
# BANK DETAILS (UI ONLY)
# ===============================
st.subheader("🏛 Bank Details")

applicant_name = st.text_input("Applicant Full Name")
bank_name = st.text_input("Bank Name")
account_number = st.text_input("Account Number")
ifsc_code = st.text_input("IFSC Code")

# ===============================
# FINANCIAL DETAILS (MODEL INPUT)
# ===============================
st.subheader("💰 Financial Information")

income = st.number_input(
    "Applicant Annual Income",
    min_value=0,
    step=1000
)

loan_amount = st.number_input(
    "Requested Loan Amount",
    min_value=0,
    step=1000
)

loan_term = st.number_input(
    "Loan Term (in months)",
    min_value=1
)

credit_history = st.selectbox(
    "Credit History",
    options=[1, 0],
    format_func=lambda x: "Good" if x == 1 else "Bad"
)

# ===============================
# LOAN HISTORY
# ===============================
st.subheader("📄 Loan History")

prev_loan = st.selectbox(
    "Any Previous Loans?",
    ["No", "Yes"]
)

if prev_loan == "Yes":
    previous_loan = 1
    num_prev_loans = st.number_input(
        "Number of Previous Loans",
        min_value=1,
        max_value=10
    )
    default_history = st.selectbox(
        "Any Loan Defaults?",
        ["No", "Yes"]
    )
    default_history = 1 if default_history == "Yes" else 0
else:
    previous_loan = 0
    num_prev_loans = 0
    default_history = 0

# ===============================
# WEBCAM CAPTURE
# ===============================
st.subheader("📸 Identity Verification")

photo = st.camera_input("Capture Live Applicant Photo")

if photo is not None:
    image = Image.open(photo)
    st.image(image, caption="Captured Photo", width=250)

    if not os.path.exists("photos"):
        os.makedirs("photos")

    image.save(f"photos/{applicant_name}_photo.jpg")

# ===============================
# PREDICTION
# ===============================
st.markdown("---")

if st.button("🔍 Predict Loan Status"):

    if applicant_name == "":
        st.warning("⚠ Please enter applicant name")
    elif photo is None:
        st.warning("⚠ Please capture a live photo")
    else:
        # IMPORTANT: Feature order MUST match training code
        input_data = np.array([[
            income,
            loan_amount,
            loan_term,
            credit_history,
            previous_loan,
            num_prev_loans,
            default_history
        ]])

        prediction = model.predict(input_data)

        st.subheader("📊 Loan Decision")

        if prediction[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("© Smart Loan Approval System | Machine Learning Project")
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key="AIzaSyDl-112d9iUo_uelF7JIaW_nH3WL4IPFbo")

model = genai.GenerativeModel("gemini-pro")

st.markdown("---")
st.subheader("🤖 Jarvis – Banking Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask Jarvis about loans, eligibility, documents...")

if st.button("Send"):
    if user_input:
        response = model.generate_content(
            f"You are Jarvis, a professional banking assistant. Answer clearly:\n{user_input}"
        )

        st.session_state.chat_history.append(
            ("You", user_input)
        )
        st.session_state.chat_history.append(
            ("Jarvis", response.text)
        )

# Display chat
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑 {sender}:** {msg}")
    else:
        st.markdown(f"**🤖 {sender}:** {msg}")
