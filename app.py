import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from inference import load_model, predict_signal

st.set_page_config(page_title="EEG Seizure Detection", layout="wide")

st.title("🧠 EEG Epileptic Seizure Detection System")
st.markdown("### CNN • BiLSTM • CNN-LSTM Hybrid")

# Sidebar
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox(
    "Choose Model",
    ["1D_CNN", "BiLSTM", "CNN_LSTM_Hybrid"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload EEG Signal (.txt)", type=["txt"]
)

# Load model
@st.cache_resource
def get_model(name):
    return load_model(name)

model = get_model(model_name)

if uploaded_file:
    signal_data = np.loadtxt(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("EEG Signal Waveform")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(signal_data[:1000])
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

    with col2:
        st.subheader("Prediction Result")
        label, confidence = predict_signal(model, signal_data)

        st.success(f"**Predicted Class:** {label}")
        st.info(f"**Confidence:** {confidence*100:.2f}%")

        if label == "Seizure":
            st.error("⚠️ Seizure Detected")
        else:
            st.success("✅ No Seizure Detected")

st.markdown("---")

# Show results
st.subheader("📊 Model Performance Summary")
st.image("./results/figures/ALL_comparison.png")

st.subheader("📉 Confusion Matrices")
cols = st.columns(3)
cols[0].image("./results/figures/1D_CNN_cm.png", caption="1D CNN")
cols[1].image("./results/figures/BiLSTM_cm.png", caption="BiLSTM")
cols[2].image("./results/figures/CNN_LSTM_Hybrid_cm.png", caption="CNN-LSTM Hybrid")