import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Nhận Diện Biển Báo Giao Thông Việt Nam - Hà Thu",
    page_icon="🚦",
    layout="centered",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Be Vietnam Pro', sans-serif !important;
        font-size: 18px !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 15%, rgba(56, 189, 248, 0.22), transparent 36%),
            radial-gradient(circle at 88% 18%, rgba(16, 185, 129, 0.18), transparent 38%),
            linear-gradient(180deg, #f8fafc 0%, #eef6ff 100%);
    }

    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #0284c7 100%);
        border-radius: 24px;
        padding: 28px 30px;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 18px 38px rgba(29, 78, 216, 0.22);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 24px;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.28);
        color: #e0f2fe;
        font-size: 14px;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 999px;
        margin-bottom: 12px;
        letter-spacing: 0.4px;
    }

    .hero-logo {
        font-size: 36px;
        font-weight: 800;
        line-height: 1.2;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.18);
    }

    .hero-sub {
        font-size: 17px;
        font-weight: 500;
        color: #e2e8f0;
        margin-top: 10px;
        line-height: 1.45;
    }

    [data-testid="stFileUploader"] label {
        font-size: 19px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    [data-testid="stFileUploader"] section {
        background: #ffffff !important;
        border: 2px dashed #3b82f6 !important;
        border-radius: 20px !important;
        padding: 24px !important;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05) !important;
    }

    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small {
        font-size: 16px !important;
    }

    .prediction-card {
        background: #ffffff;
        border-radius: 24px;
        padding: 28px 26px;
        margin-top: 22px;
        text-align: center;
        border: 2px solid #cbd5e1;
        box-shadow: 0 16px 34px rgba(15, 23, 42, 0.09);
    }

    .prediction-tag {
        display: inline-block;
        background: #dbeafe;
        color: #1d4ed8;
        font-size: 15px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 6px 16px;
        border-radius: 999px;
    }

    .prediction-title {
        font-size: 32px;
        font-weight: 800;
        color: #0f172a;
        margin: 14px 0 16px 0;
        line-height: 1.25;
    }

    .prediction-conf {
        display: inline-block;
        background: linear-gradient(90deg, #059669 0%, #10b981 100%);
        color: #ffffff;
        font-size: 20px;
        font-weight: 800;
        padding: 10px 22px;
        border-radius: 999px;
        box-shadow: 0 8px 18px rgba(16, 185, 129, 0.28);
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero-card">
    <div class="hero-badge">MOBILENETV2 • AI TRAFFIC SIGN</div>
    <div class="hero-logo">🚦 Nhận Diện Biển Báo Giao Thông Việt Nam - Hà Thu</div>
    <div class="hero-sub">Hệ thống nhận diện biển báo giao thông Việt Nam qua hình ảnh</div>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("traffic_sign_model.keras")
    with open("class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


try:
    model, class_names = load_model_and_classes()
except Exception as e:
    st.error(f"Không tải được mô hình hoặc class_names.json: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Tải ảnh biển báo giao thông để nhận diện",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh biển báo đã tải lên", use_container_width=True)

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch, verbose=0)[0]
    top_idx = int(np.argmax(predictions))
    confidence = float(predictions[top_idx]) * 100
    best_label = class_names[top_idx]

    st.markdown(
        f"""
    <div class="prediction-card">
        <div class="prediction-tag">Kết quả nhận diện</div>
        <div class="prediction-title">{best_label}</div>
        <div class="prediction-conf">Độ tin cậy: {confidence:.2f}%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )