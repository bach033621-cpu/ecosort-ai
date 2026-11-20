import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("EcoSort_Model_B.keras")

@st.cache_data
def load_waste_data():
    with open("data.json", "r", encoding="utf-8") as f:
        return json.load(f)["classes"]

st.title("♻️ EcoSort AI")
st.caption("Hệ thống nhận diện và hướng dẫn phân loại rác thông minh")

try:
    model = load_model()
    waste_classes = load_waste_data()
except Exception as e:
    st.error(f"Lỗi tải mô hình hoặc dữ liệu: {e}")
    st.stop()

input_mode = st.radio(
    "Chọn nguồn ảnh:",
    ["Tải ảnh từ thiết bị", "Chụp trực tiếp bằng camera"],
    horizontal=True,
)

uploaded_file = (
    st.file_uploader("Chọn ảnh rác cần phân loại", type=["jpg", "jpeg", "png", "webp"])
    if input_mode == "Tải ảnh từ thiết bị"
    else st.camera_input("Chụp ảnh rác")
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã chọn", use_container_width=True)

    # Tự động lấy kích thước ảnh chuẩn theo model
    input_shape = model.input_shape
    img_h = input_shape[1] if input_shape[1] else 224
    img_w = input_shape[2] if input_shape[2] else 224

    img_resized = image.resize((img_w, img_h))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_batch, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100

    if top_idx < len(waste_classes):
        result = waste_classes[top_idx]
        st.success(f"**Kết quả:** {result['name']} ({confidence:.1f}%)")
        st.info(f"**Phân loại thùng:** {result['bin']}")
        st.write(f"**Hướng dẫn xử lý:** {result['guide']}")
    else:
        st.warning(f"Nhận diện lớp số #{top_idx} ({confidence:.1f}%) - cần bổ sung vào data.json.")

    with st.expander("Xem xác suất chi tiết các nhóm"):
        for idx, prob in enumerate(preds):
            label = waste_classes[idx]["name"] if idx < len(waste_classes) else f"Lớp {idx}"
            st.write(f"- **{label}:** {float(prob) * 100:.2f}%")
