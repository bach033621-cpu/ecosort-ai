
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image, ImageOps
from pathlib import Path

# ======================================
# CẤU HÌNH WEBSITE
# ======================================

st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide"
)

TEN_NHAN = [
    "Rác hữu cơ",
    "Rác còn lại",
    "Rác tái chế"
]

HUONG_DAN = {
    "Rác hữu cơ": (
        "Gợi ý: Thu gom riêng vỏ trái cây, rau củ và "
        "thức ăn thừa. Có thể xử lý bằng phương pháp "
        "ủ hữu cơ phù hợp."
    ),

    "Rác còn lại": (
        "Gợi ý: Kiểm tra vật liệu và tình trạng rác. "
        "Các loại rác không có phương án tái chế thông "
        "thường có thể cần được thu gom riêng."
    ),

    "Rác tái chế": (
        "Gợi ý: Làm rỗng, làm sạch nếu cần và giữ khô "
        "chai nhựa, lon, giấy hoặc bìa carton trước "
        "khi đưa đến điểm thu gom phù hợp."
    )
}

# ======================================
# TẢI MÔ HÌNH
# ======================================

@st.cache_resource
def tai_mo_hinh():
    duong_dan = Path("EcoSort_Model_B.keras")

    if not duong_dan.exists():
        raise FileNotFoundError(
            "Không tìm thấy file mô hình."
        )

    return tf.keras.models.load_model(
        duong_dan,
        compile=False
    )

model = tai_mo_hinh()

# ======================================
# GIAO DIỆN CHÍNH
# ======================================

st.title("♻️ ECOSORT AI")
st.subheader(
    "Hệ thống hỗ trợ nhận diện và phân loại rác thông minh"
)

st.write(
    "Tải hình ảnh hoặc sử dụng camera để AI "
    "đưa ra gợi ý phân loại rác."
)

st.info(
    "Đây là mô hình thử nghiệm. Kết quả có thể "
    "không chính xác, đặc biệt với rác hỗn hợp, "
    "ảnh không rõ hoặc những đồ vật nằm ngoài "
    "dữ liệu huấn luyện. Hãy kiểm tra lại trước "
    "khi quyết định cách xử lý."
)

# ======================================
# NHẬN HÌNH ẢNH
# ======================================

cot1, cot2 = st.columns(2)

with cot1:
    anh_tai_len = st.file_uploader(
        "Tải hình ảnh từ thiết bị",
        type=["jpg", "jpeg", "png", "webp"]
    )

with cot2:
    anh_camera = st.camera_input(
        "Hoặc chụp ảnh bằng camera"
    )

anh_nguon = anh_tai_len or anh_camera

# ======================================
# NHẬN DIỆN
# ======================================

if anh_nguon is not None:

    anh = Image.open(anh_nguon)

    anh = ImageOps.exif_transpose(
        anh
    ).convert("RGB")

    st.image(
        anh,
        caption="Hình ảnh được cung cấp",
        width=400
    )

    if st.button(
        "🔍 PHÂN LOẠI RÁC",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "AI đang phân tích hình ảnh..."
        ):

            # Đưa ảnh về kích thước 224x224
            anh_xu_ly = anh.resize(
                (224, 224)
            )

            mang_anh = np.asarray(
                anh_xu_ly,
                dtype=np.float32
            )

            mang_anh = np.expand_dims(
                mang_anh,
                axis=0
            )

            # Model đã có lớp Rescaling bên trong
            du_doan = model.predict(
                mang_anh,
                verbose=0
            )[0]

            thu_tu = np.argsort(
                du_doan
            )[::-1]

            vi_tri = int(thu_tu[0])

            nhom = TEN_NHAN[vi_tri]

            diem = float(
                du_doan[vi_tri]
            )

        st.divider()

        st.header("Kết quả nhận diện")

        st.success(
            f"AI dự đoán: {nhom}"
        )

        st.metric(
            "Điểm dự đoán của mô hình",
            f"{diem * 100:.2f}%"
        )

        st.progress(diem)

        st.subheader(
            "Hướng dẫn xử lý tham khảo"
        )

        st.write(HUONG_DAN[nhom])

        if diem < 0.60:
            st.warning(
                "Điểm dự đoán thấp. Bạn nên kiểm tra "
                "hình ảnh và phân loại thủ công."
            )
        else:
            st.info(
                "Điểm dự đoán cao không bảo đảm "
                "AI luôn nhận diện đúng. Hãy kiểm "
                "tra vật liệu và quy định thu gom "
                "tại địa phương."
            )

        st.subheader(
            "So sánh điểm của ba nhóm"
        )

        for i in thu_tu:
            st.write(
                f"{TEN_NHAN[i]}: "
                f"{du_doan[i] * 100:.2f}%"
            )

            st.progress(
                float(du_doan[i])
            )

        st.caption(
            "Điểm số là đầu ra Softmax của mô hình, "
            "không phải xác suất đã được hiệu chuẩn."
        )

st.divider()

st.caption(
    "EcoSort AI | Dự án Tin học 10 | "
    "Gia Bách – Bảo Minh"
)

st.caption(
    "Độ chính xác Model B trong lần đánh giá "
    "độc lập hiện tại: 79,97% trên 2.910 ảnh. "
    "Đây là kết quả trên bộ dữ liệu thử nghiệm, "
    "không phải cam kết độ chính xác ngoài thực tế."
)
