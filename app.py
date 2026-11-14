from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# ============================================================
# ECOSORT AI – GIAO DIỆN NÂNG CẤP
# Mô hình: MobileNetV2 fine-tuned (Model B)
# ============================================================

st.set_page_config(
    page_title="EcoSort AI | Phân loại rác thông minh",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Cấu hình dự đoán phải khớp thứ tự lớp lúc huấn luyện.
TEN_NHAN = ["Rác hữu cơ", "Rác còn lại", "Rác tái chế"]
KICH_THUOC_ANH = (224, 224)
NGUONG_DIEM = 0.60
NGUONG_KHOANG_CACH = 0.15
DUONG_DAN_MODEL = Path(__file__).resolve().parent / "EcoSort_Model_B.keras"

HUONG_DAN = {
    "Rác hữu cơ": (
        "Thu gom riêng phần thức ăn thừa, vỏ trái cây và rau củ. "
        "Có thể xử lý theo quy trình ủ hữu cơ phù hợp tại địa phương."
    ),
    "Rác còn lại": (
        "Kiểm tra vật liệu và hướng dẫn thu gom tại địa phương. "
        "Không mặc định mọi vật dụng khó nhận diện đều thuộc nhóm này."
    ),
    "Rác tái chế": (
        "Nếu vật liệu thuộc chương trình thu gom tại địa phương, hãy làm rỗng, "
        "làm sạch khi phù hợp và để khô trước khi phân loại riêng."
    ),
}

RAC_NGOAI_PHAM_VI = (
    "EcoSort AI hiện chỉ phân loại ba nhóm rác thông thường. "
    "Không dùng kết quả của mô hình để phân loại pin, ắc quy, bóng đèn, "
    "thuốc, hóa chất, rác điện tử hoặc vật liệu không rõ thành phần. "
    "Không tự xử lý vật liệu nguy hại; hãy để riêng và nhờ người lớn "
    "hoặc đơn vị thu gom có trách nhiệm hướng dẫn."
)

# Giao diện: ưu tiên tương phản, khoảng trắng và điều hướng rõ ràng.
st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem;}
      .eco-hero {padding: 1.5rem 1.7rem; border-radius: 18px;
        background: linear-gradient(115deg, #e5f5eb 0%, #f4faf6 60%, #eaf4ef 100%);
        border: 1px solid #cfe5d7; margin-bottom: 1.2rem;}
      .eco-kicker {color:#167447; font-size:.82rem; font-weight:700; letter-spacing:.08em;}
      .eco-hero h1 {color:#123b2a; margin:.25rem 0 .35rem 0; font-size:2.2rem;}
      .eco-hero p {color:#315b46; margin:0; font-size:1.03rem;}
      .eco-small {color:#5b7165; font-size:.88rem;}
      div[data-testid="stMetric"] {background:#f4faf6; border:1px solid #dcebe1;
        border-radius:14px; padding: .8rem 1rem;}
      div.stButton > button[kind="primary"] {background:#16824e; border-color:#16824e;}
      div.stButton > button[kind="primary"]:hover {background:#10683e; border-color:#10683e;}
      .stAlert {border-radius:12px;}
      h2, h3 {color:#183c2b;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Đang tải mô hình EcoSort AI...")
def tai_mo_hinh():
    if not DUONG_DAN_MODEL.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp mô hình: {DUONG_DAN_MODEL.name}. "
            "Hãy kiểm tra tệp mô hình đã nằm cùng thư mục với app.py chưa."
        )
    return tf.keras.models.load_model(DUONG_DAN_MODEL, compile=False)


def chuan_bi_anh(du_lieu_anh: bytes) -> tuple[Image.Image, np.ndarray]:
    """Đọc ảnh, chỉnh hướng EXIF, đổi RGB và resize bilinear giống lúc nạp dữ liệu."""
    with Image.open(io.BytesIO(du_lieu_anh)) as anh_goc:
        anh = ImageOps.exif_transpose(anh_goc).convert("RGB")
    anh_da_doi_kich_thuoc = anh.resize(KICH_THUOC_ANH, Image.Resampling.BILINEAR)
    mang = np.asarray(anh_da_doi_kich_thuoc, dtype=np.float32)
    mang = np.expand_dims(mang, axis=0)
    return anh, mang


def du_doan(model, mang_anh: np.ndarray) -> dict:
    dau_ra = np.asarray(model.predict(mang_anh, verbose=0))[0]
    if dau_ra.ndim != 1 or len(dau_ra) != len(TEN_NHAN):
        raise ValueError("Đầu ra mô hình không khớp với ba nhóm rác đã cấu hình.")
    if not np.all(np.isfinite(dau_ra)):
        raise ValueError("Mô hình trả về kết quả không hợp lệ.")
    thu_tu = np.argsort(dau_ra)[::-1]
    vi_tri_cao_nhat = int(thu_tu[0])
    vi_tri_thu_hai = int(thu_tu[1])
    diem_cao_nhat = float(dau_ra[vi_tri_cao_nhat])
    diem_thu_hai = float(dau_ra[vi_tri_thu_hai])
    return {
        "nhan": TEN_NHAN[vi_tri_cao_nhat],
        "diem": diem_cao_nhat,
        "nhan_2": TEN_NHAN[vi_tri_thu_hai],
        "diem_2": diem_thu_hai,
        "khoang_cach": diem_cao_nhat - diem_thu_hai,
        "cac_diem": [float(x) for x in dau_ra],
    }


def tao_csv_phan_hoi(danh_sach: list[dict]) -> bytes:
    buffer = io.StringIO()
    cot = ["thoi_gian", "nhan_ai_du_doan", "diem_softmax", "nhan_nguoi_danh_gia", "ghi_chu"]
    writer = csv.DictWriter(buffer, fieldnames=cot)
    writer.writeheader()
    writer.writerows(danh_sach)
    return buffer.getvalue().encode("utf-8-sig")


# Khởi tạo vùng nhớ phiên làm việc cho kết quả và phản hồi.
if "ket_qua_hien_tai" not in st.session_state:
    st.session_state.ket_qua_hien_tai = None
if "phan_hoi" not in st.session_state:
    st.session_state.phan_hoi = []

# Phần đầu trang
st.markdown(
    """
    <div class="eco-hero">
      <div class="eco-kicker">DỰ ÁN TIN HỌC 10 · TRÍ TUỆ NHÂN TẠO VÌ MÔI TRƯỜNG</div>
      <h1>♻️ EcoSort AI</h1>
      <p>Trợ lý gợi ý nhận diện và phân loại rác từ hình ảnh</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "EcoSort AI là mô hình thử nghiệm, chỉ hỗ trợ ba nhóm rác thông thường. "
    "Kết quả là gợi ý từ AI, không thay thế việc kiểm tra vật liệu và quy định thu gom tại địa phương."
)

try:
    mo_hinh = tai_mo_hinh()
except Exception as loi:
    st.error(f"Không thể tải mô hình AI: {loi}")
    st.stop()

trang_phan_loai, trang_huong_dan, trang_gioi_thieu = st.tabs(
    ["🔎 Phân loại rác", "📗 Hướng dẫn phân loại", "ℹ️ Về dự án"]
)

with trang_phan_loai:
    st.subheader("1. Cung cấp hình ảnh")
    cot_tai, cot_camera = st.columns(2, gap="large")
    with cot_tai:
        tep_tai_len = st.file_uploader(
            "Tải ảnh từ thiết bị",
            type=["jpg", "jpeg", "png", "webp"],
            max_upload_size=10,
            help="Nên dùng ảnh rõ, đủ sáng và tập trung vào một vật thể. Giới hạn 10 MB/ảnh.",
        )
    with cot_camera:
        tep_camera = st.camera_input(
            "Chụp ảnh bằng camera",
            help="Trình duyệt cần được cấp quyền sử dụng camera.",
        )

    tep_duoc_chon = tep_tai_len if tep_tai_len is not None else tep_camera
    anh_hien_thi = None
    mang_anh = None
    if tep_duoc_chon is not None:
        try:
            anh_hien_thi, mang_anh = chuan_bi_anh(tep_duoc_chon.getvalue())
            st.image(anh_hien_thi, caption="Ảnh đầu vào", width=360)
        except (UnidentifiedImageError, OSError, ValueError) as loi:
            st.error(f"Không đọc được ảnh. Hãy chọn tệp JPG, PNG hoặc WEBP hợp lệ. Chi tiết: {loi}")

    st.subheader("2. Kiểm tra phạm vi ảnh")
    cot_nguy_hai, cot_hon_hop = st.columns(2)
    with cot_nguy_hai:
        co_rac_nguy_hai = st.checkbox(
            "Ảnh có pin, ắc quy, bóng đèn, thuốc, hóa chất hoặc rác điện tử?"
        )
    with cot_hon_hop:
        anh_hon_hop = st.checkbox(
            "Ảnh có nhiều loại rác trộn lẫn hoặc nhiều vật thể?"
        )

    if co_rac_nguy_hai:
        st.warning(RAC_NGOAI_PHAM_VI)
    elif anh_hon_hop:
        st.warning(
            "Ảnh nhiều loại rác nằm ngoài tình huống mô hình được thiết kế để xử lý. "
            "Hãy chụp từng vật thể riêng biệt; không nên dùng một nhãn duy nhất cho cả đống rác."
        )

    st.caption(
        "Mẹo chụp ảnh: đặt một vật thể ở giữa khung hình, đủ sáng, không che khuất; "
        "tránh ảnh chụp quá xa, ảnh ghép hoặc ảnh có nhiều đồ vật."
    )

    nut_bi_vo_hieu = tep_duoc_chon is None or mang_anh is None or co_rac_nguy_hai or anh_hon_hop
    if st.button("🔍 PHÂN LOẠI RÁC", type="primary", use_container_width=True, disabled=nut_bi_vo_hieu):
        try:
            with st.spinner("EcoSort AI đang phân tích hình ảnh..."):
                ket_qua = du_doan(mo_hinh, mang_anh)
                ket_qua["ten_tep"] = getattr(tep_duoc_chon, "name", "Ảnh chụp camera")
                ket_qua["thoi_gian"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.ket_qua_hien_tai = ket_qua
        except Exception as loi:
            st.error(f"Không thể phân loại ảnh: {loi}")
            st.session_state.ket_qua_hien_tai = None

    ket_qua = st.session_state.ket_qua_hien_tai
    if ket_qua is not None:
        st.divider()
        st.subheader("3. Kết quả nhận diện")
        diem = ket_qua["diem"]
        khoang_cach = ket_qua["khoang_cach"]
        if diem < NGUONG_DIEM or khoang_cach < NGUONG_KHOANG_CACH:
            st.warning(
                f"Kết quả chưa đủ rõ. AI đang nghiêng về **{ket_qua['nhan']}**, "
                f"nhưng nhóm đứng thứ hai là **{ket_qua['nhan_2']}**. "
                "Hãy kiểm tra thủ công hoặc chụp lại ảnh."
            )
        else:
            st.success(f"AI gợi ý: **{ket_qua['nhan']}**")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Nhóm AI dự đoán", ket_qua["nhan"])
        with c2:
            st.metric("Điểm đầu ra Softmax", f"{diem * 100:.2f}%")

        st.caption(
            "Điểm Softmax thể hiện mức độ ưu tiên giữa ba nhãn của mô hình; "
            "đây không phải xác suất đã được hiệu chuẩn và không bảo đảm dự đoán đúng."
        )

        st.markdown("**So sánh điểm của ba nhóm**")
        bang_diem = sorted(
            zip(TEN_NHAN, ket_qua["cac_diem"]),
            key=lambda x: x[1],
            reverse=True,
        )
        for ten_nhan, gia_tri in bang_diem:
            st.write(f"**{ten_nhan}:** {gia_tri * 100:.2f}%")
            st.progress(min(max(gia_tri, 0.0), 1.0))

        st.markdown("**Hướng dẫn xử lý tham khảo**")
        st.write(HUONG_DAN[ket_qua["nhan"]])
        st.warning(
            "Nếu vật liệu bị bẩn, là đồ vật nhiều lớp hoặc không rõ thành phần, "
            "hãy kiểm tra hướng dẫn thu gom tại địa phương."
        )

        with st.expander("Gửi phản hồi để con người kiểm tra kết quả"):
            st.write(
                "Phản hồi chỉ được lưu trong phiên sử dụng hiện tại để tải xuống. "
                "Ứng dụng không tự động dùng phản hồi để huấn luyện lại mô hình. "
                "Nhãn người dùng chọn cần được kiểm tra trước khi trở thành dữ liệu huấn luyện."
            )
            with st.form("phan_hoi_form", clear_on_submit=True):
                nhan_nguoi_dung = st.selectbox(
                    "Theo bạn, nhãn phù hợp hơn là:",
                    TEN_NHAN + ["Không thuộc 3 nhóm / chưa thể xác định"],
                )
                ghi_chu = st.text_area(
                    "Ghi chú (không bắt buộc)",
                    placeholder="Ví dụ: ảnh chụp một phần chai nhựa; vật liệu bị trộn...",
                    max_chars=300,
                )
                gui_phan_hoi = st.form_submit_button("Ghi nhận phản hồi")
            if gui_phan_hoi:
                st.session_state.phan_hoi.append({
                    "thoi_gian": datetime.now().isoformat(timespec="seconds"),
                    "nhan_ai_du_doan": ket_qua["nhan"],
                    "diem_softmax": round(diem, 6),
                    "nhan_nguoi_danh_gia": nhan_nguoi_dung,
                    "ghi_chu": ghi_chu,
                })
                st.success("Đã ghi nhận phản hồi trong phiên hiện tại.")

    if st.session_state.phan_hoi:
        with st.expander(f"Phản hồi đã ghi nhận trong phiên này ({len(st.session_state.phan_hoi)})"):
            st.dataframe(st.session_state.phan_hoi, use_container_width=True)
            st.download_button(
                "Tải bảng phản hồi dạng CSV",
                data=tao_csv_phan_hoi(st.session_state.phan_hoi),
                file_name="EcoSort_AI_Phan_hoi_nguoi_dung.csv",
                mime="text/csv",
            )

with trang_huong_dan:
    st.subheader("Hướng dẫn phân loại theo quy ước của EcoSort AI")
    st.write(
        "Ba nhóm dưới đây là quy ước phục vụ mô hình thử nghiệm. "
        "Khả năng thu gom và tái chế thực tế còn phụ thuộc vật liệu, tình trạng rác "
        "và quy định của địa phương."
    )
    for ten, mo_ta in HUONG_DAN.items():
        with st.container(border=True):
            st.markdown(f"### {ten}")
            st.write(mo_ta)
    st.error(RAC_NGOAI_PHAM_VI)
    st.markdown("**Để kết quả ổn định hơn:**")
    st.markdown(
        "- Chụp từng vật thể riêng biệt, không chụp cả đống rác.\n"
        "- Đặt vật thể ở giữa ảnh, đủ sáng và không bị che khuất.\n"
        "- Không dùng ảnh ghép, ảnh quảng cáo hoặc hình minh họa.\n"
        "- Với rác hỗn hợp hoặc vật liệu không rõ thành phần, ưu tiên kiểm tra thủ công."
    )

with trang_gioi_thieu:
    st.subheader("Thông tin dự án")
    st.write(
        "EcoSort AI là sản phẩm thử nghiệm của dự án Tin học 10, "
        "sử dụng MobileNetV2 và học chuyển giao để gợi ý phân loại ảnh rác."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Kiến trúc", "MobileNetV2")
    m2.metric("Số nhóm", "3")
    m3.metric("Phiên bản", "Model B")
    st.markdown("**Kết quả đánh giá đã ghi nhận**")
    st.write(
        "Model B đạt Accuracy 92,64% trên tập xác thực 3.385 ảnh. "
        "Trong lần kiểm thử độc lập ban đầu trên 2.910 ảnh, Accuracy đạt 79,97%. "
        "Kết quả cho thấy mô hình còn hạn chế khi gặp ảnh có phân bố khác dữ liệu huấn luyện, "
        "đặc biệt ở nhóm rác tái chế."
    )
    st.caption(
        "Không diễn giải điểm Softmax của một ảnh thành độ chính xác thực tế. "
        "Mô hình hiện chưa được xác nhận cho rác nguy hại, rác hỗn hợp hoặc mọi điều kiện ngoài đời."
    )
    st.markdown("**Bảo vệ dữ liệu:** Ứng dụng không lưu ảnh tải lên thành bộ sưu tập. Tránh tải ảnh có khuôn mặt, biển số, giấy tờ hoặc thông tin cá nhân; tệp phản hồi chỉ chứa nhãn và ghi chú do người dùng nhập, không chứa ảnh hay thông tin định danh.")
    st.markdown("**Thành viên thực hiện:** Tràn Bảo Minh và Phạm Gia Bách")
    st.markdown("**Nguồn và giới hạn dữ liệu:** Phiên bản mô hình hiện tại được huấn luyện bằng bộ dữ liệu công khai. Nhóm cần bổ sung và ghi nhận riêng ảnh tự chụp trước khi tuyên bố đã đáp ứng yêu cầu dữ liệu tự thu thập trong tiêu chí Mức 4.")

st.divider()
st.caption(
    "EcoSort AI · Dự án Tin học 10 · Tràn Bảo Minh – Phạm Gia Bách | "
    "AI đưa ra gợi ý; người dùng chịu trách nhiệm kiểm tra trước khi phân loại."
)
