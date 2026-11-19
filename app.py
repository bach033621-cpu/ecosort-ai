from __future__ import annotations

import hashlib
import io
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import tensorflow as tf
except Exception:  # Chat and manual lookup can still work if TF is unavailable.
    tf = None

# ============================================================
# ECOSORT AI | CHATBOX PHÂN LOẠI RÁC
# Giữ Streamlit Community Cloud; không cần API trả phí.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.json"
GROUP_MODEL_CANDIDATES = [
    BASE_DIR / "EcoSort_Model_B.keras",
    BASE_DIR / "EcoSort_Model_B_Best.keras",
]
ITEM_MODEL_PATH = BASE_DIR / "EcoSort_Item_Model.keras"
CONFIDENCE_THRESHOLD = 0.65
MAX_IMAGE_BYTES = 5 * 1024 * 1024
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
GROUP_MODEL_CLASS_ORDER = ["huu_co", "rac_con_lai", "tai_che"]

st.set_page_config(
    page_title="EcoSort AI | Trợ lý phân loại rác",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="expanded",
)


def load_knowledge() -> dict[str, Any]:
    try:
        with DATA_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data.get("materials"), dict):
            raise ValueError("data.json thiếu mục materials hợp lệ")
        return data
    except Exception as exc:
        st.error(f"Không đọc được data.json: {exc}")
        st.stop()


KNOWLEDGE = load_knowledge()
GROUPS = KNOWLEDGE.get("groups", {})
MATERIALS = KNOWLEDGE.get("materials", {})


st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .stApp { background: #f7faf8; }
      .block-container { max-width: 920px; padding-top: 1.5rem; padding-bottom: 7rem; }
      [data-testid="stSidebar"] { background: #eef6f0; }
      .eco-header {display:flex;align-items:center;gap:14px;padding:16px 18px;
        background:linear-gradient(120deg,#e4f3e8,#f7fbf8);border:1px solid #d4e9d9;
        border-radius:20px;margin-bottom:14px;}
      .eco-logo {width:48px;height:48px;border-radius:15px;background:#16824e;color:white;
        display:flex;align-items:center;justify-content:center;font-size:25px;flex-shrink:0;}
      .eco-title {font-size:1.55rem;font-weight:800;color:#17452d;line-height:1.15;}
      .eco-subtitle {font-size:.9rem;color:#567362;margin-top:4px;}
      .eco-tag {display:inline-block;padding:4px 10px;border-radius:999px;font-size:.78rem;
        font-weight:700;margin:2px 0 8px 0;}
      .eco-card-title {font-size:1.12rem;font-weight:800;color:#173b2a;margin:0 0 4px 0;}
      .eco-muted {color:#61756a;font-size:.86rem;}
      .eco-section {font-weight:750;color:#244a34;margin-top:10px;margin-bottom:3px;}
      .eco-answer {background:#f2f8f3;border:1px solid #deebe1;border-radius:12px;
        padding:10px 12px;margin:6px 0;}
      div[data-testid="stChatMessage"] {border-radius:16px;}
      div[data-testid="stChatInput"] {background:#fff;}
      div.stButton > button[kind="primary"] {background:#16824e;border-color:#16824e;}
      div.stButton > button[kind="primary"]:hover {background:#11673e;border-color:#11673e;}
      @media (prefers-color-scheme: dark) {
        .eco-header {background:#172d21;border-color:#31563e;}
        .eco-title {color:#e5f4e9;}
        .eco-subtitle,.eco-muted {color:#bfd4c5;}
        .eco-card-title,.eco-section {color:#e2f0e6;}
        .eco-answer {background:#1d3225;border-color:#385844;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "messages": [],
        "last_detection": None,
        "classified_count": 0,
        "estimated_mass_kg": 0.0,
        "group_counts": {},
        "processed_image_hashes": set(),
        "last_camera_hash": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not st.session_state.messages:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "Xin chào! Mình là **EcoSort AI** 🌱\n\n"
                    "Bạn có thể gửi ảnh ngay tại ô chat hoặc mở camera để chụp một vật thể. "
                    "Mình sẽ giúp bạn tìm hiểu vật liệu, nhóm rác và cách xử lý phù hợp."
                ),
            }
        )


init_state()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d")
    return re.sub(r"\s+", " ", value).strip()


def group_info(group_id: str) -> dict[str, str]:
    return GROUPS.get(
        group_id,
        {
            "display_name": "Chưa xác định",
            "color": "#667085",
            "description": "Chưa có thông tin cho nhóm rác này.",
        },
    )


def find_material_from_text(text: str) -> tuple[str, dict[str, Any]] | None:
    normalized = normalize_text(text)
    matches: list[tuple[int, str, dict[str, Any]]] = []
    for material_id, material in MATERIALS.items():
        names = [material.get("display_name", ""), *material.get("aliases", [])]
        for name in names:
            candidate = normalize_text(str(name))
            if candidate and candidate in normalized:
                matches.append((len(candidate), material_id, material))
    if not matches:
        return None
    _, material_id, material = max(matches, key=lambda item: item[0])
    return material_id, material


def resize_for_model(image: Image.Image, model: Any) -> np.ndarray:
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    height = int(input_shape[1]) if input_shape and input_shape[1] else 224
    width = int(input_shape[2]) if input_shape and input_shape[2] else 224
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


@st.cache_resource(show_spinner="Đang tải mô hình nhận diện...")
def load_models() -> tuple[Any | None, Any | None, str | None]:
    if tf is None:
        return None, None, "TensorFlow chưa được cài trong môi trường này."

    item_model = None
    group_model = None
    warnings: list[str] = []

    if ITEM_MODEL_PATH.exists():
        try:
            item_model = tf.keras.models.load_model(ITEM_MODEL_PATH, compile=False)
            class_order = KNOWLEDGE.get("item_model_classes", [])
            output_size = int(item_model.output_shape[-1])
            if len(class_order) != output_size:
                warnings.append(
                    "Có EcoSort_Item_Model.keras nhưng item_model_classes trong data.json "
                    "không khớp số đầu ra; ứng dụng sẽ bỏ qua model vật thể chi tiết."
                )
                item_model = None
        except Exception as exc:
            warnings.append(f"Không tải được mô hình vật thể chi tiết: {exc}")
            item_model = None

    for path in GROUP_MODEL_CANDIDATES:
        if path.exists():
            try:
                candidate = tf.keras.models.load_model(path, compile=False)
                if int(candidate.output_shape[-1]) == len(GROUP_MODEL_CLASS_ORDER):
                    group_model = candidate
                    break
                warnings.append(f"Bỏ qua {path.name}: đầu ra không có 3 lớp như cấu hình Model B.")
            except Exception as exc:
                warnings.append(f"Không tải được {path.name}: {exc}")

    if item_model is None and group_model is None:
        warnings.append(
            "Không tìm thấy model tương thích. Có thể chat bằng cách gõ tên vật liệu, "
            "nhưng chưa thể nhận diện ảnh tự động."
        )
    return item_model, group_model, "\n".join(warnings) if warnings else None


def predict_scores(model: Any, image: Image.Image) -> np.ndarray:
    array = resize_for_model(image, model)
    scores = np.asarray(model.predict(array, verbose=0))[0].astype(float)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("Mô hình trả về điểm không hợp lệ.")
    if np.any(scores < 0) or np.any(scores > 1.001):
        # Model outputs are expected to be probabilities from a softmax head.
        raise ValueError("Đầu ra mô hình không phải điểm Softmax trong khoảng 0–1.")
    return scores


def make_top3(scores: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    order = np.argsort(scores)[::-1][:3]
    return [
        {"label": labels[int(index)], "confidence": float(scores[int(index)])}
        for index in order
    ]


def detect_image(image_bytes: bytes) -> dict[str, Any]:
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Ảnh vượt quá giới hạn 5 MB. Hãy chọn ảnh nhỏ hơn.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Không đọc được ảnh. Hãy gửi ảnh JPG, PNG hoặc WEBP hợp lệ.") from exc

    item_model, group_model, _ = load_models()

    # Ưu tiên mô hình nhiều lớp vật thể nếu nhóm đã huấn luyện và đặt đúng tên file.
    if item_model is not None:
        item_labels = KNOWLEDGE.get("item_model_classes", [])
        item_scores = predict_scores(item_model, image)
        top3 = make_top3(item_scores, item_labels)
        best = top3[0]
        confidence = float(best["confidence"])
        material_id = str(best["label"])
        material = MATERIALS.get(material_id)
        if material is None:
            return {
                "display_name": "Vật thể chưa có hướng dẫn",
                "material_id": material_id,
                "group_id": "unknown",
                "confidence": confidence,
                "top3": top3,
                "uncertain": True,
                "source": "item_model",
                "image": image,
                "note": "Nhãn dự đoán chưa có mục tương ứng trong data.json.",
            }
        return {
            "display_name": material["display_name"],
            "material_id": material_id,
            "group_id": material["group"],
            "confidence": confidence,
            "top3": top3,
            "uncertain": confidence < CONFIDENCE_THRESHOLD,
            "source": "item_model",
            "image": image,
        }

    # Model B hiện tại chỉ có ba đầu ra nhóm rác, không thể suy ra tên vật thể cụ thể.
    if group_model is None:
        return {
            "display_name": "Chưa thể nhận diện ảnh",
            "material_id": None,
            "group_id": "unknown",
            "confidence": 0.0,
            "top3": [],
            "uncertain": True,
            "source": "no_model",
            "image": image,
            "note": "Chưa có mô hình phù hợp trong repository.",
        }

    group_scores = predict_scores(group_model, image)
    top3 = make_top3(group_scores, GROUP_MODEL_CLASS_ORDER)
    best = top3[0]
    group_id = str(best["label"])
    group_map = {
        "huu_co": "organic",
        "rac_con_lai": "residual",
        "tai_che": "recyclable",
    }
    mapped_group = group_map.get(group_id, "unknown")
    confidence = float(best["confidence"])
    return {
        "display_name": "Chưa xác định vật thể cụ thể",
        "material_id": None,
        "group_id": mapped_group,
        "confidence": confidence,
        "top3": [
            {
                "label": group_info(group_map.get(item["label"], "unknown"))["display_name"],
                "confidence": item["confidence"],
            }
            for item in top3
        ],
        "uncertain": confidence < CONFIDENCE_THRESHOLD,
        "source": "group_model",
        "image": image,
        "note": (
            "Model B hiện chỉ được huấn luyện để dự đoán 3 nhóm rác, "
            "chưa phân biệt chai nhựa, túi nilon hay từng loại vỏ trái cây."
        ),
    }


def update_impact(detection: dict[str, Any], image_hash: str) -> None:
    if detection.get("uncertain"):
        return
    processed = st.session_state.processed_image_hashes
    if image_hash in processed:
        return
    processed.add(image_hash)
    st.session_state.classified_count += 1
    group_id = detection.get("group_id", "unknown")
    st.session_state.group_counts[group_id] = st.session_state.group_counts.get(group_id, 0) + 1
    material = MATERIALS.get(detection.get("material_id") or "", {})
    default_mass = float(KNOWLEDGE.get("impact", {}).get("default_item_mass_kg", 0.05))
    mass = float(material.get("estimated_item_mass_kg", default_mass))
    st.session_state.estimated_mass_kg += max(mass, 0.0)


def append_detection(image_bytes: bytes, filename: str, caption: str = "") -> dict[str, Any]:
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    with st.spinner("EcoSort AI đang phân tích ảnh..."):
        detection = detect_image(image_bytes)
    detection["filename"] = filename
    detection["timestamp"] = datetime.now().isoformat(timespec="seconds")
    detection["image_hash"] = image_hash
    detection["caption"] = caption
    st.session_state.messages.append(
        {"role": "user", "content": caption or "Mình gửi ảnh này để nhận diện.", "image_bytes": image_bytes}
    )
    st.session_state.messages.append(
        {"role": "assistant", "content": "Kết quả phân tích ảnh:", "detection": detection}
    )
    st.session_state.last_detection = detection
    update_impact(detection, image_hash)
    return detection


def answer_question(question: str) -> str:
    normalized = normalize_text(question)
    found = find_material_from_text(question)
    last = st.session_state.last_detection or {}

    if found:
        material_id, material = found
        st.session_state.last_detection = {
            "display_name": material["display_name"],
            "material_id": material_id,
            "group_id": material["group"],
            "confidence": None,
            "uncertain": False,
            "source": "text_lookup",
        }
        return (
            f"Theo tên vật liệu bạn vừa nhập, **{material['display_name']}** thuộc nhóm "
            f"**{group_info(material['group'])['display_name']}**. "
            "Mình đã ghi nhớ vật liệu này cho câu hỏi tiếp theo. Xem thẻ thông tin bên dưới nhé."
        )

    material_id = last.get("material_id")
    material = MATERIALS.get(material_id or "")

    if material is None:
        if last.get("source") == "group_model":
            return (
                "Ảnh gần nhất mới được mô hình xếp vào một nhóm rác, chưa xác định được tên vật liệu cụ thể. "
                "Bạn có thể gõ tên món đồ, ví dụ **chai nhựa**, **túi nilon** hoặc **vỏ cam**, "
                "để mình tra hướng dẫn phù hợp."
            )
        return (
            "Bạn hãy gửi ảnh hoặc gõ tên vật liệu cụ thể (ví dụ: chai nhựa, lon nhôm, vỏ cam). "
            "Mình sẽ dựa vào danh mục hướng dẫn trong EcoSort AI để trả lời nhé."
        )

    faq = material.get("faq", {})
    keyword_map = KNOWLEDGE.get("question_keywords", {})
    matched_faq = None
    for faq_key, keywords in keyword_map.items():
        if any(normalize_text(keyword) in normalized for keyword in keywords):
            matched_faq = faq_key
            break

    if matched_faq and matched_faq in faq:
        return str(faq[matched_faq])

    if any(term in normalized for term in ["tai che duoc khong", "co tai che", "tai che khong"]):
        return str(faq.get("recycle", material.get("disposal_guide", "Hãy kiểm tra hướng dẫn tại địa phương.")))
    if any(term in normalized for term in ["bo vao thung nao", "nhom rac nao", "phan loai vao dau"]):
        return (
            f"**{material['display_name']}** thuộc nhóm **{group_info(material['group'])['display_name']}**. "
            f"{material.get('disposal_guide', '')}"
        )
    if any(term in normalized for term in ["tai su dung", "tan dung", "lam gi voi", "y tuong"]):
        return str(material.get("reuse_tips", "Chưa có gợi ý tái sử dụng cụ thể."))
    return (
        f"Với **{material['display_name']}**, {material.get('short_explanation', '')} "
        f"{material.get('disposal_guide', '')}"
    ).strip()


def render_detection_card(detection: dict[str, Any]) -> None:
    group_id = detection.get("group_id", "unknown")
    group = group_info(group_id)
    material = MATERIALS.get(detection.get("material_id") or "", {})
    color = group.get("color", "#667085")
    display_name = detection.get("display_name", "Chưa xác định")
    confidence = detection.get("confidence")
    uncertain = bool(detection.get("uncertain"))

    with st.container(border=True):
        preview = detection.get("image")
        if preview is not None:
            preview_col, detail_col = st.columns([1, 2.8], gap="medium")
            with preview_col:
                st.image(preview, caption="Ảnh đã gửi", use_container_width=True)
            with detail_col:
                st.markdown(f'<div class="eco-card-title">{display_name}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<span class="eco-tag" style="background:{color}1A;color:{color};border:1px solid {color}55">'
                    f'{group.get("display_name", "Chưa xác định")}</span>',
                    unsafe_allow_html=True,
                )
                if confidence is not None:
                    st.markdown(f"**Độ tin cậy của mô hình: {confidence * 100:.1f}%**")
                    st.progress(min(max(float(confidence), 0.0), 1.0))
                    st.caption("Điểm Softmax chỉ là mức độ ưu tiên của mô hình, không phải xác suất đã hiệu chuẩn.")
                else:
                    st.caption("Vật liệu được tra cứu từ tên bạn nhập, không phải dự đoán ảnh.")
        else:
            st.markdown(f'<div class="eco-card-title">{display_name}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="eco-tag" style="background:{color}1A;color:{color};border:1px solid {color}55">'
                f'{group.get("display_name", "Chưa xác định")}</span>',
                unsafe_allow_html=True,
            )
            if confidence is not None:
                st.markdown(f"**Độ tin cậy của mô hình: {confidence * 100:.1f}%**")
                st.progress(min(max(float(confidence), 0.0), 1.0))
                st.caption("Điểm Softmax chỉ là mức độ ưu tiên của mô hình, không phải xác suất đã hiệu chuẩn.")
            else:
                st.caption("Vật liệu được tra cứu từ tên bạn nhập, không phải dự đoán ảnh.")

        if uncertain:
            st.warning(
                "Mình chưa nhận diện rõ vật thể (độ tin cậy dưới 65%). "
                "Bạn thử chụp gần hơn, đủ sáng, chỉ lấy một vật thể; hoặc gõ tên món đồ nhé."
            )
            if detection.get("top3"):
                st.markdown("**Ba kết quả mô hình đang cân nhắc:**")
                for candidate in detection["top3"]:
                    st.write(f"• {candidate['label']}: {candidate['confidence'] * 100:.1f}%")
        elif detection.get("source") == "group_model":
            st.info(
                detection.get(
                    "note",
                    "Mô hình hiện nhận diện nhóm rác, chưa xác định tên vật thể cụ thể.",
                )
            )
        else:
            guide = material if material else group
            st.markdown('<div class="eco-section">Cách xử lý</div>', unsafe_allow_html=True)
            st.write(guide.get("disposal_guide", "Hãy kiểm tra quy định thu gom tại địa phương."))
            st.markdown('<div class="eco-section">Tại sao cần xử lý như vậy?</div>', unsafe_allow_html=True)
            st.write(guide.get("why", group.get("description", "Phân loại đúng giúp quản lý rác phù hợp hơn.")))
            st.markdown('<div class="eco-section">Gợi ý tái sử dụng</div>', unsafe_allow_html=True)
            st.write(guide.get("reuse_tips", "Ưu tiên tái sử dụng an toàn nếu vật liệu còn phù hợp."))
            collection = guide.get("collection_points")
            if collection:
                st.markdown('<div class="eco-section">Gợi ý điểm thu gom</div>', unsafe_allow_html=True)
                st.write(collection)
                st.caption("Đây là gợi ý loại điểm thu gom; hãy kiểm tra địa chỉ và lịch nhận tại địa phương.")

        if detection.get("note") and detection.get("source") != "group_model":
            st.caption(detection["note"])


def render_message(message: dict[str, Any], index: int) -> None:
    role = message.get("role", "assistant")
    with st.chat_message(role, avatar="🧑" if role == "user" else "♻️"):
        if message.get("content"):
            st.markdown(message["content"])
        if message.get("image_bytes"):
            st.image(message["image_bytes"], use_container_width=True)
        if message.get("detection"):
            render_detection_card(message["detection"])
            quick_replies = message["detection"].get("quick_replies", [])
            if not quick_replies:
                material = MATERIALS.get(message["detection"].get("material_id") or "", {})
                fallback_group = group_info(message["detection"].get("group_id", "unknown"))
                quick_replies = material.get("quick_replies", fallback_group.get("quick_replies", []))
            if index == len(st.session_state.messages) - 1 and quick_replies:
                st.caption("Bạn có thể hỏi nhanh:")
                cols = st.columns(min(3, len(quick_replies)))
                for idx, label in enumerate(quick_replies[:3]):
                    with cols[idx]:
                        if st.button(label, key=f"quick_{index}_{idx}", use_container_width=True):
                            st.session_state.pending_question = label
                            st.rerun()


def add_text_exchange(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    found = find_material_from_text(question)
    if found:
        material_id, material = found
        detection = {
            "display_name": material["display_name"],
            "material_id": material_id,
            "group_id": material["group"],
            "confidence": None,
            "uncertain": False,
            "source": "text_lookup",
            "quick_replies": material.get("quick_replies", []),
        }
        st.session_state.last_detection = detection
        text_hash = hashlib.sha256(
            f"text:{material_id}:{len(st.session_state.messages)}".encode("utf-8")
        ).hexdigest()
        update_impact(detection, text_hash)
        st.session_state.messages.append(
            {"role": "assistant", "content": "Mình tìm thấy vật liệu trong danh mục hướng dẫn:", "detection": detection}
        )
    else:
        reply = answer_question(question)
        st.session_state.messages.append({"role": "assistant", "content": reply})


def process_image_upload(uploaded_file: Any, caption: str = "") -> None:
    image_bytes = uploaded_file.getvalue()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        st.session_state.messages.append({"role": "user", "content": caption or "Mình đã gửi một ảnh."})
        st.session_state.messages.append(
            {"role": "assistant", "content": "Ảnh vượt quá giới hạn 5 MB. Hãy chọn ảnh nhỏ hơn nhé."}
        )
        return
    try:
        append_detection(image_bytes, getattr(uploaded_file, "name", "anh_camera.jpg"), caption)
    except Exception as exc:
        st.session_state.messages.append({"role": "assistant", "content": f"Mình chưa xử lý được ảnh: {exc}"})


# Sidebar: Eco-Impact Tracker (số liệu ước tính vui, không phải đo lường thực tế).
with st.sidebar:
    st.markdown("## 🌱 Eco-Impact Tracker")
    st.metric("Vật thể đã phân loại trong phiên", st.session_state.classified_count)
    st.metric("Khối lượng phân loại ước tính", f"{st.session_state.estimated_mass_kg:.2f} kg")
    st.caption(
        "Ước tính minh họa: khối lượng trung bình giả định theo từng vật liệu. "
        "Không phải lượng rác thực tế đã giảm hay số liệu đo phát thải CO₂."
    )
    if st.session_state.group_counts:
        st.markdown("**Theo nhóm rác**")
        for group_id, count in st.session_state.group_counts.items():
            st.write(f"{group_info(group_id)['display_name']}: **{count}**")
    st.divider()
    st.markdown("**Mẹo sử dụng**")
    st.caption("Chụp một vật thể mỗi ảnh, đủ sáng và không bị che khuất.")
    if st.button("🧹 Xóa cuộc trò chuyện", use_container_width=True):
        for key in ["messages", "last_detection", "classified_count", "estimated_mass_kg", "group_counts", "processed_image_hashes", "last_camera_hash", "pending_question"]:
            st.session_state.pop(key, None)
        st.rerun()

# Header
st.markdown(
    """
    <div class="eco-header">
      <div class="eco-logo">♻️</div>
      <div>
        <div class="eco-title">EcoSort AI</div>
        <div class="eco-subtitle">Trò chuyện với trợ lý phân loại rác thông minh</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Gửi ảnh để nhận diện hoặc nhập tên vật liệu để hỏi cách phân loại, xử lý và tái sử dụng.")

# Show model status without blocking manual chat.
item_model, group_model, model_warning = load_models()
if model_warning:
    st.warning(model_warning)

# Process quick reply selected from an earlier render.
pending_question = st.session_state.pop("pending_question", None)
if pending_question:
    add_text_exchange(pending_question)

# Render conversation history.
for message_index, message in enumerate(st.session_state.messages):
    render_message(message, message_index)

# Camera is immediately adjacent to the chat composer. Upload is built into st.chat_input.
with st.popover("📷 Chụp ảnh bằng camera", use_container_width=True):
    camera_file = st.camera_input(
        "Đưa một vật thể vào giữa khung hình rồi chụp",
        key="ecosort_camera_capture",
    )
    st.caption("Nếu không muốn cấp quyền camera, bạn vẫn có thể đính kèm ảnh ở thanh chat bên dưới.")

if camera_file is not None:
    camera_bytes = camera_file.getvalue()
    camera_hash = hashlib.sha256(camera_bytes).hexdigest()
    if camera_hash != st.session_state.last_camera_hash:
        st.session_state.last_camera_hash = camera_hash
        try:
            process_image_upload(camera_file, "Ảnh chụp từ camera")
        except Exception as exc:
            st.session_state.messages.append({"role": "assistant", "content": f"Không xử lý được ảnh camera: {exc}"})
        st.rerun()

submission = st.chat_input(
    "Nhập câu hỏi hoặc đính kèm ảnh rác...",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp"],
    max_upload_size=5,
    key="ecosort_chat_input",
)

if submission is not None:
    text = str(getattr(submission, "text", "") or "").strip()
    files = list(getattr(submission, "files", []) or [])
    if text and not files:
        add_text_exchange(text)
        st.rerun()
    elif files:
        for uploaded in files:
            process_image_upload(uploaded, text)
        if text:
            # process_image_upload stores the text once beside the attached image;
            # answer it against the new detection context without duplicating the user bubble.
            st.session_state.messages.append({"role": "assistant", "content": answer_question(text)})
        st.rerun()

st.markdown(
    '<div class="eco-muted" style="text-align:center;margin-top:18px">'
    'EcoSort AI chỉ đưa ra gợi ý. Với pin, rác điện tử, hóa chất hoặc vật liệu không rõ thành phần, '
    'hãy để riêng và nhờ người lớn/đơn vị thu gom phù hợp hướng dẫn.'
    '</div>',
    unsafe_allow_html=True,
)
