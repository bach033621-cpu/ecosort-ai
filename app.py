from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # Local compatibility fallback; production uses ai-edge-litert.
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except ImportError:
        Interpreter = None  # type: ignore

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    ROOT / "EcoSort_40Class_Optimized.tflite",
    ROOT / "EcoSort_40Class.tflite",
]
LABELS_PATH = ROOT / "ecosort_labels.json"
GUIDE_PATH = ROOT / "waste_guide.json"

CONFIDENCE_THRESHOLD = 0.65
MARGIN_THRESHOLD = 0.08
GEMINI_MODEL_DEFAULT = "gemini-3.5-flash-lite"
MAX_HISTORY_TURNS = 12
MAX_IMAGE_BYTES = 12 * 1024 * 1024

GROUP_COLORS = {
    "Rác hữu cơ": "#16834A",
    "Rác tái chế": "#2878C8",
    "Rác điện tử": "#7656C5",
    "Rác nguy hại": "#D04444",
    "Rác còn lại": "#737780",
    "Cần kiểm tra": "#B7791F",
    "Ưu tiên tái sử dụng": "#0F8B8D",
}

st.markdown(
    """
    <style>
      .block-container {max-width: 1050px; padding-top: 1.6rem; padding-bottom: 6rem;}
      [data-testid="stChatMessage"] {border-radius: 16px; padding: 0.7rem 0.9rem;}
      .ecosort-subtitle {color: #718096; margin-top: -0.5rem;}
      .ecosort-section {font-size: 1.02rem; font-weight: 700; margin: 0.4rem 0 0.2rem 0;}
      .ecosort-muted {color: #718096; font-size: 0.88rem;}
      .ecosort-chip {display: inline-block; border-radius: 999px; padding: 0.28rem 0.65rem; font-size: 0.82rem; font-weight: 650;}
      div[data-testid="stMetric"] {background: rgba(128,128,128,0.06); border: 1px solid rgba(128,128,128,0.15); padding: 0.7rem; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Load local recognition assets
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_labels(path_text: str, modified: float) -> list[str]:
    del modified
    with open(path_text, "r", encoding="utf-8") as file:
        labels = json.load(file)
    if not isinstance(labels, list) or not labels:
        raise ValueError("ecosort_labels.json phải là một danh sách nhãn.")
    return [str(item) for item in labels]


@st.cache_data(show_spinner=False)
def load_guide(path_text: str, modified: float) -> dict[str, dict[str, Any]]:
    del modified
    with open(path_text, "r", encoding="utf-8") as file:
        guide = json.load(file)
    if not isinstance(guide, dict):
        raise ValueError("waste_guide.json phải là JSON object.")
    return guide


@st.cache_resource(show_spinner="Đang tải mô hình nhận diện EcoSort...")
def load_interpreter(model_path: str, modified: float, size: int):
    del modified, size
    if Interpreter is None:
        raise RuntimeError(
            "Không tìm thấy LiteRT. Hãy kiểm tra requirements.txt có ai-edge-litert."
        )
    interpreter = Interpreter(model_path=model_path, num_threads=2)
    interpreter.allocate_tensors()
    return interpreter


def load_knowledge_assets():
    missing = [p.name for p in [LABELS_PATH, GUIDE_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError("Thiếu file: " + ", ".join(missing))

    labels = load_labels(str(LABELS_PATH), LABELS_PATH.stat().st_mtime)
    guide = load_guide(str(GUIDE_PATH), GUIDE_PATH.stat().st_mtime)
    missing_guides = sorted(set(labels) - set(guide))
    if missing_guides:
        raise ValueError("Thiếu hướng dẫn cho nhãn: " + ", ".join(missing_guides))
    return labels, guide


def load_model_assets(labels: list[str]):
    model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            "Không tìm thấy EcoSort_40Class_Optimized.tflite hoặc EcoSort_40Class.tflite."
        )

    interpreter = load_interpreter(
        str(model_path), model_path.stat().st_mtime, model_path.stat().st_size
    )
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    output_shape = output_details.get("shape", [])
    if len(output_shape) < 2 or int(output_shape[-1]) != len(labels):
        raise ValueError(
            f"Model xuất {output_shape}, nhưng ecosort_labels.json có {len(labels)} nhãn. "
            "Thứ tự hoặc số lượng nhãn không khớp model."
        )

    return interpreter, input_details, output_details, model_path


def _prepare_input(image: Image.Image, input_details: dict) -> np.ndarray:
    shape = input_details["shape"]
    if len(shape) != 4:
        raise ValueError(f"Kích thước input model không hợp lệ: {shape}")

    height, width = int(shape[1]), int(shape[2])
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)

    # Important: training used MobileNetV2 preprocess_input = pixel / 127.5 - 1.
    array = np.asarray(image, dtype=np.float32)
    array = (array / 127.5) - 1.0
    array = np.expand_dims(array, axis=0)

    expected_dtype = input_details["dtype"]
    if expected_dtype in (np.uint8, np.int8):
        scale, zero_point = input_details.get("quantization", (0.0, 0))
        if not scale:
            raise ValueError("Model quantized nhưng thiếu thông số quantization.")
        array = np.round(array / scale + zero_point)
        limits = np.iinfo(expected_dtype)
        array = np.clip(array, limits.min, limits.max).astype(expected_dtype)
    else:
        array = array.astype(expected_dtype, copy=False)

    return array


def predict_image(
    image: Image.Image,
    labels: list[str],
    interpreter,
    input_details: dict,
    output_details: dict,
) -> dict[str, Any]:
    input_tensor = _prepare_input(image, input_details)
    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    output = np.asarray(output, dtype=np.float32)
    if output_details["dtype"] in (np.uint8, np.int8):
        scale, zero_point = output_details.get("quantization", (0.0, 0))
        if scale:
            output = (output.astype(np.float32) - zero_point) * scale

    # Model output is a softmax vector. Normalize defensively for display.
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    output = np.clip(output, 0.0, None)
    total = float(output.sum())
    if total <= 0:
        raise ValueError("Model không trả về xác suất hợp lệ.")
    probabilities = output / total

    order = np.argsort(probabilities)[::-1]
    top_idx = int(order[0])
    second_idx = int(order[1]) if len(order) > 1 else top_idx
    confidence = float(probabilities[top_idx])
    margin = confidence - float(probabilities[second_idx])

    uncertain = confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD
    top_items = [
        {"label": labels[int(i)], "confidence": float(probabilities[int(i)])}
        for i in order[:3]
    ]

    return {
        "label": labels[top_idx],
        "confidence": confidence,
        "margin": margin,
        "uncertain": uncertain,
        "top_items": top_items,
    }


# -----------------------------------------------------------------------------
# Chat assistant: Gemini API when configured; safe local fallback otherwise
# -----------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """
Bạn là EcoSort AI, trợ lý giải thích nhận diện và xử lý rác bằng tiếng Việt.
Quy tắc:
- Trả lời tự nhiên như một trợ lý hội thoại, hiểu các câu hỏi tiếp nối và dùng ngữ cảnh chat đã cung cấp.
- Ưu tiên giải thích kỹ: vật thể là gì, vật liệu/thành phần thường gặp, vì sao thuộc nhóm đó, chuẩn bị trước khi bỏ, từng bước xử lý, tái sử dụng/tái chế, lưu ý an toàn và trường hợp cần kiểm tra quy định địa phương.
- Dùng hồ sơ EcoSort được cung cấp làm nguồn ưu tiên; chỉ áp dụng kết quả nhận diện gần nhất khi câu hỏi đang nói về vật thể đó. Không tự bịa địa chỉ, điểm thu gom, quy định cụ thể của một địa phương, thành phần hóa học chính xác hoặc khả năng tái chế nếu dữ liệu không xác nhận.
- Phân biệt rõ điều chắc chắn và điều phụ thuộc vật liệu/đơn vị thu gom. Với vật thể nguy hại hoặc rác điện tử, khuyên người dùng không tự tháo, đốt, đập, chọc thủng hoặc tiếp xúc với phần bên trong; hướng dẫn chuyển đến điểm thu gom phù hợp.
- Model ảnh chỉ nhận diện 40 lớp đã được huấn luyện. Không khẳng định đã nhận diện vật thể ngoài danh sách.
- Nếu độ tin cậy thấp, nói rõ chưa chắc chắn và yêu cầu ảnh rõ hơn hoặc mô tả thêm; không kết luận chắc chắn.
- Nếu không biết, nói thẳng là chưa đủ thông tin và đề nghị cách kiểm tra an toàn.
- Không đưa ra câu trả lời một dòng khi người dùng hỏi cách xử lý chi tiết. Dùng tiêu đề/ngắn gọn và các bước dễ làm theo.
""".strip()


def get_api_key() -> str | None:
    try:
        value = st.secrets.get("GEMINI_API_KEY", "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "").strip() or None


def get_gemini_model() -> str:
    try:
        return str(st.secrets.get("GEMINI_MODEL", GEMINI_MODEL_DEFAULT))
    except Exception:
        return GEMINI_MODEL_DEFAULT


def make_llm_contents(messages: list[dict[str, Any]], latest_prompt: str):
    contents = []
    if genai_types is None:
        raise RuntimeError("Chưa cài google-genai.")

    history_messages = messages
    # The latest user message is already stored in Streamlit state before the API call.
    # Exclude it here because latest_prompt is appended separately below.
    if (
        history_messages
        and history_messages[-1].get("role") == "user"
        and str(history_messages[-1].get("llm_text", history_messages[-1].get("content", ""))).strip()
        == latest_prompt.strip()
    ):
        history_messages = history_messages[:-1]

    for message in history_messages[-(MAX_HISTORY_TURNS * 2):]:
        text = str(message.get("llm_text", message.get("content", ""))).strip()
        role = message.get("role")
        if not text or role not in {"user", "assistant"}:
            continue
        api_role = "user" if role == "user" else "model"
        contents.append(
            genai_types.Content(
                role=api_role,
                parts=[genai_types.Part.from_text(text=text)],
            )
        )

    contents.append(
        genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=latest_prompt)],
        )
    )
    return contents


def ask_gemini(question: str, guide: dict, labels: list[str]) -> str:
    if genai is None or genai_types is None:
        raise RuntimeError("Thư viện google-genai chưa được cài đặt.")

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("Chưa cấu hình GEMINI_API_KEY trong Streamlit Secrets.")

    current = st.session_state.get("current_detection")
    context = ""
    if current:
        profile = guide.get(current.get("label", ""), {})
        context = (
            "\n\nNgữ cảnh nhận diện gần nhất của người dùng (không phải ảnh gửi cho Gemini):\n"
            + json.dumps(
                {
                    "prediction": current,
                    "waste_profile": profile,
                },
                ensure_ascii=False,
            )
        )

    prompt = question.strip() + context
    contents = make_llm_contents(st.session_state.chat_messages, prompt)

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=get_gemini_model(),
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.35,
            max_output_tokens=1500,
        ),
    )
    answer = (response.text or "").strip()
    if not answer:
        raise RuntimeError("Trợ lý AI không trả về nội dung. Hãy gửi lại câu hỏi.")
    return answer


def format_profile_as_text(label: str, profile: dict[str, Any]) -> str:
    lines = [
        f"{profile.get('name', label)} ({profile.get('english_name', label)})",
        f"Nhóm: {profile.get('group_name', 'Cần kiểm tra')}",
        f"Đây là gì: {profile.get('what_it_is', 'Chưa có mô tả.')}",
        f"Vì sao thuộc nhóm này: {profile.get('why_group', 'Cần kiểm tra vật liệu và quy định tiếp nhận.')}",
        "Cách xử lý: " + "; ".join(profile.get("preparation_steps", []) + profile.get("disposal_steps", [])),
        "Tái sử dụng: " + "; ".join(profile.get("reuse_options", [])),
        "Lưu ý an toàn: " + "; ".join(profile.get("safety_notes", [])),
    ]
    return "\n".join(lines)


def local_fallback(question: str, guide: dict) -> str:
    """Knowledge-base fallback. This is not presented as a general LLM."""
    q = question.casefold()
    current = st.session_state.get("current_detection")

    follow_up_markers = [
        "này", "nó", "cái này", "vật này", "rác này", "xử lý", "phân loại",
        "tái chế", "vứt", "bỏ đi", "thu gom", "tận dụng", "vì sao", "là gì",
        "cách nào", "làm sao", "an toàn", "đặc điểm", "hướng dẫn"
    ]
    refers_to_current = any(marker in q for marker in follow_up_markers)

    # Use the latest recognition only when the message looks like a follow-up.
    if current and refers_to_current:
        label = current.get("label")
        profile = guide.get(label)
        if profile:
            if current.get("uncertain"):
                return (
                    "Mình vẫn đang giữ kết quả nhận diện gần nhất, nhưng model chưa đủ chắc chắn. "
                    "Bạn gửi ảnh rõ hơn hoặc mô tả vật thể giúp mình nhé.\n\n"
                    + format_profile_as_text(label, profile)
                )
            return (
                "Mình đang dựa trên vật thể bạn vừa quét. Đây là hướng dẫn trong dữ liệu EcoSort:\n\n"
                + format_profile_as_text(label, profile)
                + "\n\nBạn có thể hỏi tiếp về một bước cụ thể. Lưu ý: chatbot đang ở chế độ tra cứu cục bộ, "
                "chưa có mô hình ngôn ngữ nên chưa thể trò chuyện mở rộng như ChatGPT."
            )

    # Search the 40 local profiles by Vietnamese/English names and aliases.
    for label, profile in guide.items():
        terms = [label.replace("_", " ").casefold()]
        terms.extend(x.casefold() for x in profile.get("keywords", []))
        terms.append(str(profile.get("name", "")).casefold())
        if any(term and term in q for term in terms):
            return format_profile_as_text(label, profile) + (
                "\n\nĐây là câu trả lời tra cứu từ hồ sơ EcoSort. "
                "Để có hội thoại tự nhiên nhiều lượt, cần cấu hình mô hình ngôn ngữ."
            )

    if any(greeting in q for greeting in ["xin chào", "chào", "hello", "hi", "hey"]):
        return (
            "Chào bạn! ♻️ Mình là EcoSort. Bạn có thể gửi ảnh rác để nhận diện hoặc hỏi về "
            "các loại rác, cách phân loại và xử lý. Hiện mình đang ở chế độ tra cứu cục bộ; "
            "hội thoại mở rộng sẽ hoạt động khi quản trị viên cấu hình mô hình ngôn ngữ."
        )

    return (
        "Mình vẫn nhận và lưu câu hỏi của bạn, nhưng hiện chưa có khóa mô hình ngôn ngữ nên "
        "chỉ tra cứu được hồ sơ của 40 loại rác EcoSort đã hỗ trợ. "
        "Bạn có thể hỏi về một vật thể trong danh sách hoặc nhờ người quản lý dự án cấu hình "
        "GEMINI_API_KEY trong Streamlit Secrets để bật hội thoại AI nhiều lượt."
    )


# -----------------------------------------------------------------------------
# Rendering helpers
# -----------------------------------------------------------------------------
def render_profile(profile: dict[str, Any], label: str, prediction: dict[str, Any]):
    group_name = profile.get("group_name", "Cần kiểm tra")
    color = GROUP_COLORS.get(group_name, "#737780")
    st.markdown(
        f'<span class="ecosort-chip" style="background:{color}18;color:{color};">{group_name}</span>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    col1.metric("Độ tin cậy dự đoán", f"{prediction['confidence'] * 100:.1f}%")
    col2.metric("Tên nhãn", label)

    st.markdown("#### Đây là rác gì?")
    st.write(profile.get("what_it_is", "Chưa có mô tả."))
    if profile.get("common_materials"):
        st.markdown("**Vật liệu / thành phần thường gặp:** " + profile["common_materials"])

    st.markdown("#### Vì sao được xếp vào nhóm này?")
    st.write(profile.get("why_group", "Cần kiểm tra vật liệu và quy định tiếp nhận tại địa phương."))

    st.markdown("#### Cách xử lý từng bước")
    steps = profile.get("preparation_steps", []) + profile.get("disposal_steps", [])
    for index, step in enumerate(steps, start=1):
        st.markdown(f"**{index}.** {step}")

    with st.expander("♻️ Cách tái sử dụng hoặc kéo dài vòng đời", expanded=True):
        for item in profile.get("reuse_options", []):
            st.markdown(f"- {item}")

    with st.expander("⚠️ Lưu ý an toàn và điều cần tránh"):
        for item in profile.get("safety_notes", []):
            st.markdown(f"- {item}")

    with st.expander("📍 Lưu ý khi tìm điểm thu gom"):
        st.write(profile.get("local_variation", "Kiểm tra quy định tại địa phương và xác nhận cơ sở có tiếp nhận loại vật liệu này."))

    st.caption(
        "Độ tin cậy là điểm số của mô hình, không phải cam kết chắc chắn. "
        "EcoSort hiện phân loại một vật thể chính trong ảnh và hỗ trợ 40 nhãn đã huấn luyện."
    )


def render_assistant_message(message: dict[str, Any], guide: dict[str, Any]):
    if message.get("kind") == "recognition":
        prediction = message["prediction"]
        label = prediction["label"]
        profile = guide.get(label, {})

        if prediction.get("uncertain"):
            st.warning(
                "Mình chưa thể xác định chắc chắn vật thể trong ảnh. "
                "Dưới đây chỉ là dự đoán có khả năng cao nhất, chưa nên dùng làm kết luận phân loại."
            )
            st.markdown(
                f"**Có thể là:** {profile.get('name', label)} — "
                f"{prediction['confidence'] * 100:.1f}%"
            )
            st.markdown("**Bạn thử:** chụp gần hơn, đủ sáng, chỉ để một vật thể chính trong khung hình và tránh che khuất.")
        else:
            st.success(f"Đã nhận diện: **{profile.get('name', label)}**")
            render_profile(profile, label, prediction)

        st.markdown("**Ba dự đoán hàng đầu:**")
        for item in prediction.get("top_items", []):
            candidate = guide.get(item["label"], {})
            st.write(
                f"• {candidate.get('name', item['label'])}: "
                f"{item['confidence'] * 100:.1f}%"
            )
    else:
        st.markdown(message.get("content", ""))


def add_recognition(uploaded_bytes: bytes, filename: str, assets):
    labels, guide, interpreter, input_details, output_details, _ = assets
    try:
        with Image.open(io.BytesIO(uploaded_bytes)) as image:
            image.load()
            image_copy = image.copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("File ảnh không đọc được. Hãy chọn JPG, PNG hoặc WEBP hợp lệ.") from exc

    result = predict_image(
        image_copy, labels, interpreter, input_details, output_details
    )
    result["filename"] = filename
    st.session_state.current_detection = result

    display_name = guide[result["label"]].get("name", result["label"])
    user_text = f"Mình vừa gửi ảnh để EcoSort nhận diện: {filename}."
    st.session_state.chat_messages.append({
        "role": "user",
        "content": user_text,
        "image_bytes": uploaded_bytes,
        "llm_text": user_text,
    })

    summary = (
        f"Kết quả dự đoán gần nhất: {display_name} "
        f"({result['confidence'] * 100:.1f}%). "
        + ("Mô hình chưa đủ chắc chắn." if result["uncertain"] else "Đây là dự đoán có độ tin cậy đạt ngưỡng hiển thị.")
    )
    st.session_state.chat_messages.append({
        "role": "assistant",
        "kind": "recognition",
        "content": summary,
        "prediction": result,
        "llm_text": summary,
    })

    # Count only confident predictions in this browser session.
    if not result["uncertain"]:
        st.session_state.detection_count += 1


def get_api_status() -> bool:
    return bool(get_api_key()) and genai is not None


# -----------------------------------------------------------------------------
# Initialize state and assets
# -----------------------------------------------------------------------------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "current_detection" not in st.session_state:
    st.session_state.current_detection = None
if "detection_count" not in st.session_state:
    st.session_state.detection_count = 0

labels: list[str] = []
guide: dict[str, dict[str, Any]] = {}
knowledge_ready = False
knowledge_error = None
model_ready = False
model_error = None
assets = None

try:
    labels, guide = load_knowledge_assets()
    knowledge_ready = True
except Exception as exc:
    knowledge_error = str(exc)

if knowledge_ready:
    try:
        interpreter, input_details, output_details, active_model_path = load_model_assets(labels)
        assets = (labels, guide, interpreter, input_details, output_details, active_model_path)
        model_ready = True
    except Exception as exc:
        model_error = str(exc)
else:
    model_error = knowledge_error


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# ♻️ EcoSort AI")
    st.caption("Nhận diện vật thể và hướng dẫn xử lý rác")
    st.divider()

    if knowledge_ready:
        st.success(f"Kho hướng dẫn đã sẵn sàng · {len(labels)} nhãn")
    else:
        st.error("Kho hướng dẫn chưa sẵn sàng")
        st.caption(knowledge_error)

    if model_ready:
        st.success("Model nhận diện đã sẵn sàng")
        st.caption(f"{len(labels)} nhãn · {active_model_path.name}")
    else:
        st.error("Model nhận diện chưa sẵn sàng")
        st.caption(model_error)

    if get_api_status():
        st.success("Chat AI: đã cấu hình API")
        st.caption(f"Model hội thoại: {get_gemini_model()}")
    else:
        st.warning("Chat AI: đang ở chế độ dự phòng cục bộ")
        st.caption("Cần GEMINI_API_KEY trong Streamlit Secrets để bật hội thoại LLM.")

    st.metric("Lượt nhận diện đạt ngưỡng trong phiên", st.session_state.detection_count)

    if st.button("Xóa cuộc trò chuyện", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.current_detection = None
        st.session_state.detection_count = 0
        st.rerun()

    st.divider()
    st.caption("EcoSort chỉ hỗ trợ 40 loại vật thể đã huấn luyện. Quy định thu gom có thể khác nhau theo địa phương.")


# -----------------------------------------------------------------------------
# Main interface
# -----------------------------------------------------------------------------
st.title("♻️ EcoSort AI")
st.markdown(
    '<p class="ecosort-subtitle">Chụp ảnh hoặc gửi câu hỏi. EcoSort sẽ nhận diện vật thể, giải thích đó là rác gì và hướng dẫn xử lý từng bước.</p>',
    unsafe_allow_html=True,
)

if not model_ready:
    st.error(
        "Chưa thể nhận diện ảnh vì thiếu model hoặc dữ liệu nhãn. "
        "Hãy thêm model TFLite, ecosort_labels.json và waste_guide.json vào cùng thư mục với app.py."
    )

# Camera capture stays available alongside the chat composer.
with st.expander("📷 Chụp ảnh trực tiếp bằng camera"):
    camera_file = st.camera_input("Chụp một vật thể rác", key="ecosort_camera")
    if camera_file is not None:
        if st.button("Nhận diện ảnh camera", key="recognize_camera"):
            if not model_ready:
                st.error("Model chưa sẵn sàng.")
            else:
                try:
                    add_recognition(
                        camera_file.getvalue(), camera_file.name or "camera.jpg", assets
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Không thể xử lý ảnh camera: {exc}")

# Render existing chat history.
for message in st.session_state.chat_messages:
    role = message.get("role", "assistant")
    with st.chat_message(role):
        if role == "user":
            st.markdown(message.get("content", ""))
            if message.get("image_bytes"):
                st.image(message["image_bytes"], use_container_width=True)
        else:
            render_assistant_message(message, guide)

# Unified chat composer: text and image upload in the same input.
prompt = st.chat_input(
    "Hỏi EcoSort hoặc đính kèm ảnh rác...",
    accept_file="multiple",
    file_type=["jpg", "jpeg", "png", "webp"],
    max_upload_size=12,
    key="ecosort_chat_input",
)

if prompt is not None:
    user_text = (prompt.text or "").strip()
    uploaded_files = list(prompt.files or [])

    if not user_text and not uploaded_files:
        st.stop()

    # Handle up to three images in one submission to avoid accidental heavy uploads.
    if uploaded_files:
        if not model_ready:
            st.error("Model chưa sẵn sàng nên chưa thể nhận diện ảnh.")
        else:
            for uploaded in uploaded_files[:3]:
                data = uploaded.getvalue()
                if len(data) > MAX_IMAGE_BYTES:
                    st.error(f"Ảnh {uploaded.name} vượt quá giới hạn 12 MB.")
                    continue
                try:
                    add_recognition(data, uploaded.name, assets)
                except Exception as exc:
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": f"Mình chưa xử lý được ảnh **{uploaded.name}**: {exc}",
                    })

    if user_text:
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_text,
            "llm_text": user_text,
        })

        with st.spinner("EcoSort đang suy nghĩ..."):
            try:
                answer = ask_gemini(user_text, guide, labels)
            except Exception as exc:
                # Keep the chat usable during missing keys, quota limits, or network failures.
                answer = local_fallback(user_text, guide)
                if get_api_key():
                    answer += (
                        "\n\n*Lưu ý: dịch vụ hội thoại AI hiện không phản hồi. "
                        "Mình đang dùng dữ liệu tra cứu cục bộ để tránh bỏ trống câu trả lời. "
                        f"Mã lỗi kỹ thuật: {type(exc).__name__}.*"
                    )

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": answer,
            "llm_text": answer,
        })

    st.rerun()
