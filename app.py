from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
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
# EcoSort Hybrid: YOLO11n multi-object detector + EcoSort 40-class fallback
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
YOLO_MODEL_PATH = ROOT / "EcoSort_YOLO11_19.onnx"
YOLO_LABELS_PATH = ROOT / "yolo19_labels.json"
CLASSIFIER_MODEL_CANDIDATES = [
    ROOT / "EcoSort_40Class_Optimized.tflite",
    ROOT / "EcoSort_40Class.tflite",
]
CLASSIFIER_LABELS_PATH = ROOT / "ecosort_labels.json"
GUIDE_PATH = ROOT / "waste_guide.json"
YOLO_GUIDE_PATH = ROOT / "yolo19_waste_guide.json"

YOLO_IMAGE_SIZE = 640
YOLO_CONFIDENCE = 0.25
YOLO_IOU = 0.50
YOLO_MAX_DETECTIONS = 40
CLASSIFIER_CONFIDENCE = 0.65
CLASSIFIER_MARGIN = 0.08
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_CHAT_HISTORY = 16
GEMINI_MODEL_DEFAULT = "gemini-3.8-flash"
GEMINI_BACKUP_MODEL = "gemini-3.5-flash-lite"

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
      .block-container {max-width: 1180px; padding-top: 1.35rem; padding-bottom: 6rem;}
      [data-testid="stChatMessage"] {border-radius: 16px; padding: 0.7rem 0.9rem;}
      .ecosort-subtitle {color: #718096; margin-top: -0.45rem;}
      .ecosort-chip {display:inline-block;border-radius:999px;padding:.28rem .65rem;font-size:.82rem;font-weight:650;}
      .ecosort-muted {color:#718096;font-size:.88rem;}
      div[data-testid="stMetric"] {background:rgba(128,128,128,.06);border:1px solid rgba(128,128,128,.15);padding:.65rem;border-radius:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Load metadata and runtimes
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_json(path_text: str, modified: float):
    del modified
    with open(path_text, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Đang tải detector YOLO11n 19 lớp...")
def load_yolo_onnx(path_text: str, modified: float, size: int):
    del modified, size
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(path_text, providers=providers)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or not outputs:
        raise RuntimeError("ONNX detector có cấu trúc input/output không được hỗ trợ.")
    return session, inputs[0].name, outputs[0].name, inputs[0].shape


@st.cache_resource(show_spinner="Đang tải model EcoSort 40 lớp...")
def load_classifier(path_text: str, modified: float, size: int):
    del modified, size
    if Interpreter is None:
        raise RuntimeError("Thiếu ai-edge-litert. Kiểm tra requirements.txt.")
    interpreter = Interpreter(model_path=path_text, num_threads=2)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details()[0], interpreter.get_output_details()[0]


def load_assets():
    required = [CLASSIFIER_LABELS_PATH, GUIDE_PATH, YOLO_LABELS_PATH, YOLO_GUIDE_PATH]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Thiếu file cấu hình: " + ", ".join(missing))

    classifier_labels = read_json(str(CLASSIFIER_LABELS_PATH), CLASSIFIER_LABELS_PATH.stat().st_mtime)
    guide = read_json(str(GUIDE_PATH), GUIDE_PATH.stat().st_mtime)
    yolo_labels = read_json(str(YOLO_LABELS_PATH), YOLO_LABELS_PATH.stat().st_mtime)
    yolo_guide = read_json(str(YOLO_GUIDE_PATH), YOLO_GUIDE_PATH.stat().st_mtime)

    if not isinstance(classifier_labels, list) or len(classifier_labels) != 40:
        raise ValueError("ecosort_labels.json phải chứa đúng 40 nhãn theo model classifier.")
    if not isinstance(yolo_labels, list) or len(yolo_labels) != 19:
        raise ValueError("yolo19_labels.json phải chứa đúng 19 nhãn theo checkpoint YOLO11n.")
    missing_classifier_guides = sorted(set(classifier_labels) - set(guide))
    missing_yolo_guides = sorted(set(yolo_labels) - set(yolo_guide))
    if missing_classifier_guides:
        raise ValueError("Thiếu hướng dẫn classifier: " + ", ".join(missing_classifier_guides))
    if missing_yolo_guides:
        raise ValueError("Thiếu hướng dẫn detector: " + ", ".join(missing_yolo_guides))

    yolo_assets = None
    yolo_error = None
    if YOLO_MODEL_PATH.exists():
        try:
            yolo_assets = load_yolo_onnx(
                str(YOLO_MODEL_PATH), YOLO_MODEL_PATH.stat().st_mtime,
                YOLO_MODEL_PATH.stat().st_size,
            )
        except Exception as exc:
            yolo_error = f"Không tải được YOLO11 ONNX: {type(exc).__name__}: {exc}"
    else:
        yolo_error = "Chưa có EcoSort_YOLO11_19.onnx. Hãy chạy export_yolo11_colab.py và upload model ONNX."

    classifier_assets = None
    classifier_error = None
    model_path = next((p for p in CLASSIFIER_MODEL_CANDIDATES if p.exists()), None)
    if model_path is not None:
        try:
            interpreter, input_details, output_details = load_classifier(
                str(model_path), model_path.stat().st_mtime, model_path.stat().st_size
            )
            output_shape = output_details.get("shape", [])
            if len(output_shape) < 2 or int(output_shape[-1]) != len(classifier_labels):
                raise ValueError(
                    f"Classifier output {output_shape} không khớp {len(classifier_labels)} nhãn."
                )
            classifier_assets = (interpreter, input_details, output_details, model_path)
        except Exception as exc:
            classifier_error = f"Không tải được classifier: {type(exc).__name__}: {exc}"
    else:
        classifier_error = "Thiếu EcoSort_40Class_Optimized.tflite."

    all_guides = dict(guide)
    all_guides.update(yolo_guide)

    return {
        "classifier_labels": classifier_labels,
        "guide": guide,
        "all_guides": all_guides,
        "yolo_labels": yolo_labels,
        "yolo_guide": yolo_guide,
        "yolo": yolo_assets,
        "yolo_error": yolo_error,
        "classifier": classifier_assets,
        "classifier_error": classifier_error,
    }


# -----------------------------------------------------------------------------
# ONNX YOLO11 pre-processing and post-processing
# -----------------------------------------------------------------------------
def letterbox_rgb(image: Image.Image, target: int = YOLO_IMAGE_SIZE):
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = min(target / width, target / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    pad_x = (target - new_w) / 2
    pad_y = (target - new_h) / 2
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas = Image.new("RGB", (target, target), (114, 114, 114))
    canvas.paste(resized, (left, top))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
    return tensor, scale, left, top, image.size


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter + 1e-7
    return inter / union


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, limit: int):
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    kept = []
    while order.size and len(kept) < limit:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        overlaps = _iou_one_to_many(boxes[current], boxes[order[1:]])
        order = order[1:][overlaps <= iou_threshold]
    return kept


def decode_yolo_output(output: np.ndarray, class_count: int):
    """Accepts NMS output [1,N,6] or raw YOLO output [1,4+C,N]/[1,N,4+C]."""
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"YOLO output không có 2 chiều sau batch: {arr.shape}")

    # NMS-enabled ONNX export: x1,y1,x2,y2,confidence,class_id.
    if arr.shape[-1] == 6:
        rows = arr
        rows = rows[np.isfinite(rows).all(axis=1)]
        rows = rows[rows[:, 4] >= YOLO_CONFIDENCE]
        detections = []
        for row in rows:
            class_id = int(round(float(row[5])))
            if 0 <= class_id < class_count:
                detections.append({
                    "box": row[:4].astype(np.float32),
                    "confidence": float(row[4]),
                    "class_id": class_id,
                })
        return detections

    # Raw YOLO11 export: one channel dimension contains 4 box values + class scores.
    if arr.shape[0] == 4 + class_count:
        rows = arr.T
    elif arr.shape[1] == 4 + class_count:
        rows = arr
    else:
        raise ValueError(
            f"Không nhận ra YOLO output {arr.shape}. Hãy export ONNX bằng nms=True."
        )

    xywh = rows[:, :4]
    scores_all = rows[:, 4:]
    class_ids = np.argmax(scores_all, axis=1)
    scores = scores_all[np.arange(len(rows)), class_ids]
    keep = np.isfinite(rows).all(axis=1) & (scores >= YOLO_CONFIDENCE)
    xywh, scores, class_ids = xywh[keep], scores[keep], class_ids[keep]
    if len(scores) == 0:
        return []

    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    result = []
    for class_id in np.unique(class_ids):
        idx = np.where(class_ids == class_id)[0]
        selected = _nms(boxes[idx], scores[idx], YOLO_IOU, YOLO_MAX_DETECTIONS)
        for local_idx in selected:
            j = int(idx[local_idx])
            result.append({
                "box": boxes[j].astype(np.float32),
                "confidence": float(scores[j]),
                "class_id": int(class_ids[j]),
            })
    result.sort(key=lambda item: item["confidence"], reverse=True)
    return result[:YOLO_MAX_DETECTIONS]


def run_yolo_detector(image: Image.Image, yolo_assets, labels: list[str]):
    session, input_name, output_name, input_shape = yolo_assets
    tensor, scale, pad_left, pad_top, original_size = letterbox_rgb(image, YOLO_IMAGE_SIZE)
    output = session.run([output_name], {input_name: tensor})[0]
    raw = decode_yolo_output(output, len(labels))

    original_w, original_h = original_size
    detections = []
    for item in raw:
        box = np.asarray(item["box"], dtype=np.float32).copy()
        # Some NMS exports return normalized coordinates; normalize to input pixels.
        if float(np.max(np.abs(box))) <= 1.5:
            box *= YOLO_IMAGE_SIZE
        box[[0, 2]] = (box[[0, 2]] - pad_left) / scale
        box[[1, 3]] = (box[[1, 3]] - pad_top) / scale
        box[[0, 2]] = np.clip(box[[0, 2]], 0, original_w)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, original_h)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        class_id = int(item["class_id"])
        detections.append({
            "label": labels[class_id],
            "class_id": class_id,
            "confidence": float(item["confidence"]),
            "box": [float(x) for x in box],
        })
    return detections


# -----------------------------------------------------------------------------
# EcoSort 40-class classifier fallback
# -----------------------------------------------------------------------------
def prepare_classifier_input(image: Image.Image, input_details: dict):
    shape = input_details["shape"]
    height, width = int(shape[1]), int(shape[2])
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32)
    arr = (arr / 127.5) - 1.0  # Exact MobileNetV2 preprocessing used in Colab.
    arr = arr[None, ...]
    dtype = input_details["dtype"]
    if dtype in (np.uint8, np.int8):
        scale, zero_point = input_details.get("quantization", (0.0, 0))
        if not scale:
            raise ValueError("Classifier quantization thiếu scale.")
        limits = np.iinfo(dtype)
        arr = np.clip(np.round(arr / scale + zero_point), limits.min, limits.max).astype(dtype)
    else:
        arr = arr.astype(dtype, copy=False)
    return arr


def run_classifier(image: Image.Image, classifier_assets, labels: list[str]):
    interpreter, input_details, output_details, _ = classifier_assets
    tensor = prepare_classifier_input(image, input_details)
    interpreter.set_tensor(input_details["index"], tensor)
    interpreter.invoke()
    probs = np.asarray(interpreter.get_tensor(output_details["index"])[0], dtype=np.float32)
    if output_details["dtype"] in (np.uint8, np.int8):
        scale, zero_point = output_details.get("quantization", (0.0, 0))
        probs = (probs.astype(np.float32) - zero_point) * scale
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.clip(probs, 0, None)
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("Classifier không trả về xác suất hợp lệ.")
    probs = probs / total
    order = np.argsort(probs)[::-1]
    first = int(order[0])
    second = int(order[1]) if len(order) > 1 else first
    confidence = float(probs[first])
    margin = confidence - float(probs[second])
    return {
        "label": labels[first],
        "confidence": confidence,
        "margin": margin,
        "uncertain": confidence < CLASSIFIER_CONFIDENCE or margin < CLASSIFIER_MARGIN,
        "top_items": [
            {"label": labels[int(i)], "confidence": float(probs[int(i)])}
            for i in order[:3]
        ],
    }


# -----------------------------------------------------------------------------
# Visual and knowledge rendering
# -----------------------------------------------------------------------------
def annotated_image(image: Image.Image, detections: list[dict], yolo_guide: dict):
    canvas = ImageOps.exif_transpose(image).convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for item in detections:
        x1, y1, x2, y2 = [int(round(x)) for x in item["box"]]
        color = "#18A66A"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(3, canvas.width // 450))
        profile = yolo_guide.get(item["label"], {})
        text = f"{profile.get('name', item['label'])} {item['confidence'] * 100:.0f}%"
        bbox = draw.textbbox((x1, max(0, y1 - 24)), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((bbox[0] + 3, bbox[1] + 2), text, fill="white", font=font)
    return canvas


def render_profile(profile: dict, confidence: float, source: str):
    group_name = profile.get("group_name", "Cần kiểm tra")
    color = GROUP_COLORS.get(group_name, "#737780")
    st.markdown(
        f'<span class="ecosort-chip" style="background:{color}18;color:{color};">{group_name}</span>',
        unsafe_allow_html=True,
    )
    st.metric("Độ tin cậy của dự đoán", f"{confidence * 100:.1f}%")
    st.markdown("**Đây là rác gì?**")
    st.write(profile.get("what_it_is", "Chưa có mô tả chi tiết."))
    if profile.get("common_materials"):
        st.markdown("**Vật liệu / thành phần thường gặp:**")
        st.write(profile["common_materials"])
    st.markdown("**Vì sao được xếp vào nhóm này?**")
    st.write(profile.get("why_group", "Cần kiểm tra vật liệu và quy định của nơi thu gom."))
    st.markdown("**Cách xử lý từng bước**")
    steps = profile.get("preparation_steps", []) + profile.get("disposal_steps", [])
    for i, step in enumerate(steps, 1):
        st.markdown(f"**{i}.** {step}")
    with st.expander("♻️ Tái sử dụng / kéo dài vòng đời", expanded=True):
        for item in profile.get("reuse_options", []):
            st.markdown(f"- {item}")
    with st.expander("⚠️ Lưu ý an toàn"):
        for item in profile.get("safety_notes", []):
            st.markdown(f"- {item}")
    st.caption(profile.get("local_variation", "Quy định tiếp nhận có thể khác nhau giữa các địa phương."))
    st.caption(f"Nguồn nhận diện: {source}. Điểm số là độ tin cậy mô hình, không phải cam kết chắc chắn.")


def local_fallback(question: str, guide: dict, current_context: list[dict]):
    q = question.casefold()
    markers = ["này", "nó", "cái đó", "vừa quét", "vật này", "xử lý", "phân loại", "tái chế", "vứt", "bỏ đi", "tại sao", "vì sao", "là gì", "thu gom"]
    if current_context and any(x in q for x in markers):
        names = [x.get("display_name", x.get("label", "vật thể")) for x in current_context[:5]]
        return (
            "Mình đang dựa trên ảnh bạn vừa gửi. Những vật thể nhận diện gần nhất là: "
            + ", ".join(names)
            + ". Bạn có thể hỏi cụ thể về một vật thể. Hiện Gemini chưa phản hồi nên mình chỉ có thể tra cứu các hồ sơ đã lưu, chưa thể trả lời mở rộng như ChatGPT."
        )
    for label, profile in guide.items():
        terms = [label.replace("_", " ").casefold(), str(profile.get("name", "")).casefold()]
        terms.extend(str(x).casefold() for x in profile.get("keywords", []))
        if any(term and term in q for term in terms):
            return profile.get("what_it_is", "") + "\n\n" + profile.get("disposal_advice", "")
    return "Mình vẫn nhận được câu hỏi, nhưng Gemini chưa kết nối thành công. Bạn có thể hỏi về tên một vật thể cụ thể đã có trong kho hướng dẫn EcoSort."


# -----------------------------------------------------------------------------
# Gemini conversation
# -----------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """
Bạn là EcoSort AI, trợ lý phân loại và xử lý rác bằng tiếng Việt.
Trả lời tự nhiên như một cuộc trò chuyện nhiều lượt, nhớ vật thể vừa được nhận diện và hiểu các câu hỏi tiếp nối.
Khi giải thích, trình bày chi tiết: vật thể là gì, vật liệu/thành phần thường gặp, vì sao thuộc nhóm, chuẩn bị trước khi bỏ, từng bước xử lý, tái sử dụng, lưu ý an toàn và quy định địa phương.
Dùng kho hướng dẫn EcoSort làm nguồn ưu tiên. Không bịa điểm thu gom, quy định địa phương, thành phần hóa học hoặc khả năng tái chế không có căn cứ.
Nếu model chỉ nhận diện được vật thể chung như chai hoặc hộp đựng, không tự kết luận đó là nhựa; hãy yêu cầu kiểm tra vật liệu/ký hiệu.
Nếu ảnh có nhiều vật thể, chỉ nói về các vật thể trong danh sách detector trả về. Không khẳng định đã nhìn thấy vật thể mà detector không phát hiện.
Không được khẳng định kết quả AI chắc chắn 100%. Khi không đủ thông tin, nói rõ giới hạn và hướng dẫn kiểm tra an toàn.
""".strip()


def get_api_key():
    try:
        value = st.secrets.get("GEMINI_API_KEY", "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "").strip() or None


def get_gemini_model():
    try:
        return str(st.secrets.get("GEMINI_MODEL", GEMINI_MODEL_DEFAULT)).strip()
    except Exception:
        return GEMINI_MODEL_DEFAULT


def ask_gemini(question: str, current_context: list[dict], guide: dict):
    if genai is None or genai_types is None:
        raise RuntimeError("Chưa cài google-genai.")
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("Chưa cấu hình GEMINI_API_KEY trong Streamlit Secrets.")

    context_lines = []
    for item in current_context[:8]:
        context_lines.append({
            "object": item.get("display_name"),
            "confidence": round(float(item.get("confidence", 0)) * 100, 1),
            "guide": item.get("profile", {}),
        })
    context_text = "Ngữ cảnh ảnh mới nhất: " + json.dumps(context_lines, ensure_ascii=False)

    history = st.session_state.chat_messages[-(MAX_CHAT_HISTORY * 2):]
    contents = []
    for msg in history:
        if msg.get("role") not in {"user", "assistant"}:
            continue
        text = msg.get("llm_text", msg.get("content", ""))
        if not text:
            continue
        contents.append(genai_types.Content(
            role="user" if msg["role"] == "user" else "model",
            parts=[genai_types.Part.from_text(text=str(text))],
        ))
    if not contents or contents[-1].role != "user":
        contents.append(genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=question)]))
    contents[-1].parts.append(genai_types.Part.from_text(text=context_text))

    client = genai.Client(api_key=api_key)
    model_names = [get_gemini_model(), GEMINI_BACKUP_MODEL]
    model_names = list(dict.fromkeys([x for x in model_names if x]))
    errors = []
    for model_name in model_names:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.35,
                    max_output_tokens=1800,
                ),
            )
            answer = (response.text or "").strip()
            if not answer:
                raise RuntimeError("Model không trả về văn bản.")
            st.session_state.gemini_status = f"Hoạt động · {model_name}"
            st.session_state.gemini_error = ""
            return answer
        except Exception as exc:
            message = str(exc).replace(api_key, "[API KEY HIDDEN]")
            errors.append(f"{model_name}: {type(exc).__name__}: {message[:300]}")
    st.session_state.gemini_status = "Không kết nối được"
    st.session_state.gemini_error = " | ".join(errors)
    raise RuntimeError(st.session_state.gemini_error)


# -----------------------------------------------------------------------------
# Streamlit state and page
# -----------------------------------------------------------------------------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "latest_context" not in st.session_state:
    st.session_state.latest_context = []
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0
if "gemini_status" not in st.session_state:
    st.session_state.gemini_status = "Chưa gọi thử"
if "gemini_error" not in st.session_state:
    st.session_state.gemini_error = ""


def get_image_bytes(uploaded):
    data = uploaded.getvalue()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Ảnh vượt quá giới hạn 12 MB.")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Không đọc được ảnh. Hãy chọn JPG, PNG hoặc WEBP.") from exc
    return image, data


def scan_image(image: Image.Image, assets):
    results = []
    detector_ready = assets["yolo"] is not None
    classifier_ready = assets["classifier"] is not None

    if detector_ready:
        try:
            detections = run_yolo_detector(image, assets["yolo"], assets["yolo_labels"])
        except Exception as exc:
            st.warning(f"Detector chưa xử lý được ảnh ({type(exc).__name__}). Thử classifier dự phòng.")
            detections = []
    else:
        detections = []

    if detections:
        for item in detections:
            profile = assets["yolo_guide"].get(item["label"], {})
            results.append({
                **item,
                "display_name": profile.get("name", item["label"]),
                "profile": profile,
                "source": "YOLO11n 19 lớp",
                "uncertain": item["confidence"] < 0.35,
            })
        return results, "detector"

    # The 40-class classifier is the fallback for single items outside the 19 detector classes.
    if classifier_ready:
        prediction = run_classifier(image, assets["classifier"], assets["classifier_labels"])
        profile = assets["guide"].get(prediction["label"], {})
        results.append({
            **prediction,
            "display_name": profile.get("name", prediction["label"]),
            "profile": profile,
            "source": "EcoSort MobileNetV2 40 lớp (fallback)",
            "uncertain": prediction["uncertain"],
        })
        return results, "classifier"

    raise RuntimeError("Không có detector ONNX hoặc classifier TFLite khả dụng.")


def render_scan_message(message: dict):
    results = message.get("results", [])
    if not results:
        st.warning("Không tìm thấy vật thể rác đủ tin cậy trong ảnh. Hãy thử ảnh khác, rõ hơn và ít bị che khuất.")
        return

    if message.get("annotated_bytes"):
        st.image(message["annotated_bytes"], caption="Ảnh đã đánh dấu vật thể nhận diện", use_container_width=True)

    uncertain_count = sum(bool(x.get("uncertain")) for x in results)
    if uncertain_count:
        st.warning("Một số dự đoán có độ tin cậy thấp. Hãy kiểm tra lại trước khi phân loại thực tế.")
    else:
        st.success(f"Đã phát hiện {len(results)} vật thể.")

    for idx, item in enumerate(results, 1):
        label = item.get("display_name", item.get("label", "Vật thể"))
        conf = float(item.get("confidence", 0))
        title = f"{idx}. {label} · {conf * 100:.1f}%"
        profile = item.get("profile", {})
        with st.expander(title, expanded=(idx == 1)):
            if item.get("uncertain"):
                st.warning("Dự đoán này chưa đủ chắc chắn để dùng làm kết luận phân loại.")
            render_profile(profile, conf, item.get("source", "AI"))


def process_uploaded_image(uploaded, assets):
    image, data = get_image_bytes(uploaded)
    results, mode = scan_image(image, assets)
    if mode == "detector" and results:
        drawn = annotated_image(image, results, assets["yolo_guide"])
        buf = io.BytesIO()
        drawn.save(buf, format="JPEG", quality=90)
        annotated_bytes = buf.getvalue()
    else:
        annotated_bytes = None

    # Keep compact text context for chat; do not send image bytes to Gemini.
    st.session_state.latest_context = results
    summary = "; ".join(
        f"{x['display_name']} ({x['confidence'] * 100:.1f}%)"
        for x in results[:8]
    ) or "Không phát hiện vật thể đủ tin cậy"
    st.session_state.chat_messages.append({
        "role": "user",
        "content": f"Mình vừa gửi ảnh: {uploaded.name}.",
        "llm_text": f"Người dùng vừa tải ảnh {uploaded.name} lên EcoSort.",
    })
    st.session_state.chat_messages.append({
        "role": "assistant",
        "kind": "scan",
        "results": results,
        "annotated_bytes": annotated_bytes,
        "content": f"Kết quả nhận diện: {summary}",
        "llm_text": f"Kết quả nhận diện gần nhất: {summary}. Hãy dùng danh sách này làm ngữ cảnh nếu người dùng hỏi tiếp.",
    })
    if results and not all(x.get("uncertain") for x in results):
        st.session_state.scan_count += 1


def render_history(assets):
    for message in st.session_state.chat_messages:
        role = message.get("role", "assistant")
        with st.chat_message(role):
            if message.get("kind") == "scan":
                render_scan_message(message)
            else:
                st.markdown(message.get("content", ""))


try:
    ASSETS = load_assets()
    ASSET_ERROR = None
except Exception as exc:
    ASSETS = None
    ASSET_ERROR = f"{type(exc).__name__}: {exc}"

with st.sidebar:
    st.markdown("# ♻️ EcoSort AI")
    st.caption("YOLO11n multi-object + EcoSort 40-class fallback")
    st.divider()

    if ASSETS and ASSETS["yolo"] is not None:
        st.success("YOLO11n detector: sẵn sàng · 19 lớp")
    else:
        st.warning("YOLO11n detector chưa sẵn sàng")
        st.caption(ASSETS["yolo_error"] if ASSETS else ASSET_ERROR)

    if ASSETS and ASSETS["classifier"] is not None:
        st.success("EcoSort classifier: sẵn sàng · 40 lớp")
    else:
        st.warning("Classifier dự phòng chưa sẵn sàng")
        st.caption(ASSETS["classifier_error"] if ASSETS else ASSET_ERROR)

    if get_api_key() and genai is not None:
        st.info("Gemini API key đã được tìm thấy")
        st.caption(f"Model: {get_gemini_model()}")
        st.caption(f"Trạng thái gọi gần nhất: {st.session_state.gemini_status}")
        if st.session_state.gemini_error:
            with st.expander("Chi tiết lỗi Gemini"):
                st.code(st.session_state.gemini_error)
    else:
        st.warning("Chatbox đang ở chế độ tra cứu cục bộ")
        st.caption("Thêm GEMINI_API_KEY trong Streamlit Secrets để bật hội thoại mở rộng.")

    st.metric("Lượt quét đạt ngưỡng trong phiên", st.session_state.scan_count)
    if st.button("Xóa cuộc trò chuyện", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.latest_context = []
        st.session_state.scan_count = 0
        st.rerun()
    st.divider()
    st.caption("Detector YOLO11n 19 lớp xử lý nhiều vật thể trong một ảnh. Nếu không phát hiện, EcoSort dùng classifier MobileNetV2 40 lớp làm phương án dự phòng cho ảnh một vật thể.")
    st.caption("Lưu ý: nhãn bottle_awss và foodContainer_awss không xác định vật liệu; bottle_awss không có mẫu validation trong bộ đánh giá nguồn.")

st.title("♻️ EcoSort AI")
st.markdown(
    '<p class="ecosort-subtitle">Chụp một ảnh có nhiều vật thể hoặc hỏi EcoSort về cách phân loại và xử lý rác.</p>',
    unsafe_allow_html=True,
)

if ASSETS is None:
    st.error("Không tải được các file nhãn / hướng dẫn. Kiểm tra JSON trong repository.")
elif ASSETS["yolo"] is None and ASSETS["classifier"] is None:
    st.error("Chưa có model nào khả dụng. Cần thêm ONNX detector hoặc TFLite classifier.")

with st.expander("📷 Chụp ảnh trực tiếp bằng camera"):
    camera_file = st.camera_input("Chụp rác cần nhận diện", key="ecosort_camera")
    if camera_file is not None and st.button("Quét ảnh camera", key="scan_camera"):
        if ASSETS is None:
            st.error("Thiếu file model hoặc dữ liệu hướng dẫn.")
        else:
            try:
                process_uploaded_image(camera_file, ASSETS)
                st.rerun()
            except Exception as exc:
                st.error(f"Không xử lý được ảnh: {type(exc).__name__}: {exc}")

render_history(ASSETS or {"guide": {}})

prompt = st.chat_input(
    "Hỏi EcoSort hoặc đính kèm ảnh rác...",
    accept_file="multiple",
    file_type=["jpg", "jpeg", "png", "webp"],
    max_upload_size=12,
    key="ecosort_chat_input",
)

if prompt is not None:
    user_text = (prompt.text or "").strip()
    files = list(prompt.files or [])
    if not user_text and not files:
        st.stop()

    if files:
        if ASSETS is None:
            st.error("Kho dữ liệu chưa sẵn sàng.")
        else:
            for uploaded in files[:3]:
                try:
                    process_uploaded_image(uploaded, ASSETS)
                except Exception as exc:
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": f"Mình chưa xử lý được ảnh **{uploaded.name}**: {type(exc).__name__}: {exc}",
                        "llm_text": f"Không xử lý được ảnh {uploaded.name}.",
                    })

    if user_text:
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_text,
            "llm_text": user_text,
        })
        with st.spinner("EcoSort đang suy nghĩ..."):
            try:
                answer = ask_gemini(user_text, st.session_state.latest_context, ASSETS["all_guides"] if ASSETS else {})
            except Exception as exc:
                answer = local_fallback(user_text, ASSETS["all_guides"] if ASSETS else {}, st.session_state.latest_context)
                if get_api_key():
                    answer += (
                        "\n\n*Gemini hiện chưa trả lời được sau khi thử model chính và model dự phòng. "
                        "Quản trị viên có thể mở Chi tiết lỗi Gemini ở sidebar hoặc Streamlit Logs. "
                        f"Loại lỗi: {type(exc).__name__}.*"
                    )
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": answer,
            "llm_text": answer,
        })
    st.rerun()
