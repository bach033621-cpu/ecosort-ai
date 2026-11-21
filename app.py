from __future__ import annotations

import io
import json
import os
import time
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
GEMINI_MODEL_DEFAULT = "gemini-3.8-flash"
GEMINI_BACKUP_MODEL = "gemini-3.5-flash-lite"
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
