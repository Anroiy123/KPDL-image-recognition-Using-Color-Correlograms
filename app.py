"""
app.py - Ứng dụng demo Streamlit cho nhận dạng ảnh bằng Color Correlogram

Chạy: streamlit run app.py
"""

import json
import os
import re
import sys
from html import escape
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from color_correlogram import extract_correlogram_feature
from dataset_profile import DEFAULT_DATASET_PROFILE
from dataset_split import load_split_metadata, resolve_split_indices
from experiment_runner import EVAL_LABELS, run_experiment


PROJECT_DIR = Path(__file__).parent
FEATURES_DIR = PROJECT_DIR / "data" / "features"
MODELS_DIR = PROJECT_DIR / "models"

MODEL_CONFIGS = {
    "hsv": {
        "label": "HSV Spatial",
        "model_file": "corel-1k_svm_correlogram_hsv_spatial.pkl",
        "features_file": "corel-1k_correlogram_hsv_spatial.npy",
        "sidebar_message": "Mô hình HSV dùng spatial correlogram: vector toàn ảnh + 4 ô 2x2.",
        "h_bins": 8,
        "s_bins": 3,
        "v_bins": 3,
        "rgb_bins": 4,
        "spatial_grid": 2,
    },
    "rgb": {
        "label": "RGB",
        "model_file": "corel-1k_svm_correlogram_rgb.pkl",
        "features_file": "correlogram_rgb.npy",
        "sidebar_message": "Mô hình RGB được huấn luyện cố định với RGB bins = 4.",
        "h_bins": 8,
        "s_bins": 3,
        "v_bins": 3,
        "rgb_bins": 4,
        "spatial_grid": None,
    },
}

EVAL_METHOD_OPTIONS = [
    "independent_test",
    "holdout",
    "stratified_holdout",
    "repeated_holdout",
    "kfold",
    "leave_one_out",
    "bootstrap",
]

EVAL_METHOD_LABELS_VI = {
    "independent_test": "Independent Test (Chuẩn từ File Split)",
    "holdout": "Hold-out (Random Train/Test)",
    "stratified_holdout": "Stratified Hold-out",
    "repeated_holdout": "Repeated Hold-out",
    "kfold": "k-Fold Cross-Validation",
    "leave_one_out": "Leave-One-Out",
    "bootstrap": "Bootstrap Sampling",
}

EVAL_METHOD_HINTS = {
    "independent_test": "Benchmark: Tạo mới mô hình và Train khớp theo đúng file corel-1k_split.json. Sau đó đánh giá bằng tập test độc lập.",
    "holdout": "Chia ngẫu nhiên train/test mới. Rất nhanh, nhưng độ ổn định phụ thuộc vào đợt chia.",
    "stratified_holdout": "Sẽ tự động chia lại train/test và train mô hình mới ngay lúc này. Giữ tỉ lệ lớp ổn định.",
    "repeated_holdout": "Lặp lại train từ đầu nhiều lần để lấy trung bình hiệu suất, giảm phụ thuộc may rủi.",
    "kfold": "Cross-validation tiêu chuẩn. Chia K phần, train K mô hình, đánh giá ổn định trên toàn phổ.",
    "leave_one_out": "Train N lần, test N lần. (Rất chậm, chỉ nên chạy khi thật sự rảnh).",
    "bootstrap": "Khởi tạo và train lại nhiều lần dựa trên tập mẫu lấy có hoàn lại từ dataset.",
}


THEME_PALETTES = {
    "light": {
        "bg_start": "#f8f2e9",
        "bg_end": "#efe6d7",
        "glow_left": "rgba(233,174,118,0.34)",
        "glow_right": "rgba(106,154,120,0.20)",
        "card": "rgba(255,248,238,0.92)",
        "hero_start": "rgba(255,247,238,0.96)",
        "hero_end": "rgba(251,235,214,0.92)",
        "hero_panel": "rgba(255,255,255,0.76)",
        "artifact_bg": "rgba(255,255,255,0.70)",
        "summary_bg": "rgba(255,252,248,0.72)",
        "summary_bg_emphasis": "rgba(255,244,231,0.94)",
        "expander_bg": "rgba(255,250,243,0.55)",
        "prediction_start": "rgba(255,245,232,0.96)",
        "prediction_end": "rgba(248,230,207,0.92)",
        "text": "#24160d",
        "muted": "#6b5645",
        "line": "rgba(105,76,42,0.16)",
        "accent": "#b44f28",
        "accent_hover": "#a24421",
        "accent_secondary": "#cf6f37",
        "accent_deep": "#7d3118",
        "field": "rgba(250,242,231,0.96)",
        "field_border": "rgba(117,84,54,0.20)",
        "info": "#eef5fb",
        "info_text": "#305574",
        "shadow": "0 18px 48px rgba(70,44,24,0.08)",
        "button_shadow": "0 14px 28px rgba(180,79,40,0.22)",
        "summary_emphasis_shadow": "0 14px 34px rgba(180,79,40,0.10)",
        "chart_bg": "#fffaf5",
        "chart_bar": "#c35c2c",
        "chart_edge": "#8f3a1c",
        "chart_grid": "rgba(36,22,13,0.18)",
    },
    "dark": {
        "bg_start": "#15110f",
        "bg_end": "#1c1714",
        "glow_left": "rgba(180,106,48,0.18)",
        "glow_right": "rgba(76,125,111,0.18)",
        "card": "rgba(33,29,26,0.92)",
        "hero_start": "rgba(38,33,29,0.96)",
        "hero_end": "rgba(26,22,20,0.92)",
        "hero_panel": "rgba(47,41,37,0.82)",
        "artifact_bg": "rgba(32,28,25,0.88)",
        "summary_bg": "rgba(36,31,27,0.82)",
        "summary_bg_emphasis": "rgba(56,39,31,0.94)",
        "expander_bg": "rgba(36,31,28,0.70)",
        "prediction_start": "rgba(61,42,32,0.95)",
        "prediction_end": "rgba(45,34,29,0.92)",
        "text": "#f4ebdf",
        "muted": "#cab9a8",
        "line": "rgba(237,220,197,0.14)",
        "accent": "#d58452",
        "accent_hover": "#c66f3c",
        "accent_secondary": "#e19a6f",
        "accent_deep": "#f1c19c",
        "field": "rgba(39,34,31,0.96)",
        "field_border": "rgba(237,220,197,0.16)",
        "info": "#1d2c37",
        "info_text": "#c5e1f0",
        "shadow": "0 20px 54px rgba(0,0,0,0.30)",
        "button_shadow": "0 14px 28px rgba(0,0,0,0.32)",
        "summary_emphasis_shadow": "0 16px 34px rgba(0,0,0,0.28)",
        "chart_bg": "#211b18",
        "chart_bar": "#d58452",
        "chart_edge": "#f0bc93",
        "chart_grid": "rgba(244,235,223,0.12)",
    },
}


def get_theme_type():
    """Lấy chế độ giao diện hiện tại của Streamlit."""
    try:
        theme_type = st.context.theme.type
    except Exception:
        theme_type = "light"
    return theme_type if theme_type in THEME_PALETTES else "light"


def get_theme_palette(theme_type=None):
    """Trả về bảng màu theo giao diện hiện tại."""
    resolved_theme = theme_type or get_theme_type()
    return THEME_PALETTES.get(resolved_theme, THEME_PALETTES["light"])


def sync_theme_state():
    """Đồng bộ theme giữa các lần rerun để biểu đồ Matplotlib bám đúng light/dark."""
    current_theme = get_theme_type()
    previous_theme = st.session_state.get("_last_theme_type")
    st.session_state["_last_theme_type"] = current_theme
    if previous_theme is not None and previous_theme != current_theme:
        st.rerun()


def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
        :root {
            color-scheme: light dark;
            --accent: #c8743e;
            --accent-hover: #b56233;
            --accent-secondary: #dd9262;
            --accent-deep: color-mix(in srgb, var(--accent) 68%, CanvasText 32%);
            --text: CanvasText;
            --muted: color-mix(in srgb, CanvasText 72%, Canvas 28%);
            --line: color-mix(in srgb, CanvasText 14%, Canvas 86%);
            --card: color-mix(in srgb, Canvas 92%, var(--accent) 8%);
            --hero-start: color-mix(in srgb, Canvas 94%, var(--accent) 6%);
            --hero-end: color-mix(in srgb, Canvas 97%, #8c9a82 3%);
            --hero-panel: color-mix(in srgb, Canvas 90%, var(--accent) 10%);
            --prediction-start: color-mix(in srgb, Canvas 88%, var(--accent) 12%);
            --prediction-end: color-mix(in srgb, Canvas 92%, var(--accent-secondary) 8%);
            --artifact-bg: color-mix(in srgb, Canvas 90%, var(--accent) 10%);
            --summary-bg: color-mix(in srgb, Canvas 93%, var(--accent) 7%);
            --summary-bg-emphasis: color-mix(in srgb, Canvas 86%, var(--accent) 14%);
            --expander-bg: color-mix(in srgb, Canvas 95%, var(--accent) 5%);
            --field: color-mix(in srgb, Field 88%, var(--accent) 12%);
            --field-border: color-mix(in srgb, CanvasText 18%, Canvas 82%);
            --info: color-mix(in srgb, Canvas 82%, #6f9cbd 18%);
            --info-text: color-mix(in srgb, CanvasText 72%, #6f9cbd 28%);
            --glow-left: color-mix(in srgb, var(--accent) 22%, transparent);
            --glow-right: color-mix(in srgb, #6d8a73 18%, transparent);
            --bg-start: color-mix(in srgb, Canvas 96%, var(--accent) 4%);
            --bg-end: color-mix(in srgb, Canvas 92%, var(--accent) 8%);
            --shadow: 0 18px 48px color-mix(in srgb, CanvasText 14%, transparent);
            --button-shadow: 0 14px 28px color-mix(in srgb, var(--accent) 22%, transparent);
            --summary-emphasis-shadow: 0 14px 34px color-mix(in srgb, var(--accent) 18%, transparent);
        }
        html, body, [class*="css"] { font-family: "IBM Plex Sans", sans-serif; color: var(--text); }
        .stApp {
            background:
                radial-gradient(circle at top left, var(--glow-left), transparent 34%),
                radial-gradient(circle at top right, var(--glow-right), transparent 30%),
                linear-gradient(180deg, var(--bg-start) 0%, var(--bg-end) 100%);
        }
        .block-container { max-width: min(1440px, 94vw); padding-top: 2.9rem; padding-bottom: 2rem; }
        h1, h2, h3, h4 { font-family: "Fraunces", serif; letter-spacing: -0.03em; color: var(--text) !important; }
        .hero, .card, .artifact {
            border: 1px solid var(--line); border-radius: 26px; background: var(--card);
            box-shadow: var(--shadow);
        }
        .hero { padding: 1.06rem 1.12rem 0.86rem; margin-bottom: 0.62rem; background: linear-gradient(135deg, var(--hero-start), var(--hero-end)); }
        .hero-kicker, .section-kicker, .mini-label {
            display: inline-block; color: var(--accent-deep); font-size: 0.78rem; font-weight: 700;
            letter-spacing: 0.12em; text-transform: uppercase;
        }
        .hero-kicker {
            margin-bottom: 0.45rem;
            color: var(--accent);
        }
        .hero-copy { max-width: 96ch; }
        .hero-brand {
            font-family: "Fraunces", serif;
            font-size: clamp(2.55rem, 4.35vw, 4.8rem);
            line-height: 0.9;
            letter-spacing: -0.05em;
            font-weight: 700;
            color: var(--text);
            margin: 0 0 0.42rem;
            max-width: none;
            white-space: nowrap;
        }
        .hero h1 { margin: 0.18rem 0 0.4rem; font-size: clamp(1.24rem, 1.78vw, 1.74rem); line-height: 1.14; max-width: 28ch; text-wrap: balance; color: var(--text) !important; }
        .hero p, .section-copy, .card p { color: var(--muted); line-height: 1.7; }
        .hero-copy p { max-width: 68ch; }
        .hero-grid { display: grid; grid-template-columns: minmax(0, 1.92fr) minmax(260px, 0.58fr); gap: 1rem; align-items: stretch; }
        .hero-panel { padding: 0.78rem 0.9rem; border-radius: 22px; background: var(--hero-panel); border: 1px solid var(--line); display: flex; flex-direction: column; justify-content: center; }
        .hero-panel p { margin: 0.2rem 0 0; }
        .section { margin: 1rem 0 0.85rem; }
        .section h2 { margin: 0.2rem 0 0; font-size: 1.65rem; line-height: 1.08; color: var(--text) !important; }
        .section.section-compact { margin-top: 0.9rem; }
        .section.section-compact h2 {
            font-size: clamp(1.3rem, 1.85vw, 1.5rem);
            line-height: 1.14;
            max-width: 24ch;
            text-wrap: balance;
        }
        .section.section-compact .section-copy { max-width: 72ch; }
        .card {
            padding: 1.28rem 1.1rem 1.32rem;
            min-height: 176px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 0.58rem;
        }
        .card h4 { margin: 0.42rem 0 0.55rem; font-size: 1.02rem; color: var(--text) !important; }
        .card-copy { flex: 1; display: flex; align-items: flex-start; }
        .card-copy p { margin: 0; width: 100%; }
        .card.card--compact { min-height: 166px; }
        .card.card--balanced { min-height: 166px; }
        .card.card--balanced .card-copy { min-height: 4.8rem; }
        .card.card--path p {
            font-size: 0.93rem;
            line-height: 1.52;
            word-break: break-word;
            overflow-wrap: anywhere;
        }
        .artifact-path, .artifact-path code {
            word-break: break-word;
            overflow-wrap: anywhere;
            white-space: pre-wrap;
        }
        .stat-card {
            min-height: 160px;
            padding: 1.3rem 1.05rem 1.34rem;
            margin-bottom: 1rem;
            border-radius: 22px;
            border: 1px solid var(--line);
            background: var(--card);
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 0.9rem;
        }
        .stat-card-label {
            color: var(--muted);
            font-size: 0.93rem;
            font-weight: 600;
        }
        .stat-card-value {
            font-family: "Fraunces", serif;
            color: var(--text);
            font-size: clamp(1.95rem, 2.5vw, 2.7rem);
            line-height: 0.98;
            letter-spacing: -0.04em;
            overflow-wrap: normal;
            word-break: normal;
            text-wrap: balance;
        }
        .stat-card-value.is-wide {
            font-size: clamp(1.58rem, 1.95vw, 2.1rem);
            line-height: 1.02;
        }
        .stat-card-value.is-phrase {
            font-size: clamp(1.18rem, 1.45vw, 1.56rem);
            line-height: 1.08;
            letter-spacing: -0.025em;
            max-width: 12ch;
            overflow-wrap: anywhere;
        }
        .stat-card-detail {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .prediction { padding: 1.2rem 1.3rem; border-radius: 24px; margin-bottom: 0.9rem;
            border: 1px solid var(--line); background: linear-gradient(135deg, var(--prediction-start), var(--prediction-end)); }
        .prediction h3 { margin: 0.55rem 0 0.2rem; font-size: 2.05rem; line-height: 1; color: var(--text) !important; }
        .artifact { padding: 1rem 1.1rem; margin: 0.8rem 0 1rem; border-style: dashed; background: var(--artifact-bg); }
        div[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line); border-radius: 20px; padding: 1rem 1.05rem; }
        div[data-testid="stMetric"] label { color: var(--muted) !important; }
        div[data-testid="stMetricValue"] { font-family: "Fraunces", serif; color: var(--text); }
        .stButton > button {
            border-radius: 999px; border: 0; background: linear-gradient(135deg, var(--accent), var(--accent-secondary));
            color: #fff8f0; font-weight: 700; padding: 0.72rem 1.4rem; box-shadow: var(--button-shadow);
        }
        .stButton > button:hover { background: linear-gradient(135deg, var(--accent-hover), var(--accent)); }
        .stButton > button:focus-visible { outline: 3px solid var(--line); outline-offset: 2px; }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.8rem;
            margin-bottom: 0.95rem;
            border-bottom: none !important;
            background: transparent !important;
        }
        .stTabs [data-baseweb="tab-border"],
        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.72rem 1.05rem;
            background: var(--field);
            border: 1px solid var(--line);
            color: var(--muted);
            transition: border-color 0.18s ease, background 0.18s ease, box-shadow 0.18s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            border-color: color-mix(in srgb, var(--accent) 50%, var(--line) 50%);
            color: var(--text);
        }
        .stTabs [aria-selected="true"] {
            background: var(--card) !important;
            border-color: var(--accent) !important;
            color: var(--text) !important;
            box-shadow: inset 0 -3px 0 var(--accent);
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] > div,
        div[data-baseweb="input"] > div,
        .stFileUploader > div {
            border-radius: 18px !important;
            border: 1px solid var(--field-border) !important;
            background: var(--field) !important;
            box-shadow: none !important;
            color: var(--text) !important;
        }
        div[data-baseweb="select"] * { color: var(--text) !important; }
        div[data-baseweb="select"] svg { color: var(--accent-deep) !important; }
        div[data-baseweb="base-input"] input,
        div[data-baseweb="input"] input {
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
            font-weight: 600;
        }
        div[data-testid="stFileUploaderDropzone"] {
            background: var(--field) !important;
            border: 1px dashed var(--accent) !important;
        }
        .stExpander {
            border: 1px solid var(--line) !important;
            border-radius: 18px !important;
            background: var(--expander-bg);
        }
        .stExpander summary, .stExpander details summary, .streamlit-expanderHeader {
            color: var(--text) !important;
            font-weight: 600;
        }
        .stExpander details { background: transparent !important; }
        .stExpander { margin-top: 0.85rem; }
        div[data-testid="stAlert"] {
            border-radius: 18px;
            border: 1px solid var(--line);
            background: var(--info);
        }
        div[data-testid="stAlert"] p { color: var(--info-text) !important; }
        .summary-strip {
            padding: 0.95rem 1rem;
            border-radius: 20px;
            background: var(--summary-bg);
            border: 1px solid var(--line);
            margin-bottom: 0.75rem;
        }
        .summary-strip.is-primary {
            background: var(--summary-bg-emphasis);
            border-color: var(--accent);
            box-shadow: var(--summary-emphasis-shadow);
        }
        .summary-strip strong { color: var(--text); }
        .summary-strip p { margin: 0.18rem 0 0; color: var(--muted); }
        [data-testid="stDataFrame"] { border-radius: 22px; overflow: hidden; border: 1px solid var(--line); }
        .footer-note { margin-top: 2rem; text-align: center; color: var(--muted); font-size: 0.92rem; }
        @media (max-width: 900px) {
            .block-container { max-width: calc(100vw - 1.5rem); }
            .hero-grid { grid-template-columns: 1fr; gap: 0.75rem; }
            .hero-copy, .hero h1, .hero-brand { max-width: none; }
            .hero-brand { white-space: normal; }
            .hero-panel { min-height: auto; }
            .card.card--balanced .card-copy { min-height: auto; }
            .stat-card-value.is-phrase { max-width: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <section class="hero">
            <div class="hero-grid">
                <div class="hero-copy">
                    <span class="hero-kicker">Phòng thí nghiệm thị giác màu sắc</span>
                    <div class="hero-brand">Color Correlogram Lab</div>
                    <h1>Upload ảnh, xem lớp dự đoán và đối chiếu với kết quả test.</h1>
                    <p>Giao diện này ưu tiên thao tác chính ở phía trên, giảm phần mô tả dài và giữ nội dung thuyết minh vừa đủ cho buổi demo đồ án.</p>
                </div>
                <div class="hero-panel">
                    <div class="mini-label">Trọng tâm demo</div>
                    <p>Tab Nhận dạng phục vụ thao tác trực tiếp với mô hình tĩnh (.pkl). Tab Thực nghiệm dùng để chạy benchmark các cách chia tách và đánh giá cấu hình pipeline thuật toán khác nhau.</p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(kicker, title, description, compact=False):
    section_class = "section section-compact" if compact else "section"
    st.markdown(
        f"""
        <div class="{section_class}">
            <div class="section-kicker">{escape(kicker)}</div>
            <h2>{escape(title)}</h2>
            <div class="section-copy">{escape(description)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(label, title, body, extra_class=""):
    card_class = "card"
    if extra_class:
        card_class = f"{card_class} {extra_class}"
    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="mini-label">{escape(label)}</div>
            <h4>{escape(title)}</h4>
            <div class="card-copy">
                <p>{escape(body)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_highlight(predicted_class, confidence, level):
    confidence_text = "Không có xác suất dự đoán." if confidence is None else f"Độ tin cậy: {confidence:.1%}"
    st.markdown(
        f"""
        <div class="prediction">
            <div class="mini-label">{escape(level)}</div>
            <h3>{escape(predicted_class)}</h3>
            <p>{escape(confidence_text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_artifact_box(paths):
    lines = "".join(f"<div class='artifact-path'><code>{escape(str(path))}</code></div>" for path in paths)
    st.markdown(f"<div class='artifact'><div class='mini-label'>Kết quả đã lưu</div>{lines}</div>", unsafe_allow_html=True)


def render_summary_strip(title, body, primary=False):
    primary_class = " is-primary" if primary else ""
    st.markdown(
        f"""
        <div class="summary-strip{primary_class}">
            <strong>{escape(title)}</strong>
            <p>{escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metadata_dict(data):
    st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")


def render_stat_card(label, value, detail=None):
    value_str = str(value)
    value_classes = ["stat-card-value"]
    word_count = len(value_str.replace("_", " ").split())
    if word_count >= 3 or len(value_str) >= 18:
        value_classes.append("is-phrase")
    elif len(value_str) >= 11:
        value_classes.append("is-wide")
    value_class = " ".join(value_classes)
    detail_html = f"<div class='stat-card-detail'>{escape(detail)}</div>" if detail else ""
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-card-label">{escape(label)}</div>
            <div class="{value_class}">{escape(value_str)}</div>
            {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_color_for_matplotlib(color_value):
    """Chuyển màu kiểu CSS sang định dạng Matplotlib chấp nhận."""
    if not isinstance(color_value, str):
        return color_value

    normalized = color_value.strip()
    rgba_match = re.fullmatch(
        r"rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})(?:\s*,\s*([01]?(?:\.\d+)?))?\s*\)",
        normalized,
    )
    if not rgba_match:
        return normalized

    red = int(rgba_match.group(1)) / 255.0
    green = int(rgba_match.group(2)) / 255.0
    blue = int(rgba_match.group(3)) / 255.0
    alpha = rgba_match.group(4)
    if alpha is None:
        return (red, green, blue)
    return (red, green, blue, float(alpha))


def plot_feature_chart(feature_vector, title):
    palette = get_theme_palette()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 3.3))
    ax.bar(
        range(len(feature_vector)),
        feature_vector,
        color=palette["chart_bar"],
        edgecolor=palette["chart_edge"],
        width=1,
        alpha=0.9,
    )
    ax.set_facecolor(palette["chart_bg"])
    fig.patch.set_facecolor(palette["chart_bg"])
    ax.grid(axis="y", color=normalize_color_for_matplotlib(palette["chart_grid"]), linestyle="--")
    ax.set_xlabel("Chiều đặc trưng")
    ax.set_ylabel("Giá trị")
    ax.set_title(title)
    ax.tick_params(colors=palette["muted"])
    ax.xaxis.label.set_color(palette["text"])
    ax.yaxis.label.set_color(palette["text"])
    ax.title.set_color(palette["text"])
    ax.spines["left"].set_color(normalize_color_for_matplotlib(palette["line"]))
    ax.spines["bottom"].set_color(normalize_color_for_matplotlib(palette["line"]))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


def load_class_names():
    class_names_path = FEATURES_DIR / "class_names.npy"
    if not class_names_path.exists():
        return None
    return np.load(class_names_path, allow_pickle=True)


def load_model_and_data(color_space):
    config = MODEL_CONFIGS[color_space]
    model_path = MODELS_DIR / config["model_file"]
    if not model_path.exists():
        return None, None, None, None

    class_names = load_class_names()
    if class_names is None:
        return None, None, None, None

    model = joblib.load(model_path)
    metadata_path = MODELS_DIR / f"{model_path.stem}.meta.json"
    model_metadata = None
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as file:
            model_metadata = json.load(file)

    paths_file = FEATURES_DIR / "image_paths.npy"
    features_file = FEATURES_DIR / config["features_file"]
    image_paths = None
    db_features = None
    if paths_file.exists() and features_file.exists():
        image_paths = np.load(paths_file, allow_pickle=True)
        db_features = np.load(features_file)
        if model_metadata and model_metadata.get("retrieval_split") == "train":
            split_file = model_metadata.get("split_file")
            if split_file and Path(split_file).exists():
                split_metadata = load_split_metadata(split_file)
                split_indices = resolve_split_indices(image_paths, split_metadata["data_dir"], split_metadata)
                retrieval_idx = split_indices["train"]
                image_paths = image_paths[retrieval_idx]
                db_features = db_features[retrieval_idx]

    return model, class_names, (image_paths, db_features), model_metadata


def find_similar_images(query_feat, db_features, image_paths, top_k=3):
    from scipy.spatial.distance import cdist

    distances = cdist([query_feat], db_features, metric="cosine")[0]
    indices = np.argsort(distances)[:top_k]
    return [(image_paths[i], Path(image_paths[i]).parent.name, distances[i]) for i in indices]


def render_summary_metrics(summary):
    cols = st.columns(4)
    with cols[0]:
        render_stat_card("Accuracy", f"{summary['accuracy'] * 100:.2f}%")
    with cols[1]:
        render_stat_card("Precision", f"{summary['precision']:.4f}")
    with cols[2]:
        render_stat_card("Recall", f"{summary['recall']:.4f}")
    with cols[3]:
        render_stat_card("F1-score", f"{summary['f1_score']:.4f}")
    if "accuracy_std" in summary:
        st.caption(
            f"Độ lệch chuẩn: acc={summary['accuracy_std']:.4f}, "
            f"precision={summary.get('precision_std', 0.0):.4f}, "
            f"recall={summary.get('recall_std', 0.0):.4f}, "
            f"f1={summary.get('f1_score_std', 0.0):.4f}"
        )


def render_prediction_tab():
    render_section_header(
        "Nhận dạng",
        "Upload ảnh để dự đoán",
        "Phần này đặt thao tác chính lên đầu: chọn mô hình, upload ảnh, xem kết quả và đối chiếu với ảnh gần nhất trong tập train.",
    )

    color_space_label = st.selectbox("Không gian màu", ["HSV", "RGB"], key="predict_color_space")
    color_space = color_space_label.lower()
    config = MODEL_CONFIGS[color_space]
    model, class_names, db_data, model_metadata = load_model_and_data(color_space)
    if model is None:
        st.error("Chưa có mô hình hoặc file đặc trưng cần thiết.")
        st.code("pip install -r requirements.txt\npython src/feature_extraction.py\npython src/train.py\nstreamlit run app.py")
        return

    top_cols = st.columns([1.15, 0.85])
    with top_cols[0]:
        uploaded_file = st.file_uploader("Tải lên ảnh cần nhận dạng", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], key="predict_upload")
        st.caption("Hỗ trợ JPG, JPEG, PNG, BMP, TIF. Ảnh sẽ được đưa về 128x128 trước khi trích xuất đặc trưng.")
        config_cols = st.columns(3)
        if color_space == "hsv":
            config_cols[0].number_input("H bins", value=config["h_bins"], step=1, disabled=True, key="predict_h_bins")
            config_cols[1].number_input("S bins", value=config["s_bins"], step=1, disabled=True, key="predict_s_bins")
            config_cols[2].number_input("V bins", value=config["v_bins"], step=1, disabled=True, key="predict_v_bins")
        else:
            config_cols[0].number_input("RGB bins", value=config["rgb_bins"], step=1, disabled=True, key="predict_rgb_bins")
        st.caption(config["sidebar_message"])
    with top_cols[1]:
        if color_space == "hsv":
            quantization = f"H={config['h_bins']}, S={config['s_bins']}, V={config['v_bins']}"
        else:
            quantization = f"RGB bins = {config['rgb_bins']}"
        render_summary_strip("Mô hình đang dùng", f"{config['label']} | {quantization}", primary=True)
        retrieval_note = "Truy hồi trên train split nếu feature và image paths có sẵn."
        if model_metadata:
            retrieval_note = (
                f"Tune trên {model_metadata.get('tuning_split', 'train')}, "
                f"final trên {model_metadata.get('final_training_split', 'train')}, "
                f"test ở {model_metadata.get('held_out_test_split', 'test')}."
            )
        render_summary_strip("Nguồn gốc thí nghiệm", retrieval_note)
        render_summary_strip("Phù hợp để demo", "Mô hình học trên 10 lớp Corel-1K, nên ảnh ngoài miền dữ liệu gốc cần được diễn giải thận trọng.")

    aux_cols = st.columns([0.72, 0.28])
    with aux_cols[0]:
        with st.expander("Danh sách lớp nhận dạng", expanded=False):
            for index, name in enumerate(class_names, start=1):
                st.write(f"{index}. {name}")
    with aux_cols[1]:
        if model_metadata and isinstance(model_metadata.get("split_counts"), dict):
            with st.expander("Chi tiết split", expanded=False):
                render_metadata_dict(
                    {
                        "split_counts": model_metadata.get("split_counts"),
                        "split_file": model_metadata.get("split_file"),
                        "retrieval_split": model_metadata.get("retrieval_split"),
                    }
                )

    if uploaded_file is None:
        st.info("Tải một ảnh bất kỳ để hiển thị vector đặc trưng, lớp dự đoán và gallery ảnh tương tự.")
        return

    st.info("Mô hình chỉ học trên 10 lớp của Corel-1K. Ảnh ngoài miền dữ liệu gốc vẫn sẽ bị ép vào lớp gần nhất theo đặc trưng màu sắc.")

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Không đọc được ảnh đã upload. Hãy thử lại với file hợp lệ.")
        return
    img = cv2.resize(img, (128, 128))

    with st.spinner("Đang trích xuất đặc trưng và dự đoán..."):
        feat = extract_correlogram_feature(
            img,
            color_space=color_space,
            h_bins=config["h_bins"],
            s_bins=config["s_bins"],
            v_bins=config["v_bins"],
            rgb_bins=config["rgb_bins"],
            spatial_grid=config["spatial_grid"],
        )
        prediction = model.predict([feat])
        predicted_class = class_names[prediction[0]]
        proba = model.predict_proba([feat])[0] if hasattr(model, "predict_proba") else None

    render_section_header(
        "Phân tích",
        "Kết quả trực tiếp",
        "Bên trái là ảnh đã chuẩn hóa, bên phải là vector đặc trưng; bên dưới là lớp dự đoán và xếp hạng các lớp gần nhất.",
    )

    visual_cols = st.columns([0.86, 1.14])
    with visual_cols[0]:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Ảnh sau khi đưa về 128x128", use_container_width=True)
    with visual_cols[1]:
        fig = plot_feature_chart(feat, f"Biểu đồ Color Correlogram ({config['label']})")
        st.pyplot(fig)

    confidence = float(proba[prediction[0]]) if proba is not None else None
    level = "Độ tin cậy cao" if confidence is not None and confidence >= 0.5 else "Cần thận trọng khi diễn giải"
    if confidence is None:
        level = "Không có xác suất"
    render_prediction_highlight(predicted_class, confidence, level)

    if proba is not None:
        top_indices = np.argsort(proba)[::-1][:3]
        rank_cols = st.columns(3)
        for idx, class_index in enumerate(top_indices):
            with rank_cols[idx]:
                probability = float(proba[class_index])
                render_info_card(f"Top {idx + 1}", str(class_names[class_index]), f"Xác suất ước tính {probability:.1%}")
                st.progress(probability)
    else:
        st.info("Mô hình hiện tại không hỗ trợ predict_proba, vì vậy app chỉ hiển thị lớp dự đoán.")

    if db_data[0] is not None and db_data[1] is not None:
        render_section_header(
            "Truy hồi",
            "Ảnh tương tự trong tập train",
            "Gallery này giúp đối chiếu xem ảnh vừa upload đang gần nhất với cụm dữ liệu nào trong không gian đặc trưng.",
        )
        try:
            similar = find_similar_images(feat, db_data[1], db_data[0], top_k=5)
            cols = st.columns(5)
            for index, (path, label, distance) in enumerate(similar):
                if os.path.exists(path):
                    sim_img = cv2.imread(path)
                    sim_img = cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB)
                    with cols[index]:
                        st.image(sim_img, caption=f"{label} | cosine={distance:.3f}", use_container_width=True)
        except Exception as exc:
            st.warning(f"Không thể tìm ảnh tương tự: {exc}")


def render_experiment_table(report_dict):
    rows = []
    for label, values in report_dict.items():
        if not isinstance(values, dict) or label in ("macro avg", "weighted avg"):
            continue
        rows.append(
            {
                "class": label,
                "precision": round(values["precision"], 4),
                "recall": round(values["recall"], 4),
                "f1-score": round(values["f1-score"], 4),
                "support": int(values["support"]),
            }
        )
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)

    macro_avg = report_dict.get("macro avg")
    weighted_avg = report_dict.get("weighted avg")
    if isinstance(macro_avg, dict) and isinstance(weighted_avg, dict):
        cols = st.columns(2)
        with cols[0]:
            render_info_card(
                "Tổng hợp",
                "Macro avg",
                f"Precision {macro_avg['precision']:.4f} | Recall {macro_avg['recall']:.4f} | F1 {macro_avg['f1-score']:.4f}",
            )
        with cols[1]:
            render_info_card(
                "Tổng hợp",
                "Weighted avg",
                f"Precision {weighted_avg['precision']:.4f} | Recall {weighted_avg['recall']:.4f} | F1 {weighted_avg['f1-score']:.4f}",
            )


def render_evaluation_controls():
    render_section_header(
        "Thực nghiệm Thuật toán / Pipeline",
        "Chọn kỹ thuật lượng giá thực nghiệm (Cross-validation / Resampling)",
        "Tại đây, hệ thống sẽ tự động khởi tạo dữ liệu, chia tách và train lại pipeline mô hình một cách linh hoạt (Dynamic Training) để đo độ ổn định của thuật toán tổng thể.",
        compact=True,
    )
    eval_method = st.selectbox(
        "Kỹ thuật lượng giá",
        EVAL_METHOD_OPTIONS,
        index=0,
        format_func=lambda value: EVAL_METHOD_LABELS_VI[value],
        key="eval_method",
    )
    render_summary_strip("Quy trình", EVAL_METHOD_HINTS[eval_method], primary=(eval_method == "independent_test"))

    top_cols = st.columns(4)
    feature = top_cols[0].selectbox(
        "Đặc trưng",
        ["spatial_correlogram", "correlogram", "histogram"],
        format_func=lambda value: {
            "spatial_correlogram": "Spatial Correlogram",
            "correlogram": "Correlogram",
            "histogram": "Histogram",
        }[value],
        key="eval_feature",
    )
    if feature == "spatial_correlogram":
        color = "hsv"
        top_cols[1].markdown("**Không gian màu**  \nHSV (cố định)")
    else:
        color = top_cols[1].selectbox("Không gian màu", ["hsv", "rgb"], key="eval_color")
    model_name = top_cols[2].selectbox("Mô hình", ["svm", "knn"], key="eval_model")
    top_cols[3].markdown(f"**Lượng giá**  \n{EVAL_METHOD_LABELS_VI[eval_method]}")

    params = {
        "test_size": 0.2,
        "k": 5,
        "n_repeats": 10,
        "n_iterations": 30,
        "sample_ratio": 0.8,
        "random_state": 42,
    }

    if eval_method in {"holdout", "stratified_holdout", "repeated_holdout"}:
        sampling_cols = st.columns(3 if eval_method == "repeated_holdout" else 2)
        params["test_size"] = sampling_cols[0].slider(
            "Tỉ lệ test",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            key=f"{eval_method}_test_size",
        )
        params["random_state"] = sampling_cols[1].number_input(
            "Random state",
            min_value=0,
            value=42,
            step=1,
            key=f"{eval_method}_random_state",
        )
        if eval_method == "repeated_holdout":
            params["n_repeats"] = sampling_cols[2].number_input(
                "Số lần lặp",
                min_value=2,
                value=10,
                step=1,
                key="repeated_holdout_n_repeats",
            )
    elif eval_method == "kfold":
        cv_cols = st.columns(2)
        params["k"] = cv_cols[0].number_input(
            "Số fold (k)",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            key="kfold_k",
        )
        params["random_state"] = cv_cols[1].number_input(
            "Random state",
            min_value=0,
            value=42,
            step=1,
            key="kfold_random_state",
        )
    elif eval_method == "bootstrap":
        bootstrap_cols = st.columns(3)
        params["n_iterations"] = bootstrap_cols[0].number_input(
            "Số vòng bootstrap",
            min_value=5,
            value=30,
            step=5,
            key="bootstrap_iterations",
        )
        params["sample_ratio"] = bootstrap_cols[1].slider(
            "Tỉ lệ lấy mẫu",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05,
            key="bootstrap_sample_ratio",
        )
        params["random_state"] = bootstrap_cols[2].number_input(
            "Random state",
            min_value=0,
            value=42,
            step=1,
            key="bootstrap_random_state",
        )
    elif eval_method == "leave_one_out":
        st.caption("Leave-One-Out không cần tham số chia dữ liệu, nhưng thời gian chạy có thể lâu hơn rõ rệt.")
    else:
        st.caption("Independent Held-out Test dùng split cố định của dự án để phản ánh đúng protocol báo cáo cuối.")

    return feature, color, model_name, eval_method, params


def render_evaluation_result(result, report_text):
    render_artifact_box(
        [
            result["artifacts"]["json"],
            result["artifacts"]["report"],
            result["artifacts"]["confusion_matrix"],
        ]
    )
    info_cols = st.columns(4)
    with info_cols[0]:
        render_stat_card("Đặc trưng", str(result["feature"]))
    with info_cols[1]:
        render_stat_card("Màu", str(result["color_space"]).upper())
    with info_cols[2]:
        render_stat_card("Mô hình", str(result["model"]).upper())
    with info_cols[3]:
        render_stat_card("Kỹ thuật", str(result["evaluation_label"]))
    render_summary_metrics(result["summary"])

    render_section_header(
        "Nguồn gốc thí nghiệm",
        "Thông tin dữ liệu và phân vùng (Split)",
        "Giải thích nhanh kích thước dữ liệu, file vector đặc trưng đang được nạp để train/test lại từ đầu cho lượt đánh giá này.",
        compact=True,
    )
    data_cols = st.columns(2)
    with data_cols[0]:
        render_info_card(
            "Dữ liệu",
            "Thông tin chính",
            f"Kích thước dataset: {result['dataset_shape']} | Kích thước eval: {result.get('evaluation_dataset_shape')} | Số lớp: {result['class_count']}",
            extra_class="card--compact card--balanced",
        )
    with data_cols[1]:
        render_info_card(
            "File đặc trưng",
            "Nguồn artifact",
            str(result["feature_file"]),
            extra_class="card--compact card--balanced card--path",
        )
    if result.get("evaluation_details"):
        with st.expander("Chi tiết split", expanded=False):
            render_metadata_dict(
                {
                    "split_file": result["evaluation_details"].get("split_file"),
                    "split_counts": result["evaluation_details"].get("split_counts"),
                    "final_training_split": result["evaluation_details"].get("final_training_split"),
                }
            )

    render_section_header(
        "Báo cáo",
        "Báo cáo phân loại theo lớp",
        "Bảng dưới đây phù hợp cho việc đọc nhanh precision, recall, f1-score và support theo từng nhãn.",
        compact=True,
    )
    render_experiment_table(result["classification_report"])
    with st.expander("Xem classification report dạng text", expanded=False):
        st.text(report_text)

    render_section_header(
        "Ma trận nhầm lẫn",
        "Tổng hợp lỗi nhầm lẫn",
        "Quan sát các ô đậm màu để thấy lớp nào đang bị nhầm với nhau nhiều nhất trên tập test độc lập.",
        compact=True,
    )
    st.image(result["artifacts"]["confusion_matrix"], use_container_width=True)


def render_evaluation_tab():
    feature, color, model_name, eval_method, params = render_evaluation_controls()
    if st.button("Chạy thí nghiệm", type="primary"):
        with st.spinner("Đang chạy thí nghiệm, vui lòng chờ..."):
            result, report_text = run_experiment(
                dataset_profile_key=DEFAULT_DATASET_PROFILE,
                feature=feature,
                color=color,
                model_name=model_name,
                eval_method=eval_method,
                test_size=params["test_size"],
                k=params["k"],
                n_repeats=params["n_repeats"],
                n_iterations=params["n_iterations"],
                sample_ratio=params["sample_ratio"],
                random_state=params["random_state"],
            )
        st.session_state["experiment_result"] = result
        st.session_state["experiment_report_text"] = report_text

    result = st.session_state.get("experiment_result")
    report_text = st.session_state.get("experiment_report_text")
    if not result or not report_text:
        st.caption("Chọn cấu hình ở trên và bấm 'Chạy thí nghiệm' để hiển thị metric, report và confusion matrix.")
        return
    render_evaluation_result(result, report_text)


def main():
    st.set_page_config(page_title="Color Correlogram - Nhận dạng ảnh", page_icon="🎨", layout="wide")
    sync_theme_state()
    inject_custom_css()
    render_hero()
    tab_predict, tab_evaluate = st.tabs(["Nhận dạng ảnh", "Thực nghiệm thuật toán"])
    with tab_predict:
        render_prediction_tab()
    with tab_evaluate:
        render_evaluation_tab()
    st.markdown(
        """
        <div class="footer-note">
            Đồ án môn Khai phá dữ liệu đa phương tiện | Color Correlogram + bộ công cụ đánh giá ML
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
