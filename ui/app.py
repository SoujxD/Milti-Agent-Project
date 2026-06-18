from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agents.analyst_agent import AnalystRAGAgent
from agents.eda_agent import EDAAgent
from agents.presentation_agent import PresentationGeneratorAgent
from evaluation.evaluator import EvaluationPipeline


DEFAULT_DATASET_PATH = BASE_DIR / "data" / "dataset.csv"
QUESTIONS_PATH = BASE_DIR / "data" / "evaluation_questions.json"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_ANALYSIS_ROWS = 25_000

SAMPLE_QUESTIONS = [
    "Which customer segments should we prioritize?",
    "What patterns distinguish converting sessions from non-converting ones?",
    "Which channels are most promising for budget allocation?",
    "What recommendations would improve conversion performance?",
]


# ─────────────────────────────────────────
# Styles
# ─────────────────────────────────────────
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --border: rgba(21, 76, 97, 0.09);
            --shadow: 0 8px 24px rgba(17, 52, 68, 0.07);
            --shadow-md: 0 14px 38px rgba(17, 52, 68, 0.11);
            --text: #163747;
            --muted: #5e7a87;
            --accent: #1d667d;
            --accent-soft: rgba(29, 102, 125, 0.09);
        }

        /* ── App background ── */
        .stApp {
            background:
                radial-gradient(ellipse at 80% 0%, rgba(78, 165, 217, 0.12) 0%, transparent 50%),
                linear-gradient(180deg, #f9fcfe 0%, #f1f6f9 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1200px;
            padding-top: 1rem;
            padding-bottom: 3rem;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: rgba(247, 251, 253, 0.96);
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebarNav"] { display: none; }

        /* ── Topbar ── */
        .topbar {
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: var(--shadow);
            padding: 1.2rem 1.5rem 1rem;
            margin-bottom: 0.6rem;
        }
        .eyebrow {
            font-size: 0.68rem;
            letter-spacing: 0.16em;
            font-weight: 700;
            text-transform: uppercase;
            color: #4a8599;
        }
        .hero-title {
            font-size: 1.75rem;
            font-weight: 750;
            line-height: 1.1;
            background: linear-gradient(130deg, #163747 30%, #1d849e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0.25rem 0 0.3rem;
        }
        .hero-sub {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.5;
        }
        .pill-row { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.5rem; }
        .pill {
            background: var(--accent-soft);
            color: var(--accent);
            border: 1px solid rgba(29,102,125,0.1);
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            white-space: nowrap;
        }
        .pill.hi { background: var(--accent); color: white; border-color: transparent; }

        /* ── Section header ── */
        .sec-head {
            border-left: 3px solid var(--accent);
            padding-left: 0.6rem;
            margin: 0.4rem 0 1rem;
        }
        .sec-title { font-size: 1.05rem; font-weight: 750; color: #173746; }
        .sec-sub { color: var(--muted); font-size: 0.86rem; margin-top: 0.12rem; }

        /* ── Cards ── */
        .card {
            background: rgba(255,255,255,0.9);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: var(--shadow);
            padding: 1.1rem 1.2rem;
        }
        .card-accent { border-top: 3px solid var(--accent); }

        /* ── Metric card ── */
        .mc {
            background: rgba(255,255,255,0.9);
            border: 1px solid var(--border);
            border-top: 3px solid var(--accent);
            border-radius: 18px;
            box-shadow: var(--shadow);
            padding: 0.95rem 1.1rem;
            height: 100%;
        }
        .mc-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: #6b8794; }
        .mc-val { font-size: 1.55rem; font-weight: 760; color: var(--accent); line-height: 1.1; margin-top: 0.15rem; }
        .mc-cap { font-size: 0.84rem; color: var(--muted); margin-top: 0.2rem; }

        /* ── Upload zone ── */
        .upload-zone {
            border: 2px dashed rgba(29,102,125,0.25);
            border-radius: 18px;
            padding: 1.5rem 1.2rem;
            text-align: center;
            background: rgba(29,102,125,0.03);
            margin-bottom: 0.8rem;
        }
        .upload-zone-title { font-weight: 700; color: var(--accent); font-size: 0.96rem; }
        .upload-zone-sub { color: var(--muted); font-size: 0.84rem; margin-top: 0.3rem; }

        /* ── Schema table ── */
        .schema-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.42rem 0;
            border-bottom: 1px solid var(--border);
            font-size: 0.88rem;
        }
        .schema-row:last-child { border-bottom: none; }
        .schema-key { color: var(--muted); font-weight: 600; }
        .schema-val { font-family: monospace; color: var(--accent); font-size: 0.84rem; font-weight: 700; }
        .schema-nd { color: #aac; font-style: italic; }

        /* ── Analyst response ── */
        .res-section {
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.65rem;
            border: 1px solid var(--border);
        }
        .res-summary { background: rgba(29,102,125,0.05); border-left: 3px solid var(--accent); }
        .res-insights { background: rgba(29,102,125,0.03); border-left: 3px solid #4ea5d9; }
        .res-patterns { background: rgba(180,220,200,0.12); border-left: 3px solid #3aaa82; }
        .res-recs { background: rgba(255,230,180,0.15); border-left: 3px solid #c98a2e; }
        .res-label {
            font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
            text-transform: uppercase; margin-bottom: 0.4rem;
        }
        .res-summary .res-label { color: var(--accent); }
        .res-insights .res-label { color: #2a7da8; }
        .res-patterns .res-label { color: #2a8a68; }
        .res-recs .res-label { color: #a07020; }
        .res-text { color: var(--text); font-size: 0.9rem; line-height: 1.6; }
        .res-list { margin: 0; padding-left: 1.1rem; color: var(--text); font-size: 0.9rem; line-height: 1.65; }

        /* ── Evidence block ── */
        .evidence-wrap {
            background: #f4f8fb;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            font-family: "SF Mono", "Fira Code", monospace;
            font-size: 0.78rem;
            line-height: 1.6;
            color: #3d5a66;
            max-height: 320px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }

        /* ── Score badges ── */
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
        }
        .badge-hi { background: rgba(29,122,97,0.11); color: #1d7a61; }
        .badge-md { background: rgba(176,125,46,0.11); color: #b07d2e; }
        .badge-lo { background: rgba(160,48,48,0.11); color: #a03030; }

        /* ── Eval section divider ── */
        .eval-group {
            background: rgba(255,255,255,0.85);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 1.1rem;
            box-shadow: var(--shadow);
        }
        .eval-group-title {
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }

        /* ── Slide card ── */
        .slide-card {
            background: rgba(255,255,255,0.9);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: var(--shadow);
            overflow: hidden;
            margin-bottom: 1rem;
        }
        .slide-header {
            background: linear-gradient(135deg, rgba(29,102,125,0.07), rgba(78,165,217,0.06));
            padding: 0.9rem 1.1rem 0.7rem;
            border-bottom: 1px solid var(--border);
        }
        .slide-num {
            font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
            text-transform: uppercase; color: var(--accent); margin-bottom: 0.2rem;
        }
        .slide-title { font-size: 1rem; font-weight: 730; color: #173746; }
        .slide-body { padding: 0.85rem 1.1rem; }
        .slide-bullets { margin: 0; padding-left: 1.1rem; color: var(--text); font-size: 0.88rem; line-height: 1.7; }
        .notes-box {
            margin-top: 0.6rem;
            padding: 0.7rem 0.85rem;
            border-radius: 12px;
            background: rgba(245,249,252,0.95);
            border: 1px solid rgba(29,102,125,0.08);
            color: #526a76;
            font-size: 0.85rem;
            line-height: 1.5;
        }
        .notes-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.25rem; }

        /* ── Buttons ── */
        .stButton > button, .stDownloadButton > button {
            border-radius: 999px !important;
            font-weight: 600 !important;
            transition: all 0.18s !important;
        }
        .stButton > button[kind="primary"] { background: var(--accent) !important; color: white !important; border: none !important; }

        /* ── Inputs ── */
        .stTextArea textarea, .stTextInput input, .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div { border-radius: 12px !important; }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.3rem;
            background: transparent;
            border-bottom: 1px solid var(--border);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: 1px solid transparent;
            border-bottom: none;
            border-radius: 10px 10px 0 0;
            padding: 0.5rem 1rem;
            color: var(--muted);
            font-weight: 600;
            font-size: 0.88rem;
        }
        .stTabs [data-baseweb="tab"]:hover { background: rgba(29,102,125,0.05); color: var(--accent); }
        .stTabs [aria-selected="true"] { background: rgba(255,255,255,0.9) !important; color: var(--accent) !important; border-color: var(--border) !important; }

        /* ── Expander ── */
        .stExpander { border: 1px solid var(--border) !important; border-radius: 14px !important; }

        /* ── Alerts ── */
        .stAlert { border-radius: 14px !important; }

        /* ── Misc ── */
        .gap { height: 0.6rem; }
        .small { color: var(--muted); font-size: 0.84rem; line-height: 1.5; }

        /* ── Sidebar stats ── */
        .sb-row {
            display: flex; justify-content: space-between;
            font-size: 0.86rem; padding: 5px 0;
            border-bottom: 1px solid var(--border);
        }
        .sb-row span:last-child { font-weight: 700; color: var(--accent); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────
# Loaders & Cache
# ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def build_agent(path: str) -> AnalystRAGAgent:
    return AnalystRAGAgent(dataset_path=path)


@st.cache_resource(show_spinner=False)
def build_presenter() -> PresentationGeneratorAgent:
    return PresentationGeneratorAgent(output_dir=OUTPUT_DIR)


@st.cache_resource(show_spinner=False)
def build_eda_agent() -> EDAAgent:
    return EDAAgent(output_dir=OUTPUT_DIR)


def load_results() -> pd.DataFrame | None:
    p = OUTPUT_DIR / "results.csv"
    return pd.read_csv(p) if p.exists() else None


# ─────────────────────────────────────────
# Session
# ─────────────────────────────────────────
def ensure_defaults() -> None:
    defaults: dict[str, Any] = {
        "ds_path": str(DEFAULT_DATASET_PATH),
        "ds_name": DEFAULT_DATASET_PATH.name,
        "sampling_info": None,
        "question": SAMPLE_QUESTIONS[0],
        "analyst_result": None,
        "eda_report": None,
        "ppt_path": None,
        "slides": [],
        "chart_paths": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def rerun() -> None:
    st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()


def get_active() -> tuple[pd.DataFrame, str, str, AnalystRAGAgent]:
    path = st.session_state["ds_path"]
    name = st.session_state["ds_name"]
    return load_dataset(path), name, path, build_agent(path)


def reset_derived_state() -> None:
    st.session_state["analyst_result"] = None
    st.session_state["eda_report"] = None
    st.session_state["ppt_path"] = None
    st.session_state["slides"] = []
    st.session_state["chart_paths"] = {}


# ─────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────
def metric_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f'<div class="mc"><div class="mc-label">{label}</div>'
        f'<div class="mc-val">{value}</div>'
        f'<div class="mc-cap">{caption}</div></div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    sub = f'<div class="sec-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="sec-head"><div class="sec-title">{title}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def gap() -> None:
    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)


def conf_badge(conf: str) -> str:
    cls = "badge-hi" if conf == "high" else ("badge-md" if conf in ("medium", "moderate") else "badge-lo")
    return f'<span class="badge {cls}">{conf.title()}</span>'


# ─────────────────────────────────────────
# Charts
# ─────────────────────────────────────────
def _fig_style(fig: plt.Figure, ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.2)
    ax.spines["bottom"].set_alpha(0.2)
    fig.patch.set_facecolor("#f7fafb")
    ax.set_facecolor("#f7fafb")


def plot_bar(series: pd.Series, title: str, horizontal: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, max(3.2, len(series) * 0.55)) if horizontal else (8, 3.8))
    colors = [f"#{int(29 + i * 12):02x}{int(102 + i * 8):02x}{int(125 + i * 6):02x}" for i in range(len(series))]
    if horizontal:
        series.sort_values().plot(kind="barh", ax=ax, color=colors)
        ax.set_ylabel("")
        ax.grid(alpha=0.1, axis="x")
    else:
        series.plot(kind="bar", ax=ax, color=colors)
        ax.set_xlabel("")
        ax.grid(alpha=0.1, axis="y")
        plt.xticks(rotation=20, ha="right")
    ax.set_title(title, fontsize=11, pad=8, color="#163747", fontweight="bold")
    _fig_style(fig, ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, max(2.8, len(df) * 0.7)))
    img = ax.imshow(df.values, cmap="GnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_title(title, fontsize=11, pad=8, color="#163747", fontweight="bold")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=22, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=8.5)
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            v = df.iloc[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.62 else "#153847")
    fig.colorbar(img, ax=ax, fraction=0.04, pad=0.03)
    _fig_style(fig, ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────
# Sidebar — status only
# ─────────────────────────────────────────
def render_sidebar(dataset: pd.DataFrame, dataset_name: str) -> None:
    with st.sidebar:
        st.markdown("### Active dataset")
        is_custom = dataset_name != DEFAULT_DATASET_PATH.name
        tag = " _(custom)_" if is_custom else " _(default)_"
        st.markdown(f"**{dataset_name}**{tag}")
        st.markdown(
            f'<div class="sb-row"><span>Rows</span><span>{len(dataset):,}</span></div>'
            f'<div class="sb-row"><span>Columns</span><span>{len(dataset.columns)}</span></div>',
            unsafe_allow_html=True,
        )
        sampling_info = st.session_state.get("sampling_info")
        if sampling_info and sampling_info.get("sampled"):
            st.caption(
                f"Sampled {sampling_info['analysis_rows']:,} of {sampling_info['source_rows']:,} uploaded rows for analysis."
            )
        gap()
        if st.button("Reset to default dataset", use_container_width=True):
            st.session_state["ds_path"] = str(DEFAULT_DATASET_PATH)
            st.session_state["ds_name"] = DEFAULT_DATASET_PATH.name
            st.session_state["sampling_info"] = None
            reset_derived_state()
            rerun()
        st.markdown("---")
        st.caption("ISE547 · Multi-agent ecommerce analytics · EDA · RAG analyst · Evaluation · Presentation")


# ─────────────────────────────────────────
# Topbar
# ─────────────────────────────────────────
def render_topbar(dataset_name: str, dataset: pd.DataFrame) -> None:
    is_custom = dataset_name != DEFAULT_DATASET_PATH.name
    pill_cls = "pill hi" if is_custom else "pill"
    sampling_info = st.session_state.get("sampling_info")
    sampling_pill = (
        f'<span class="pill">Sampled {sampling_info["analysis_rows"]:,}/{sampling_info["source_rows"]:,} rows</span>'
        if sampling_info and sampling_info.get("sampled")
        else ""
    )
    st.markdown(
        f"""
        <div class="topbar">
            <div class="eyebrow">Ecommerce Multi-Agent Analytics &nbsp;·&nbsp; ISE547</div>
            <div class="hero-title">From data to insights to presentation.</div>
            <div class="hero-sub">
                Upload a dataset, interrogate it with the analyst agent, benchmark models,
                and export a stakeholder deck — all without leaving this app.
            </div>
            <div class="pill-row">
                <span class="{pill_cls}">{dataset_name}</span>
                <span class="pill">{len(dataset):,} rows</span>
                <span class="pill">{len(dataset.columns)} columns</span>
                {sampling_pill}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────
# Tab 1 — Overview  (upload lives here)
# ─────────────────────────────────────────
def render_overview(dataset: pd.DataFrame, dataset_name: str) -> None:
    presenter = build_presenter()
    schema = presenter.infer_schema(dataset)
    summary = presenter.summarize_dataset(st.session_state["ds_path"])
    rate = f"{summary['target_rate']:.1%}" if summary["target_rate"] is not None else "N/A"

    # ── Upload zone ──────────────────────────
    section_header("Upload your dataset", "Drop in a new CSV to power the analyst, evaluation, and presentation tabs.")
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    st.markdown(
        '<div class="upload-zone-title">&#8593; Drag & drop a CSV file</div>'
        '<div class="upload-zone-sub">Ecommerce session data, customer data, or any tabular CSV with a target column. Large files are automatically sampled for faster analysis.</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        uploaded_df = pd.read_csv(uploaded)
        source_rows = len(uploaded_df)
        sampled = source_rows > MAX_ANALYSIS_ROWS
        analysis_df = (
            uploaded_df.sample(n=MAX_ANALYSIS_ROWS, random_state=42).reset_index(drop=True)
            if sampled else uploaded_df
        )
        saved = UPLOAD_DIR / uploaded.name
        analysis_df.to_csv(saved, index=False)
        if st.session_state["ds_path"] != str(saved):
            st.session_state["ds_path"] = str(saved)
            st.session_state["ds_name"] = uploaded.name
            st.session_state["sampling_info"] = {
                "sampled": sampled,
                "source_rows": source_rows,
                "analysis_rows": len(analysis_df),
            }
            reset_derived_state()
            rerun()

    gap()

    # ── Key metrics ──────────────────────────
    section_header("Dataset at a glance")
    cols = st.columns(4)
    sampling_info = st.session_state.get("sampling_info")
    row_label = "Source Rows" if sampling_info and sampling_info.get("sampled") else "Rows"
    row_caption = "Uploaded file size" if sampling_info and sampling_info.get("sampled") else "Total records"
    row_value = sampling_info["source_rows"] if sampling_info and sampling_info.get("sampled") else summary["rows"]
    with cols[0]: metric_card(row_label, f"{row_value:,}", row_caption)
    with cols[1]: metric_card("Columns", str(summary["columns"]), "Available fields")
    with cols[2]: metric_card("Outcome Rate", rate, "Detected conversion rate")
    with cols[3]:
        metric_card(
            "Analysis Grain" if sampling_info and sampling_info.get("sampled") else "Top Segment",
            f"{summary.get('analysis_rows', len(dataset)):,} {summary.get('analysis_grain', 'rows')}" if sampling_info and sampling_info.get("sampled") else str(summary["top_customer"]),
            "Normalized records used" if sampling_info and sampling_info.get("sampled") else "Best-performing group",
        )

    gap()

    # ── Schema + workflow ─────────────────────
    left, right = st.columns([1.1, 1])
    with left:
        section_header("Detected schema")
        rows_html = ""
        schema_items = {
            "Dataset": dataset_name,
            "Outcome column": schema.get("target"),
            "Customer grouping": schema.get("customer"),
            "Channel grouping": schema.get("channel"),
            "Time grouping": schema.get("time"),
        }
        for k, v in schema_items.items():
            val_html = f'<span class="schema-val">{v}</span>' if v else '<span class="schema-nd">not detected</span>'
            rows_html += f'<div class="schema-row"><span class="schema-key">{k}</span>{val_html}</div>'
        st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)

    with right:
        section_header("Workflow")
        st.markdown(
            '<div class="card">'
            "<p style='margin:0 0 0.6rem;font-size:0.88rem;color:var(--muted);'>Use the tabs above to move through each step:</p>"
            "<ol style='margin:0;padding-left:1.2rem;font-size:0.9rem;line-height:2;color:#163747;'>"
            "<li>Upload a CSV on this page</li>"
            "<li>Profile the dataset in <strong>EDA</strong></li>"
            "<li>Ask business questions in <strong>Analyst</strong></li>"
            "<li>Benchmark models in <strong>Evaluation</strong></li>"
            "<li>Export a deck in <strong>Presentation</strong></li>"
            "</ol>"
            f"<div style='margin-top:0.8rem;font-size:0.82rem;color:var(--muted);'>Top channel: <strong style='color:var(--accent)'>{summary['top_channel']}</strong>"
            f" &nbsp;·&nbsp; Top period: <strong style='color:var(--accent)'>{summary['top_time']}</strong></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if sampling_info and sampling_info.get("sampled"):
            st.info(
                f"Large upload detected. The app is analyzing a {sampling_info['analysis_rows']:,}-row sample from the original {sampling_info['source_rows']:,}-row file."
            )

    gap()

    # ── Data preview ─────────────────────────
    section_header("Data preview", "First 20 rows of the active dataset")
    st.dataframe(dataset.head(20), use_container_width=True, hide_index=False)


# ─────────────────────────────────────────
# Tab 2 — EDA
# ─────────────────────────────────────────
def render_eda(dataset_path: str) -> None:
    section_header(
        "EDA Agent",
        "Profile the active ecommerce dataset before analyst reasoning: schema detection, quality checks, visuals, and handoff context.",
    )

    run_col, hint_col = st.columns([0.26, 0.74])
    with run_col:
        run = st.button("Run EDA", type="primary", use_container_width=True)
    with hint_col:
        st.markdown(
            '<div class="small" style="margin-top:0.65rem;">Uses deterministic profiling and lightweight charts. Event logs are normalized to session-level records first.</div>',
            unsafe_allow_html=True,
        )

    if run or st.session_state.get("eda_report") is None:
        eda_agent = build_eda_agent()
        dataset_file = Path(dataset_path)
        cache_key = f"{dataset_file.stem}_{dataset_file.stat().st_mtime_ns}"
        chart_prefix = f"streamlit_{dataset_file.stem[:18]}_{dataset_file.stat().st_mtime_ns % 100000}"
        with st.spinner("Profiling dataset with the EDA agent…"):
            st.session_state["eda_report"] = eda_agent.analyze_dataset(
                dataset_path=dataset_path,
                include_charts=True,
                chart_prefix=chart_prefix,
                cache_key=cache_key,
            )

    report = st.session_state.get("eda_report")
    if not report:
        st.info("Click **Run EDA** to profile the active dataset.")
        return

    profile = report["profile"]
    schema = profile.get("schema", {})
    quality = report["quality_checks"]
    rate = f"{profile['target_rate']:.1%}" if profile.get("target_rate") is not None else "N/A"

    m_cols = st.columns(4)
    with m_cols[0]: metric_card("Source Format", str(profile.get("source_format", "unknown")), "Detected input structure")
    with m_cols[1]: metric_card("Analysis Grain", str(profile.get("analysis_grain", "rows")), "Normalized level")
    with m_cols[2]: metric_card("Analysis Rows", f"{profile.get('analysis_rows', 0):,}", "Rows used after adaptation")
    with m_cols[3]: metric_card("Outcome Rate", rate, "Detected purchase/conversion rate")

    gap()
    left, right = st.columns([1, 1])
    with left:
        section_header("Detected schema")
        schema_rows = ""
        for key in ["target", "time", "customer", "channel", "product", "engagement", "friction"]:
            value = schema.get(key)
            value_html = f'<span class="schema-val">{value}</span>' if value else '<span class="schema-nd">not detected</span>'
            schema_rows += f'<div class="schema-row"><span class="schema-key">{key.title()}</span>{value_html}</div>'
        st.markdown(f'<div class="card">{schema_rows}</div>', unsafe_allow_html=True)

    with right:
        section_header("Quality checks")
        warnings = quality.get("warnings", [])
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("No major data quality warnings detected.")
        st.caption(f"Duplicate rows detected: {quality.get('duplicate_rows', 0):,}")

    gap()
    chart_manifest = report.get("chart_manifest", [])
    section_header("Auto-generated EDA visuals", "Up to four diverse charts generated from the normalized analysis dataset.")
    if chart_manifest:
        chart_cols = st.columns(2)
        for idx, chart in enumerate(chart_manifest):
            with chart_cols[idx % 2]:
                st.markdown(
                    f'<div class="card"><div class="sec-title">{chart["title"]}</div>'
                    f'<div class="sec-sub">{chart["caption"]}</div></div>',
                    unsafe_allow_html=True,
                )
                st.image(str(chart["path"]), width=540)
    else:
        st.info("No chartable schema was detected for this dataset.")

    gap()
    q_left, q_right = st.columns([0.9, 1.1])
    with q_left:
        section_header("Suggested business questions")
        questions = report.get("suggested_questions", [])
        if questions:
            st.markdown("\n".join(f"- {question}" for question in questions))
        else:
            st.caption("No suggested questions were generated.")

        section_header("Key findings")
        findings = report.get("key_findings", [])
        if findings:
            st.markdown("\n".join(f"- {finding}" for finding in findings))
        else:
            st.caption("No key findings were generated.")

    with q_right:
        section_header("Handoff summary for analyst")
        st.json(report.get("handoff_summary", {}), expanded=True)

    with st.expander("Missingness details", expanded=False):
        st.dataframe(pd.DataFrame(quality.get("missing_summary", [])), use_container_width=True)


# ─────────────────────────────────────────
# Tab 3 — Analyst
# ─────────────────────────────────────────
def render_analyst(agent: AnalystRAGAgent) -> None:
    section_header(
        "Analyst Agent",
        "Ask a business question — the agent retrieves relevant data, reasons over it, and returns a structured answer.",
    )

    # Controls
    ctrl = st.columns([1.1, 1.1, 1, 1])
    with ctrl[0]: model = st.selectbox("Model", agent.available_models(), key="a_model")
    with ctrl[1]: prompt_style = st.selectbox("Prompt style", agent.available_prompt_styles(), key="a_prompt")
    with ctrl[2]: top_k = st.slider("Retrieved rows", 1, 8, 5, key="a_topk")
    with ctrl[3]: rag = st.toggle("Use RAG context", value=True, key="a_rag")

    gap()

    # Sample questions
    st.markdown('<div class="small" style="margin-bottom:0.4rem;">Try a sample question:</div>', unsafe_allow_html=True)
    sq_cols = st.columns(len(SAMPLE_QUESTIONS))
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if sq_cols[i].button(f"Sample {i+1}", key=f"sq_{i}", use_container_width=True, help=q):
            st.session_state["question"] = q
            rerun()

    # Question input
    st.text_area(
        "Your question",
        key="question",
        height=100,
        placeholder="e.g. Which visitor segments have the highest conversion rate and why?",
        label_visibility="collapsed",
    )

    run_col, hint_col = st.columns([0.25, 0.75])
    with run_col:
        run = st.button("Run agent", type="primary", use_container_width=True)
    with hint_col:
        st.markdown('<div class="small" style="margin-top:0.65rem;">Generates an answer from the active dataset.</div>', unsafe_allow_html=True)

    if run:
        with st.spinner("Thinking…"):
            res = agent.answer_question(
                question=st.session_state["question"],
                model=model,
                prompt_style=prompt_style,
                rag_enabled=rag,
                top_k=top_k,
            )
        st.session_state["analyst_result"] = res

    result = st.session_state.get("analyst_result")
    if not result:
        st.info("Choose a sample question or write your own above, then click **Run agent**.")
        return

    gap()
    parsed = result.parsed_response
    conf = parsed.get("confidence", "low")

    # ── Structured response ───────────────────
    res_col, right_col = st.columns([1.05, 0.95])

    with res_col:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.6rem;">'
            f'<span style="font-weight:700;font-size:1rem;color:#173746;">Response</span>'
            f'{conf_badge(conf)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        summary_text = parsed.get("summary", "No summary available.")
        st.markdown(
            f'<div class="res-section res-summary">'
            f'<div class="res-label">Summary</div>'
            f'<div class="res-text">{summary_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        insights = parsed.get("key_insights", [])
        if insights:
            items = "".join(f"<li>{i}</li>" for i in insights)
            st.markdown(
                f'<div class="res-section res-insights">'
                f'<div class="res-label">Key Insights</div>'
                f'<ul class="res-list">{items}</ul>'
                f'</div>',
                unsafe_allow_html=True,
            )

        patterns = parsed.get("patterns", [])
        if patterns:
            items = "".join(f"<li>{p}</li>" for p in patterns)
            st.markdown(
                f'<div class="res-section res-patterns">'
                f'<div class="res-label">Patterns</div>'
                f'<ul class="res-list">{items}</ul>'
                f'</div>',
                unsafe_allow_html=True,
            )

        recs = parsed.get("recommendations", [])
        if recs:
            items = "".join(f"<li>{r}</li>" for r in recs)
            st.markdown(
                f'<div class="res-section res-recs">'
                f'<div class="res-label">Recommendations</div>'
                f'<ol class="res-list">{items}</ol>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with right_col:
        with st.expander("Full JSON response", expanded=False):
            st.json(parsed, expanded=True)

        if result.retrieved_context:
            st.markdown(
                '<div style="font-size:0.75rem;font-weight:700;letter-spacing:0.09em;'
                'text-transform:uppercase;color:var(--muted);margin:0.8rem 0 0.35rem;">Retrieved evidence</div>',
                unsafe_allow_html=True,
            )
            # Format evidence rows nicely
            lines = [l.strip() for l in result.retrieved_context.strip().split("\n") if l.strip()]
            formatted = "\n\n".join(lines)
            st.markdown(
                f'<div class="evidence-wrap">{formatted}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="card" style="margin-top:0.8rem;text-align:center;">'
                '<div class="small">RAG context was disabled for this run.</div>'
                '</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────
# Tab 5 — Evaluation
# ─────────────────────────────────────────
def render_evaluation() -> None:
    section_header(
        "Evaluation Dashboard",
        "Benchmark all model and prompt combinations on the standard dataset. Results persist across sessions.",
    )

    results_df = load_results()
    benchmark_agent = AnalystRAGAgent(dataset_path=DEFAULT_DATASET_PATH)
    pipeline = EvaluationPipeline(agent=benchmark_agent, questions_path=QUESTIONS_PATH, output_dir=OUTPUT_DIR)

    # Controls
    ctrl = st.columns([0.6, 0.8, 0.6, 1.2])
    with ctrl[0]:
        limit = st.number_input("Questions", min_value=10, max_value=100, value=20, step=10)
    with ctrl[1]:
        use_judge = st.checkbox("Use live LLM judge", value=False)
    with ctrl[3]:
        run_eval = st.button("Run benchmark", type="primary", use_container_width=True)

    if run_eval:
        progress = st.progress(0, text="Starting evaluation…")
        with st.spinner("Evaluating all model × prompt × RAG combinations…"):
            results_df = pipeline.run(limit=int(limit), enable_judge=use_judge)
        progress.progress(100, text="Done")
        st.success(f"Benchmark complete — {len(results_df)} rows saved.")

    if results_df is None:
        st.info("Click **Run benchmark** to generate evaluation results.")
        return

    agg = pipeline.aggregate_results(results_df)
    model_tbl = agg["model_comparison"]
    prompt_tbl = agg["prompt_comparison"]
    rag_tbl = agg["rag_comparison"]

    rag_lift = "N/A"
    score_col = "enhanced_overall_score" if "enhanced_overall_score" in model_tbl.columns else "overall_score"
    if True in rag_tbl.index and False in rag_tbl.index:
        rag_lift = f"{rag_tbl.loc[True, score_col] - rag_tbl.loc[False, score_col]:+.2f}"

    # Summary metrics
    m_cols = st.columns(4)
    with m_cols[0]: metric_card("Runs", str(len(results_df)), "Total evaluated rows")
    with m_cols[1]: metric_card("Best Model", str(model_tbl.index[0]), "Highest enhanced score")
    with m_cols[2]: metric_card("Best Prompt", str(prompt_tbl.index[0]), "Highest enhanced score")
    with m_cols[3]: metric_card("RAG Lift", rag_lift, "Score delta with vs without RAG")

    gap()

    # Score definitions
    with st.expander("What do the scores mean?", expanded=False):
        st.markdown(
            """
            | Metric | What it measures |
            |---|---|
            | `keyword_score` | Expected business keyword coverage in the answer |
            | `recommendation_score` | Presence and usefulness of action items |
            | `completeness_score` | How fully the required JSON fields are populated |
            | `groundedness_score` | Overlap between the answer and retrieved evidence |
            | `business_specificity_score` | Use of ecommerce-specific dimensions such as customer, channel, revenue, product, session, and conversion |
            | `json_validity_score` | Whether the raw model output is valid JSON and follows the required schema |
            | `retrieval_usefulness_score` | Whether retrieved context is meaningfully reflected in the answer |
            | `unique_insight_ratio` | How non-repetitive the insights, patterns, and recommendations are |
            | `avg_response_length` | Average word count across generated response sections |
            | `insight_count` | Count of generated insights, patterns, and recommendations |
            | `judge_usefulness` | LLM-as-judge rating from 1 to 5 for business decision value |
            | `judge_clarity` | LLM-as-judge rating from 1 to 5 for readability and structure |
            | `judge_correctness` | LLM-as-judge rating from 1 to 5 for support from dataset context |
            | `enhanced_overall_score` | Combined automatic + judge quality score |
            """
        )

    gap()
    score_cols = ["keyword_score", "recommendation_score", "completeness_score", "groundedness_score"]
    enhanced_cols = [
        "business_specificity_score", "json_validity_score",
        "retrieval_usefulness_score", "unique_insight_ratio",
        "enhanced_overall_score",
    ]
    judge_cols = ["judge_usefulness", "judge_clarity", "judge_correctness", "judge_average"]

    # ── LLM-as-judge ──────────────────────────
    if all(col in model_tbl.columns for col in judge_cols):
        st.markdown('<div class="eval-group">', unsafe_allow_html=True)
        st.markdown('<div class="eval-group-title">LLM-as-Judge Ratings</div>', unsafe_allow_html=True)
        plot_heatmap(model_tbl[judge_cols], "Usefulness, clarity, correctness by model")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── By model (stacked) ────────────────────
    st.markdown('<div class="eval-group">', unsafe_allow_html=True)
    st.markdown('<div class="eval-group-title">By Model</div>', unsafe_allow_html=True)
    plot_bar(model_tbl[score_col], "Enhanced overall score by model", horizontal=True)
    plot_heatmap(model_tbl[score_cols], "Metric breakdown by model")
    if all(col in model_tbl.columns for col in enhanced_cols):
        plot_heatmap(model_tbl[enhanced_cols], "Enhanced quality breakdown by model")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── By prompt style (stacked) ─────────────
    st.markdown('<div class="eval-group">', unsafe_allow_html=True)
    st.markdown('<div class="eval-group-title">By Prompt Style</div>', unsafe_allow_html=True)
    plot_bar(prompt_tbl[score_col], "Enhanced overall score by prompt style", horizontal=True)
    plot_heatmap(prompt_tbl[score_cols], "Metric breakdown by prompt style")
    if all(col in prompt_tbl.columns for col in enhanced_cols):
        plot_heatmap(prompt_tbl[enhanced_cols], "Enhanced quality breakdown by prompt style")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── RAG comparison ────────────────────────
    st.markdown('<div class="eval-group">', unsafe_allow_html=True)
    st.markdown('<div class="eval-group-title">RAG vs No-RAG</div>', unsafe_allow_html=True)
    st.dataframe(rag_tbl, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Raw rows ──────────────────────────────
    with st.expander("Raw benchmark rows", expanded=False):
        display_cols = [
            "question_id", "category", "model", "prompt_style", "rag_enabled",
            "keyword_score", "recommendation_score", "completeness_score",
            "groundedness_score", "business_specificity_score",
            "retrieval_usefulness_score", "judge_usefulness",
            "judge_clarity", "judge_correctness", score_col, "summary",
        ]
        st.dataframe(results_df[[col for col in display_cols if col in results_df.columns]].head(40), use_container_width=True)


# ─────────────────────────────────────────
# Tab 4 — Presentation
# ─────────────────────────────────────────
def render_presentation(dataset_path: str) -> None:
    section_header(
        "Presentation Generator",
        "One click to build a stakeholder-ready PowerPoint from the active dataset.",
    )

    # ── Action row ────────────────────────────
    act_left, act_right = st.columns([0.3, 0.7])
    with act_left:
        gen = st.button("Generate deck", type="primary", use_container_width=True)
    with act_right:
        ppt_path = st.session_state.get("ppt_path")
        if ppt_path and Path(ppt_path).exists():
            with open(ppt_path, "rb") as f:
                st.download_button(
                    "Download PowerPoint",
                    data=f.read(),
                    file_name="presentation.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True,
                )
        else:
            st.markdown('<div class="small" style="margin-top:0.65rem;">Generate the deck first, then download it here.</div>', unsafe_allow_html=True)

    if gen:
        presenter = build_presenter()
        bar = st.progress(0, text="Generating charts…")
        with st.spinner("Building presentation from the active dataset…"):
            ppt_path, slides, chart_paths = presenter.create_presentation(
                dataset_path=dataset_path,
                output_path=OUTPUT_DIR / "presentation.pptx",
            )
        bar.progress(100, text="Done")
        st.session_state["ppt_path"] = ppt_path
        st.session_state["slides"] = slides
        st.session_state["chart_paths"] = chart_paths
        st.success(f"Deck ready — {len(slides)} slides generated.")
        rerun()

    slides = st.session_state.get("slides", [])
    chart_paths = st.session_state.get("chart_paths", {})

    if not slides:
        # Empty state
        st.markdown(
            '<div style="text-align:center;padding:3rem 1rem;">'
            '<div style="font-size:2.5rem;margin-bottom:0.6rem;">&#128202;</div>'
            '<div style="font-weight:700;color:#173746;font-size:1rem;">No presentation yet</div>'
            '<div class="small" style="margin-top:0.4rem;">Click <strong>Generate deck</strong> above to build slides from the active dataset.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    gap()
    st.markdown(
        f'<div style="font-size:0.78rem;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;color:var(--muted);margin-bottom:0.6rem;">'
        f'{len(slides)} slides</div>',
        unsafe_allow_html=True,
    )

    for idx, slide in enumerate(slides, start=1):
        img_col, text_col = st.columns([0.95, 1.05])

        with img_col:
            if slide.chart_key and slide.chart_key in chart_paths:
                st.image(str(chart_paths[slide.chart_key]), use_column_width=True)
            else:
                st.markdown(
                    '<div class="card" style="aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;">'
                    '<div class="small">No chart</div></div>',
                    unsafe_allow_html=True,
                )

        with text_col:
            bullets_html = "".join(f"<li>{b}</li>" for b in slide.bullets)
            st.markdown(
                f'<div class="slide-card">'
                f'<div class="slide-header">'
                f'<div class="slide-num">Slide {idx} of {len(slides)}</div>'
                f'<div class="slide-title">{slide.title}</div>'
                f'</div>'
                f'<div class="slide-body">'
                f'<ul class="slide-bullets">{bullets_html}</ul>'
                f'<div class="notes-box">'
                f'<div class="notes-label">Speaker notes</div>'
                f'{slide.speaker_notes}'
                f'</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if idx < len(slides):
            st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)

    gap()
    if ppt_path and Path(ppt_path).exists():
        with open(ppt_path, "rb") as f:
            st.download_button(
                "Download PowerPoint",
                data=f.read(),
                file_name="presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Ecommerce Analytics Studio",
        page_icon="📊",
        layout="wide",
    )

    inject_styles()
    ensure_defaults()

    dataset, dataset_name, dataset_path, agent = get_active()
    render_sidebar(dataset, dataset_name)
    render_topbar(dataset_name, dataset)

    t_overview, t_eda, t_analyst, t_pres, t_eval = st.tabs(
        ["Overview", "EDA", "Analyst", "Presentation", "Evaluation"]
    )

    with t_overview:
        render_overview(dataset, dataset_name)

    with t_eda:
        render_eda(dataset_path)

    with t_analyst:
        render_analyst(agent)

    with t_pres:
        render_presentation(dataset_path)

    with t_eval:
        render_evaluation()


if __name__ == "__main__":
    main()
