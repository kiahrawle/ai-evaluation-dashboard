"""Streamlit dashboard for real-time hallucination & risk evaluation.

Data-bearing tabs read real evaluation output from results/results.json (written
by `analyze.summarize`). When no results exist yet, each tab shows an honest
empty state rather than mock numbers. The Live Evaluation and Upload tabs run
real local computation (topic detection, risk markers, RAG grounding).

The look-and-feel is a custom design system (gradient/glass metric cards, neon
accent palette, Poppins/Inter type, rounded corners, hover + transition motion)
applied purely through injected CSS and a shared Plotly theme, so none of the
evaluation logic below is coupled to presentation.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Bridge an API key from Streamlit Secrets into the environment IF one is set, so
# a hosted build never needs a committed .env (no-op when no secrets exist). Pair
# this with a hard spend cap on your Anthropic account before exposing any
# live-generation feature publicly.
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ.setdefault("ANTHROPIC_API_KEY", st.secrets["ANTHROPIC_API_KEY"])
except Exception:
    pass


def _demo_enabled() -> bool:
    """Cache-only public demo: only offline tabs are shown, so visitors can never
    trigger model loading or spend an API key. Enable with env
    FACTUAL_EVAL_DEMO=1 or a `demo_mode = true` Streamlit secret."""
    if os.environ.get("FACTUAL_EVAL_DEMO", "").strip().lower() in ("1", "true", "yes"):
        return True
    try:
        return bool(st.secrets.get("demo_mode", False))
    except Exception:
        return False


DEMO_MODE = _demo_enabled()

# `reporting` is light (pandas/json only). Heavy ML imports
# (sentence-transformers/torch, FAISS) are deferred into the tabs that actually
# need them, so the cache-only tabs boot fast and within free-tier memory.
from src import reporting

st.set_page_config(
    page_title="LLM Hallucination Intelligence Platform",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
# Palette: a vibrant, neon-accented light theme. Each token is referenced by the
# injected CSS *and* the shared Plotly template so charts and chrome stay in sync.
PRIMARY = "#0066FF"   # electric blue
ACCENT_PINK = "#FF3366"
ACCENT_LIME = "#00D26A"
ACCENT_PURPLE = "#7C3AED"
BG = "#F5F7FA"
INK = "#1F2937"
MUTED = "#6B7280"

RISK_COLORS = {"LOW": ACCENT_LIME, "MEDIUM": "#FFA15A", "HIGH": ACCENT_PINK}
CHART_SEQUENCE = [PRIMARY, ACCENT_PINK, ACCENT_LIME, ACCENT_PURPLE, "#FFA15A", "#06B6D4"]


def _inject_theme() -> None:
    """Inject the full CSS design system once per page render."""
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {{
                --primary: {PRIMARY};
                --pink: {ACCENT_PINK};
                --lime: {ACCENT_LIME};
                --purple: {ACCENT_PURPLE};
                --ink: {INK};
                --muted: {MUTED};
                --radius: 20px;
                --shadow: 0 10px 30px rgba(31, 41, 55, 0.08);
                --shadow-hover: 0 18px 45px rgba(31, 41, 55, 0.16);
            }}

            html, body, [class*="css"], .stApp, [data-testid="stMarkdownContainer"] {{
                font-family: 'Inter', 'Segoe UI', sans-serif;
                color: var(--ink);
            }}

            .stApp {{
                background:
                    radial-gradient(1200px 600px at 100% -10%, rgba(124,58,237,0.08), transparent 60%),
                    radial-gradient(1000px 500px at -10% 0%, rgba(0,102,255,0.08), transparent 55%),
                    {BG};
            }}

            .block-container {{ padding-top: 2rem; padding-bottom: 4rem; max-width: 1400px; }}

            h1, h2, h3, h4 {{ font-family: 'Poppins', 'Inter', sans-serif; letter-spacing: -0.02em; }}

            /* ---- Hero header ---- */
            .hero {{
                background: linear-gradient(120deg, var(--primary), var(--purple) 55%, var(--pink));
                border-radius: 24px;
                padding: 30px 36px;
                color: #fff;
                box-shadow: var(--shadow);
                margin-bottom: 26px;
            }}
            .hero h1 {{ color: #fff; font-size: 2.1rem; margin: 0 0 6px 0; }}
            .hero p {{ color: rgba(255,255,255,0.92); margin: 0; font-size: 1.02rem; }}
            .hero .pills {{ margin-top: 16px; display: flex; gap: 10px; flex-wrap: wrap; }}
            .pill {{
                background: rgba(255,255,255,0.18);
                border: 1px solid rgba(255,255,255,0.30);
                color: #fff; font-size: 0.78rem; font-weight: 600;
                padding: 5px 12px; border-radius: 999px; backdrop-filter: blur(6px);
            }}

            /* ---- Metric cards (glassmorphism) ---- */
            [data-testid="stMetric"] {{
                background: rgba(255,255,255,0.75);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.6);
                border-radius: var(--radius);
                padding: 18px 20px 16px 20px;
                box-shadow: var(--shadow);
                transition: transform .25s ease, box-shadow .25s ease;
                overflow: hidden; position: relative;
            }}
            [data-testid="stMetric"]::before {{
                content: ""; position: absolute; top: 0; left: 0; right: 0; height: 5px;
                background: var(--primary);
            }}
            [data-testid="stMetric"]:hover {{ transform: translateY(-4px); box-shadow: var(--shadow-hover); }}
            [data-testid="stMetricLabel"] p {{ color: var(--muted); font-weight: 600; font-size: 0.82rem; }}
            [data-testid="stMetricValue"] {{ font-family: 'Poppins', sans-serif; font-weight: 700; color: var(--ink); }}

            /* Rotate the accent bar across cards in a row */
            [data-testid="stHorizontalBlock"] > div:nth-child(4n+1) [data-testid="stMetric"]::before {{ background: var(--primary); }}
            [data-testid="stHorizontalBlock"] > div:nth-child(4n+2) [data-testid="stMetric"]::before {{ background: var(--pink); }}
            [data-testid="stHorizontalBlock"] > div:nth-child(4n+3) [data-testid="stMetric"]::before {{ background: var(--lime); }}
            [data-testid="stHorizontalBlock"] > div:nth-child(4n+4) [data-testid="stMetric"]::before {{ background: var(--purple); }}

            /* ---- Buttons (purple→blue gradient CTA) ---- */
            .stButton > button {{
                background: linear-gradient(120deg, var(--purple), var(--primary));
                color: #fff; border: none; border-radius: 14px;
                padding: 0.6rem 1.2rem; font-weight: 600; font-family: 'Inter', sans-serif;
                box-shadow: 0 8px 20px rgba(124,58,237,0.28);
                transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
            }}
            .stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 14px 30px rgba(124,58,237,0.38); filter: brightness(1.05); }}
            .stButton > button:active {{ transform: translateY(0); }}

            /* ---- Inputs ---- */
            .stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {{
                border-radius: 12px !important; border: 1px solid #E2E8F0 !important;
                background: rgba(255,255,255,0.9) !important;
            }}
            .stTextInput input:focus, .stTextArea textarea:focus {{
                border-color: var(--primary) !important; box-shadow: 0 0 0 3px rgba(0,102,255,0.15) !important;
            }}

            /* ---- Sidebar nav ---- */
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #ffffff, #f3f5fb);
                border-right: 1px solid rgba(31,41,55,0.06);
            }}
            [data-testid="stSidebar"] [role="radiogroup"] label {{
                border-radius: 12px; padding: 8px 12px; margin: 2px 0;
                transition: background .2s ease, transform .15s ease;
            }}
            [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
                background: rgba(0,102,255,0.08); transform: translateX(3px);
            }}

            /* ---- Tables, alerts, dividers ---- */
            [data-testid="stDataFrame"] {{ border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow); }}
            [data-testid="stAlert"] {{ border-radius: 14px; border: none; box-shadow: var(--shadow); }}
            hr {{ border-color: rgba(31,41,55,0.08); }}

            /* Plotly chart cards */
            [data-testid="stPlotlyChart"] {{
                background: rgba(255,255,255,0.7); border-radius: var(--radius);
                padding: 8px; box-shadow: var(--shadow); border: 1px solid rgba(255,255,255,0.6);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _style_fig(fig: go.Figure) -> go.Figure:
    """Apply the shared chart theme: transparent card background, brand font and
    the neon color sequence. Keeps every Plotly figure visually consistent."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=INK, size=13),
        colorway=CHART_SEQUENCE,
        margin=dict(t=40, r=20, b=20, l=20),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title_font=dict(family="Poppins, Inter, sans-serif"),
    )
    fig.update_xaxes(gridcolor="rgba(31,41,55,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(31,41,55,0.06)", zeroline=False)
    return fig


_inject_theme()

# ---- Hero header ----
st.markdown(
    """
    <div class="hero">
        <h1>🧠 LLM Hallucination Intelligence Platform</h1>
        <p>Real-time evaluation, risk detection, and adaptive safety for LLM responses.</p>
        <div class="pills">
            <span class="pill">⚡ Live risk scoring</span>
            <span class="pill">📑 RAG grounding</span>
            <span class="pill">🛡️ Adaptive guardrails</span>
            <span class="pill">🏆 Model leaderboard</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar navigation
st.sidebar.markdown("### 🧭 Navigation")
tab = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Live Evaluation", "Upload & Analyze", "Leaderboard", "Settings"],
    label_visibility="collapsed",
)


def _no_data_notice():
    st.info(
        "No evaluation results yet. Run an evaluation to populate this view:\n\n"
        "```\npython main.py evaluate --models claude-haiku-4-5-20251001 --limit 20\n```"
    )


if tab == "Dashboard":
    st.header("Overview Dashboard")
    results = reporting.load_results()

    if not results:
        _no_data_notice()
    else:
        m = reporting.overview_metrics(results)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Evaluations", f"{m['total']:,}")
        col2.metric("Hallucination Rate", f"{m['hallucination_rate']:.1%}")
        col3.metric("Avg Risk Score", f"{m['avg_risk_score']:.2f}")
        col4.metric("Citation Coverage", f"{m['citation_coverage']:.1%}")

        col5, col6 = st.columns(2)
        col5.metric("Confident (HIGH-danger) hallucinations", m["dangerous_high"])
        col6.metric("Numeric contradictions vs. docs", m["contradictions"])

        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Risk Level Distribution")
            dist = reporting.risk_distribution(results)
            if sum(dist.values()) == 0:
                st.caption("No risk-level data in results.")
            else:
                risk_df = pd.DataFrame({"Risk Level": list(dist.keys()),
                                        "Count": list(dist.values())})
                fig = px.pie(risk_df, values="Count", names="Risk Level",
                             color="Risk Level", color_discrete_map=RISK_COLORS, hole=0.55)
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(_style_fig(fig), width='stretch')

        with col_right:
            st.subheader("Hallucination by Category")
            by_cat = reporting.hallucination_by_category(results)
            if not by_cat:
                st.caption("No category data in results.")
            else:
                cat_df = (pd.DataFrame({"Category": list(by_cat.keys()),
                                        "Hallucination Rate": list(by_cat.values())})
                          .sort_values("Hallucination Rate", ascending=False))
                fig = px.bar(cat_df, x="Category", y="Hallucination Rate",
                             color="Hallucination Rate", color_continuous_scale="RdYlGn_r")
                st.plotly_chart(_style_fig(fig), width='stretch')


elif tab == "Live Evaluation":
    # Lazy import: loads sentence-transformers/torch only when this tab is opened.
    from src.utils import detect_topic, scan_for_risk_markers
    from src.guardrails import adaptive_guardrails_summary
    from src.evaluators import score_refusal_quality
    from src.risk import stream_risk_from_markers

    st.header("⚡ Live Evaluation")

    col1, col2 = st.columns([2, 1])
    with col1:
        question = st.text_area("Question:", placeholder="Enter a question...", height=100)
    with col2:
        response = st.text_area("Response:", placeholder="Enter LLM response...", height=100)

    if st.button("Evaluate Response", width='stretch'):
        if not (question and response):
            st.warning("Enter both a question and a response.")
        else:
            with st.spinner("Analyzing..."):
                topic_info = detect_topic(question)
                topic = topic_info["topic"]
                settings = adaptive_guardrails_summary(topic)
                refusal_qual = score_refusal_quality(response)
                markers = scan_for_risk_markers(response)
                live_risk = stream_risk_from_markers(markers)
                risk_level = "HIGH" if live_risk >= 0.67 else "MEDIUM" if live_risk >= 0.33 else "LOW"

            st.success("Evaluation Complete!")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Detected Topic", topic.upper(),
                          help=f"Confidence: {topic_info['confidence']:.1%}")
                st.metric("Enforcement Level", settings["enforcement_level"].upper())
            with c2:
                st.metric("Live Risk", f"{live_risk:.2f}", risk_level)
                st.metric("Overall Refusal Quality", f"{refusal_qual['overall_quality']:.1%}")
            with c3:
                if refusal_qual["is_refusal"]:
                    st.info("🚫 This is a refusal.")
                    st.metric("Helpfulness", f"{refusal_qual.get('helpfulness') or 0:.1%}")
                else:
                    st.metric("Is Refusal", "No")

            st.divider()
            st.subheader("Risk markers detected")
            flagged = {
                "Uncertainty": markers["uncertainty"]["uncertainty_phrases"],
                "Risky phrases": markers["risky_phrases"]["risky_phrases"],
                "Contradictions": markers["contradictions"]["contradiction_markers"],
                "Citation gaps": markers["citation_gaps"]["citation_phrases_found"],
            }
            any_flag = False
            for label, items in flagged.items():
                if items:
                    any_flag = True
                    st.write(f"**{label}:** {', '.join(items)}")
            if not any_flag:
                st.caption("No risk markers detected in the response.")

            st.divider()
            st.subheader("Detailed Analysis")
            d1, d2 = st.columns(2)
            with d1:
                st.write(f"**Explanation Quality:** {refusal_qual['explanation_quality']:.1%}")
                st.write(f"**Educational Value:** {refusal_qual['educational_value']:.1%}")
            with d2:
                st.write(f"**Requires Citations:** {settings['require_citations']}")
                st.write(f"**Strict Checking:** {settings['strict_checking']}")


elif tab == "Upload & Analyze":
    # Lazy import: FAISS + sentence-transformers load only when this tab is used.
    from src import rag
    from src.evaluators import (
        verify_citation_support, detect_numeric_contradictions, score_groundedness,
    )

    st.header("📁 Upload & Analyze")
    st.markdown("Upload a knowledge source, build a vector index, then check whether "
                "an answer is actually **grounded** in it.")

    uploaded_file = st.file_uploader("Upload a document (TXT, Markdown, or PDF)",
                                     type=["pdf", "txt", "md"])

    if uploaded_file:
        st.info(f"📄 Uploaded: {uploaded_file.name}")
        if st.button("Build index from document"):
            data = uploaded_file.getvalue()
            try:
                docs = rag.ingest_document(uploaded_file.name, data)
            except RuntimeError as e:
                docs = []
                st.error(str(e))
            if docs:
                ok = rag.initialize_index(docs)
                st.session_state["rag_ready"] = ok
                st.session_state["rag_chunks"] = len(docs)
                if not ok:
                    st.error("Failed to build FAISS index (is faiss-cpu installed?).")

    if st.session_state.get("rag_ready"):
        st.success(f"✅ Index ready: {st.session_state.get('rag_chunks', 0)} chunks.")
        top_k = st.slider("Retrieval Top-K", 1, 10, 3)
        q = st.text_input("Question (what to retrieve evidence for):")
        a = st.text_area("Answer to fact-check against the document:", height=100)

        if st.button("Evaluate grounding"):
            if not (q and a):
                st.warning("Enter both a question and an answer.")
            else:
                docs = rag.retrieve(q, top_k=top_k)
                citation = verify_citation_support(a, docs)
                contradiction = detect_numeric_contradictions(a, docs)
                ground = score_groundedness(docs, a)

                c1, c2, c3 = st.columns(3)
                c1.metric("Groundedness", f"{ground:.2f}")
                c2.metric("Citation Support", f"{citation['support_score']:.1%}")
                c3.metric("Supported?", "Yes" if citation["citation_supported"] else "No")

                if contradiction["contradiction_detected"]:
                    st.error("⚠️ Numeric contradiction vs. the document:")
                    for c in contradiction["contradictions"]:
                        st.write(f"- **{', '.join(c['context'])}**: "
                                 f"answer says `{c['answer_value']}`, "
                                 f"document says `{c['doc_value']}`")

                st.subheader("Retrieved evidence per claim")
                for ev in citation.get("evidence", []):
                    icon = "✅" if ev["supported"] else "❌"
                    st.write(f"{icon} *{ev['claim']}* (sim {ev['similarity']:.2f})")
                    if ev["supported"] and ev["evidence_text"]:
                        st.caption(f"↳ {ev['evidence_text']}")
    else:
        st.caption("Build an index above to enable grounding checks.")


elif tab == "Leaderboard":
    st.header("🏆 Model Leaderboard")
    results = reporting.load_results()
    rows = reporting.leaderboard_from_results(results)

    if rows:
        st.caption("Live data from results/results.json")
        lb = pd.DataFrame(rows)[
            ["rank", "model", "hallucination_rate", "groundedness",
             "avg_risk_score", "citation_coverage", "evaluations"]
        ]
        st.dataframe(lb, width='stretch', hide_index=True)

        st.markdown("---")
        st.subheader("🔍 Model Comparison")
        models_to_compare = st.multiselect(
            "Select models to compare:", lb["model"].tolist(),
            default=lb["model"].tolist()[: min(2, len(lb))],
        )
        if models_to_compare:
            comp = lb[lb["model"].isin(models_to_compare)]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=comp["model"], y=comp["hallucination_rate"],
                                 name="Hallucination Rate", marker_color=ACCENT_PINK))
            fig.add_trace(go.Bar(x=comp["model"], y=comp["groundedness"],
                                 name="Groundedness", marker_color=PRIMARY))
            fig.update_layout(barmode="group", xaxis_title="Model", yaxis_title="Score")
            st.plotly_chart(_style_fig(fig), width='stretch')
    else:
        _no_data_notice()
        st.caption("The leaderboard populates automatically once you evaluate one "
                   "or more models.")


elif tab == "Settings":
    st.header("⚙️ Settings & Configuration")
    st.caption("These controls are illustrative; edit config/config.yaml to change "
               "real run behavior.")

    st.subheader("RAG Configuration")
    st.toggle("Enable RAG for groundedness checks", value=True)
    st.slider("Retrieval Top-K", 1, 10, 5)

    st.subheader("Risk Engine Settings")
    st.slider("Hallucination Weight", 0.0, 1.0, 0.4)
    st.slider("Unsupported Claims Weight", 0.0, 1.0, 0.3)

    st.subheader("Adaptive Guardrails")
    st.multiselect("Strict enforcement topics:",
                   ["medical", "finance", "legal", "science"],
                   default=["medical", "finance", "legal"])

    st.subheader("Judge Configuration")
    st.selectbox("Primary Judge Model", ["claude-opus-4-8", "gpt-4", "claude-sonnet"])
    st.toggle("Enable Multi-Judge Verification", value=True)


st.markdown("---")
st.markdown(
    "By Kiah Rawle &nbsp;|&nbsp; "
    "[GitHub](https://github.com/kiahrawle/ai-evaluation-dashboard)"
)
