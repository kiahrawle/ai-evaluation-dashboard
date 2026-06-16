"""Streamlit dashboard for real-time hallucination & risk evaluation.

Data-bearing tabs read real evaluation output from results/results.json (written
by `analyze.summarize`). When no results exist yet, each tab shows an honest
empty state rather than mock numbers. The Live Evaluation and Upload tabs run
real local computation (topic detection, risk markers, RAG grounding).
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
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ LLM Hallucination Intelligence Platform")
st.markdown("Real-time evaluation, risk detection, and adaptive safety for LLM responses.")

RISK_COLORS = {"LOW": "#00CC96", "MEDIUM": "#FFA15A", "HIGH": "#FF6B6B"}

# Sidebar navigation
tab = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Live Evaluation", "Upload & Analyze", "Leaderboard", "Settings"],
)


def _no_data_notice():
    st.info(
        "No evaluation results yet. Run an evaluation to populate this view:\n\n"
        "```\npython main.py evaluate --models claude-haiku-4-5-20251001 --limit 20\n```"
    )


if tab == "Dashboard":
    st.header("📊 Overview Dashboard")
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
                             color="Risk Level", color_discrete_map=RISK_COLORS)
                st.plotly_chart(fig, width='stretch')

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
                st.plotly_chart(fig, width='stretch')


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
                                 name="Hallucination Rate"))
            fig.add_trace(go.Bar(x=comp["model"], y=comp["groundedness"], name="Groundedness"))
            fig.update_layout(barmode="group", xaxis_title="Model", yaxis_title="Score")
            st.plotly_chart(fig, width='stretch')
    else:
        _no_data_notice()
        st.caption("The leaderboard populates automatically once you evaluate one "
                   "or more models.")


elif tab == "Run locally":
    st.header("🖥️ Run locally")
    st.markdown(
        "This hosted demo is **cache-only**: the Dashboard and Leaderboard show "
        "real results committed to the repo — no API key, no model loading, no "
        "way for a visitor to spend anyone's credits.\n\n"
        "The interactive features (Live Evaluation, Upload→RAG grounding, the "
        "real-time risk monitor, and full model evaluations) run locally:"
    )
    st.code(
        "git clone <repo> && cd factual-eval\n"
        "python -m venv .venv && . .venv/Scripts/activate\n"
        "pip install -r requirements.txt\n"
        "cp .env.example .env        # add your ANTHROPIC_API_KEY\n"
        "streamlit run app/app.py    # full UI (all tabs)\n"
        'python main.py monitor --question "..."   # live risk monitor',
        language="bash",
    )
    st.caption("Set FACTUAL_EVAL_DEMO=1 to run the dashboard in this cache-only "
               "mode; unset it for the full local UI.")


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
    "By Kiah Rawle | "
    "[GitHub](https://github.com/kiahrawle/ai-evaluation-dashboard) | "
    
)

