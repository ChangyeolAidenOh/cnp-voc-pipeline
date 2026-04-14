import ast
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

## Colors setting
LG_RED    = "#A50034"
LG_DARK   = "#1A1A1A"
LG_GRAY   = "#666666"
POSITIVE  = "#00A86B"
NEUTRAL   = "#4A90D9"

## Configuration
st.set_page_config(
    page_title="CNP VoC Causal Signal Detection Pipeline",
    page_icon="🔬",
    layout="wide"
)

# Load processed dataset
@st.cache_data
def load_data():
    """Load all processed output files for dashboard rendering."""
    processed  = pd.read_csv("voc_pipeline/data/processed/cnp_processed.csv")
    causal     = pd.read_csv("voc_pipeline/data/processed/cnp_causal_signals.csv")
    temporal   = pd.read_csv("voc_pipeline/data/processed/cnp_temporal_signals.csv")
    lda        = pd.read_csv("voc_pipeline/data/processed/cnp_lda_results.csv")
    stream_a   = pd.read_csv("voc_pipeline/data/processed/stream_a_findings.csv")
    divergence = pd.read_csv("voc_pipeline/data/processed/stream_b_keyword_divergence.csv")
    bertopic   = pd.read_csv("voc_pipeline/data/processed/cnp_bertopic_results.csv")
    consensus  = pd.read_csv("voc_pipeline/data/processed/cnp_lda_bertopic_consensus.csv")
    return processed, causal, temporal, lda, stream_a, divergence, bertopic, consensus

processed, causal, temporal, lda, stream_a, divergence, bertopic, consensus = load_data()


def parse_list(val):
    """Parse a stringified Python list back to a list object."""
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else []
    except:
        return []

# clean overlap_keywords: remove brackets and quotes
def format_keywords(val):
    try:
        items = ast.literal_eval(str(val))
        if isinstance(items, list):
            return ", ".join(items)
        return str(val)
    except:
        return str(val)


# Header
st.title(" CNP VoC Causal Signal Detection ")
st.markdown(
    "Aiden Changyeol Oh"
)
st.divider()

# inject CSS for centered dataframe tables
st.markdown("""
    <style>
    [data-testid="stDataFrame"] {
        margin-left: auto;
        margin-right: auto;
    }
    [data-testid="stDataFrame"] table {
        margin-left: auto;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)


# Tab layout setting
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Overview",
    "️ LDA Topics",
    "️ Causal Signals",
    " Comparison CNP with ANUA",
    " BERTopic"
])


## Tab 1 — Overview
with tab1:
    st.header("Project Overview")

    # key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Collected", "6,522 docs")
    with col2:
        st.metric("After CNP Filter", f"{len(causal):,} docs")
    with col3:
        churn_rate = (causal["signal_type"] == "이탈위험").mean() * 100
        st.metric("Churn Risk Rate", f"{churn_rate:.1f}%")
    with col4:
        st.metric("ANUA Mention Docs", f"{int(causal['anua_mentioned'].sum())}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Distribution")
        source_counts = causal["source"].value_counts().reset_index()
        source_counts.columns = ["Source", "Count"]
        fig = px.pie(
            source_counts, values="Count", names="Source",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Signal Type Distribution")
        signal_counts = causal["signal_type"].value_counts().reset_index()
        signal_counts.columns = ["Signal", "Count"]
        fig = px.bar(
            signal_counts, x="Signal", y="Count",
            color="Signal",
            color_discrete_map={
                "이탈위험": LG_RED,
                "긍정": POSITIVE,
                "중립": NEUTRAL,
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pipeline Architecture")
    st.code(
        "Data Collection  → Naver Blog/Cafe (996) + YouTube Comments (2,500)\n"
        "Preprocessing    → kiwipiepy morphological analysis + 4 modes\n"
        "                   (unigram / bigram / unibi_mix / adj_noun)\n"
        "LDA Analysis     → per-source × per-mode coherence comparison\n"
        "                   (best: 0.6256, blog unibi_mix)\n"
        "Causal Detection → Causal Signal Detector (churn risk: 27.9%)\n"
        "Temporal         → monthly churn / positive signal trend\n"
        "Stream A         → ANUA Findings 3/4/5 validated on CNP data\n"
        "Stream B         → EN→KO translation + keyword divergence\n"
        "BERTopic         → LDA × BERTopic ensemble (14 topics)",
        language="text"
    )


## Tab 2 — LDA Topics
with tab2:
    st.header("LDA Topic Modeling Results")

    # coherence heatmap
    st.subheader("Coherence Score by Source × Mode")
    coherence_data = lda.groupby(["scope", "mode"])["coherence"].first().reset_index()
    coherence_pivot = coherence_data.pivot(
        index="scope", columns="mode", values="coherence"
    )
    fig = px.imshow(
        coherence_pivot,
        text_auto=".3f",
        color_continuous_scale=["#FFFFFF", "#F5C0CC", "#D4005A", LG_RED],
        title="c_v Coherence Score Heatmap"
    )
    fig.update_traces(textfont=dict(color="black", size=13))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # interactive topic keyword explorer
    st.subheader("Topic Keyword Explorer")

    # center-aligned selectboxes
    _, col1, col2, _ = st.columns([0.5, 2, 2, 0.5])
    with col1:
        scope_options = lda["scope"].unique().tolist()
        selected_scope = st.selectbox("Select Source", scope_options)
    with col2:
        mode_options = lda[lda["scope"] == selected_scope]["mode"].unique().tolist()
        selected_mode = st.selectbox("Select Mode", mode_options)

    filtered = lda[
        (lda["scope"] == selected_scope) &
        (lda["mode"] == selected_mode)
        ]

    if len(filtered) > 0:
        coherence_val = filtered["coherence"].iloc[0]
        optimal_k = filtered["optimal_k"].iloc[0]
        # high-contrast info box
        st.markdown(
            f"<div style='background-color:{LG_RED}; padding:12px; border-radius:6px;"
            f"color:white; font-weight:bold; font-size:16px;'>"
            f"Optimal k = {optimal_k} &nbsp;|&nbsp; Coherence = {coherence_val:.4f}"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown("")
        for _, row in filtered.iterrows():
            # dark-background-safe topic display
            st.markdown(
                f"<div style='background-color:#2B2B2B; padding:8px 12px;"
                f"border-left:3px solid {LG_RED}; margin-bottom:6px;"
                f"border-radius:4px; color:#FFFFFF; font-family:monospace;'>"
                f"<b>Topic {row['topic_id']}</b>: {row['keywords']}"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.warning("No data available")



## Tab 3 — Causal Signals
with tab3:
    st.header("Causal Signal Detection Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Risk Rate by Source")
        source_signal = causal.groupby("source").apply(
            lambda x: pd.Series({
                "Churn Risk": (x["signal_type"] == "이탈위험").mean() * 100,
                "Positive":   (x["signal_type"] == "긍정").mean() * 100,
                "Neutral":    (x["signal_type"] == "중립").mean() * 100,
            })
        ).reset_index()
        source_signal_melted = source_signal.melt(
            id_vars="source", var_name="Signal", value_name="Rate (%)"
        )
        fig = px.bar(
            source_signal_melted, x="source", y="Rate (%)",
            color="Signal", barmode="stack",
            color_discrete_map={
                "Churn Risk": LG_RED,
                "Positive":   POSITIVE,
                "Neutral":    NEUTRAL
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("ANUA Mention vs Overall Churn Risk")
        overall = (causal["signal_type"] == "이탈위험").mean() * 100
        anua_churn = (
            causal[causal["anua_mentioned"] == True]["signal_type"] == "이탈위험"
        ).mean() * 100
        fig = go.Figure(go.Bar(
            x=["Overall CNP", "ANUA-mention Docs"],
            y=[overall, anua_churn],
            marker_color=[LG_RED, NEUTRAL],
            text=[f"{overall:.1f}%", f"{anua_churn:.1f}%"],
            textposition="outside"
        ))
        fig.update_layout(
            yaxis_title="Churn Risk Rate (%)",
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Monthly Churn / Positive Signal Trend")
    temporal["month"] = temporal["month"].astype(str)
    temporal_recent = temporal[temporal["month"] >= "2025-01"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=temporal_recent["month"],
        y=temporal_recent["churn_rate"] * 100,
        name="Churn Risk",
        line=dict(color=LG_RED, width=2),
        mode="lines+markers"
    ))
    fig.add_trace(go.Scatter(
        x=temporal_recent["month"],
        y=temporal_recent["positive_rate"] * 100,
        name="Positive",
        line=dict(color=POSITIVE, width=2),
        mode="lines+markers"
    ))
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Rate (%)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Churn risk spiked to 54.3% in Jan–Feb 2026, "
        "followed by a recovery trend through April 2026."
    )

## Tab 4 — ANUA Comparison
with tab4:
    st.header("ANUA vs CNP Comparative Analysis")

    # Stream A: ANUA findings validation
    st.subheader("Stream A — ANUA Findings Validated on CNP Data")
    fig = px.bar(
        stream_a,
        x="finding",
        y=["rate", "churn_rate"],
        barmode="group",
        labels={"value": "Rate (%)", "variable": "Metric"},
        # LG red for churn, neutral blue for detection rate
        color_discrete_map={"rate": NEUTRAL, "churn_rate": LG_RED},
        title="Detection Rate vs Churn Rate per ANUA Finding"
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Finding 5: Trouble-prone Churn",
            "67.3%",
            "+38.5%p vs moisture-focused"
        )
    with col2:
        st.metric(
            "Finding 4: Channel-driven Positive",
            "33.3%",
            "+16.6%p vs non-channel"
        )
    with col3:
        st.metric(
            "ANUA Mention → Churn Risk",
            "81.0%",
            "+53.2%p vs overall"
        )

    st.divider()


    # Stream B: keyword divergence
    #### bias=1.0 for all ===> more informative
    st.subheader("Stream B — Brand-specific Keyword Divergence")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**CNP-specific Keywords**")
        cnp_spec = divergence[divergence["brand"] == "CNP 특화"].head(10)
        fig = px.bar(
            cnp_spec, x="cnp_count", y="keyword",
            orientation="h",
            color_discrete_sequence=[LG_RED],
            title="CNP-specific — keyword frequency",
            labels={"cnp_count": "CNP mention count"}
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**ANUA-specific Keywords**")
        anua_spec = divergence[divergence["brand"] == "ANUA 특화"].head(10)
        fig = px.bar(
            anua_spec, x="anua_count", y="keyword",
            orientation="h",
            color_discrete_sequence=[NEUTRAL],
            title="ANUA-specific — keyword frequency",
            labels={"anua_count": "ANUA mention count"}
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)


## Tab 5 — BERTopic
with tab5:
    st.header("BERTopic Analysis Results")

    # BERTopic distribution — centered with padding, increased height
    st.subheader("Document Count per Topic")
    _, col_center, _ = st.columns([0.1, 9.8, 0.1])
    with col_center:
        bertopic_sorted = bertopic.sort_values("Count", ascending=False)
        fig = px.bar(
            bertopic_sorted,
            x="Count", y="Name",
            orientation="h",
            color="Count",
            color_continuous_scale=["#F5C0CC", "#D4005A", LG_RED],
            title="BERTopic Topic Distribution"
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            height=600,          # increased height for readability
            yaxis_tickfont=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # LDA × BERTopic consensus table — centered, cleaned
    st.subheader("LDA × BERTopic Consensus")
    _, col_table, _ = st.columns([0.5, 9, 0.5])
    with col_table:
        # clean overlap_keywords: remove brackets
        consensus_display = consensus[
            ["lda_topic", "bertopic_id", "overlap_keywords", "confidence"]
        ].copy()
        consensus_display["overlap_keywords"] = consensus_display[
            "overlap_keywords"
        ].apply(format_keywords)
        # capitalize High / Low
        consensus_display["confidence"] = consensus_display["confidence"].str.capitalize()
        st.dataframe(consensus_display, use_container_width=True, hide_index=True)
        st.caption(
            "**High**: both models agree → high-confidence signal  \n"
            "**Low**: LDA-only signal → requires further validation"
        )

    st.divider()

    # methodology comparison table
    st.subheader("Methodology Comparison")
    _, col_method, _ = st.columns([0.5, 9, 0.5])
    with col_method:
        comparison_data = {
            "Item": ["Algorithm", "Num Topics", "Best Coherence",
                     "Strength", "Limitation"],
            "LDA": [
                "Latent Dirichlet Allocation",
                "k=2~7 (auto-optimized)",
                "0.6256 (blog, unibi_mix)",
                "Interpretable, domain-specific strength",
                "No word order or context"
            ],
            "BERTopic": [
                "BERT Embeddings + HDBSCAN",
                "14 (auto-detected)",
                "Semantic similarity-based",
                "Context-aware, handles synonyms",
                "General beauty clusters mixed in"
            ]
        }
        st.dataframe(
            pd.DataFrame(comparison_data),
            use_container_width=True,
            hide_index=True
        )