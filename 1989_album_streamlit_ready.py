from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import umap  # type: ignore
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

st.set_page_config(
    page_title="1989 (Taylor's Version) Explorer",
    page_icon="🎧",
    layout="wide",
)

CSV_URL = "https://raw.githubusercontent.com/Burhanuddin98/Taylor-s-Version-1989/main/features/1989_album_features.csv"

@st.cache_data(show_spinner="Loading feature dataframe…")
def load_main_df() -> pd.DataFrame:
    df_ = pd.read_csv(CSV_URL)
    df_["track_number"] = np.arange(1, len(df_) + 1)
    return df_

df = load_main_df()
track_legend = dict(zip(df["track_number"], df["track"]))

st.title("🎧 Taylor Swift – 1989 (Taylor's Version) Explorer")
st.markdown(
    "Explore high‑level audio & lyric features interactively. Tracks are numbered **1–21** (see legend below)."
)

with st.expander("📖 Track legend"):
    for num, name in track_legend.items():
        st.write(f"**{num}**  {name}")

c1, c2 = st.columns(2)

with c1:
    fig_loud = px.bar(
        df,
        x="track_number",
        y="loudness_LUFS",
        text="track_number",
        labels={"loudness_LUFS": "Integrated LUFS", "track_number": "Track #"},
        title="🔊 Track Loudness (LUFS)",
        color="loudness_LUFS",
        color_continuous_scale="Viridis",
    )
    fig_loud.update_traces(textposition="outside")
    st.plotly_chart(fig_loud, use_container_width=True)

with c2:
    if {"spectral_centroid_Hz", "lexical_diversity"}.issubset(df.columns):
        fig_scatter = px.scatter(
            df,
            x="spectral_centroid_Hz",
            y="lexical_diversity",
            size="duration_sec" if "duration_sec" in df.columns else None,
            color="sentiment" if "sentiment" in df.columns else None,
            text="track_number",
            hover_data={"track": True},
            labels={
                "spectral_centroid_Hz": "Brightness (Hz)",
                "lexical_diversity": "Lexical Diversity",
            },
            color_continuous_scale="RdBu",
            title="✨ Brightness vs. Lexical Diversity (bubble = duration)",
        )
        fig_scatter.update_traces(
            textposition="middle center",
            marker=dict(line=dict(width=2, color="DarkSlateGrey")),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("ℹ️ Scatter skipped – required columns missing in CSV.")

if UMAP_AVAILABLE and {"loudness_LUFS", "spectral_centroid_Hz", "transient_density", "spectral_flatness"}.issubset(df.columns):
    st.markdown("### 🌌 UMAP Sonic Landscape")
    feature_matrix = df[[
        "loudness_LUFS",
        "spectral_centroid_Hz",
        "transient_density",
        "spectral_flatness",
    ]].fillna(0)
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(feature_matrix)
    color_col = "sentiment" if "sentiment" in df.columns else "loudness_LUFS"
    fig_umap = px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        text=df["track_number"],
        color=df[color_col],
        size=df["duration_sec"] * 3 if "duration_sec" in df.columns else None,
        size_max=50,
        color_continuous_scale="Viridis",
        labels={"x": "UMAP 1", "y": "UMAP 2", color_col: color_col.title()},
        title="🌌 UMAP Projection of Track Features",
    )
    fig_umap.update_traces(textposition="top center")
    fig_umap.update_layout(coloraxis_colorbar=dict(title=color_col.title()))
    st.plotly_chart(fig_umap, use_container_width=True)
else:
    st.warning("⚠️ UMAP skipped – install `umap-learn` and verify required columns.")

if UMAP_AVAILABLE and SKLEARN_AVAILABLE:
    st.sidebar.subheader("🌌 Neon Galaxy Settings")
    n_neighbors = st.sidebar.slider("UMAP neighbors", 3, 50, 12)
    n_clusters = st.sidebar.slider("Mood clusters (k)", 2, 5, 3)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features_for_cluster = [c for c in numeric_cols if c not in ("track_number", "popularity")]
    pca = PCA(n_components=min(len(features_for_cluster), 10)).fit_transform(df[features_for_cluster].fillna(0))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(pca)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embedding)
    df_clust = df.copy()
    df_clust[["UMAP1", "UMAP2"]] = embedding
    df_clust["cluster"] = clusters
    mood_map = {
        0: "💜 Midnight Pop Anthems",
        1: "💙 Dreamy Vault Explorers",
        2: "💚 Golden Hour Ballads",
        3: "🧡 Vault Outliers",
        4: "💛 Bonus Cluster",
    }
    df_clust["mood"] = df_clust["cluster"].map(mood_map).fillna("🌟 Unknown")
    st.markdown("## 🌌 Neon Sonic Mood Galaxy")
    fig_galaxy = px.scatter(
        df_clust,
        x="UMAP1",
        y="UMAP2",
        color="mood",
        size=[25] * len(df_clust),
        text="track_number",
        hover_data={"track": True, "mood": True},
        color_discrete_sequence=px.colors.qualitative.Plotly * 3,
        template="plotly_dark",
    )
    fig_galaxy.update_traces(marker=dict(line=dict(width=1, color="white")))
    fig_galaxy.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white",
        title="🌌 Neon Sonic Mood Galaxy (1989 Taylor’s Version)",
    )
    st.plotly_chart(fig_galaxy, use_container_width=True)
    st.markdown("### 📊 Mood cluster – average feature values")
    cluster_summary = (
        df_clust.groupby("mood")[features_for_cluster]
        .mean()
        .round(2)
        .sort_index(key=lambda s: s.map({v: k for k, v in mood_map.items()}))
    )
    st.dataframe(cluster_summary)
else:
    st.sidebar.info("ℹ️ Install `umap-learn` & `scikit-learn` to unlock the Neon Galaxy ✨")
