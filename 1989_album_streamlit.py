#   conda activate taylor_ml



# 1989_album_streamlit.py – Interactive Album Explorer (Mel-Spectrogram + Waveform)
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import librosa, librosa.display
import matplotlib.pyplot as plt
import urllib.request
from sklearn.decomposition import PCA

# Automatically download the CSV if missing
features_dir = Path(__file__).parent / "features"
csv_path = features_dir / "1989_album_features.csv"
csv_url = "https://raw.githubusercontent.com/Burhanuddin98/Taylor-s-Version-1989/main/features/1989_album_features.csv"  # 👈 Replace with your actual raw URL

if not csv_path.exists():
    features_dir.mkdir(parents=True, exist_ok=True)
    print(f"📥 Downloading {csv_url} to {csv_path}...")
    urllib.request.urlretrieve(csv_url, csv_path)

# Optional dependencies
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Paths
ROOT = Path(__file__).parent
AUDIO_DIR = ROOT / "downsampled"
if not Path("features/1989_album_features.csv").exists():
    st.error("❌ features/1989_album_features.csv not found! Is it in your GitHub repo?")
    st.stop()
else:
    CSV_FILE = Path("features/1989_album_features.csv")

# Load dataset
df = pd.read_csv(CSV_FILE)
df["track_number"] = range(1, len(df) + 1)  # Number tracks 1–21

# Track Legend
track_legend = {row["track_number"]: row["track"] for _, row in df.iterrows()}

# Title
st.title("🎧 Taylor Swift – 1989 (Taylor's Version) Explorer")
st.markdown("Explore audio & lyric features interactively. Tracks are numbered 1–21 (see legend below).")

# Track Legend
with st.expander("📖 Track Legend"):
    for num, name in track_legend.items():
        st.write(f"**{num}**: {name}")

## Sidebar
#st.sidebar.header("🔧 Options")
#selected_track_num = st.sidebar.selectbox("Choose Track (for Visualizers)", df["track_number"])
#selected_track_name = track_legend[selected_track_num]
#track_path = AUDIO_DIR / selected_track_name

# === Interactive Plots ===
col1, col2 = st.columns(2)

with col1:
    fig = px.bar(df, x="track_number", y="loudness_LUFS", text="track_number",
                 labels={"loudness_LUFS": "Integrated LUFS", "track_number": "Track #"},
                 title="🔊 Track Loudness (LUFS)",
                 color="loudness_LUFS", color_continuous_scale="Viridis")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    if "lexical_diversity" in df.columns:
        fig = px.scatter(
            df,
            x="spectral_centroid_Hz",
            y="lexical_diversity",
            size="duration_sec",
            color="sentiment",
            hover_data={"track_number": True, "track": True},
            text="track_number",  # 👈 Show track number in each bubble
            labels={
                "spectral_centroid_Hz": "Brightness (Hz)",
                "lexical_diversity": "Lexical Diversity"
            },
            color_continuous_scale="RdBu"
        )
        fig.update_traces(
            textposition='middle center',  # 👈 Place numbers inside
            marker=dict(line=dict(width=3, color='DarkSlateGrey'))
        )
        fig.update_layout(
            title="✨ Brightness vs Lexical Diversity (Bubble = Duration)"
        )
        st.plotly_chart(fig, use_container_width=True)


if UMAP_AVAILABLE and all(col in df.columns for col in ["loudness_LUFS", "spectral_centroid_Hz", "transient_density", "spectral_flatness"]):
    st.markdown("### 🌌 UMAP Sonic Landscape")
    features = df[["loudness_LUFS", "spectral_centroid_Hz", "transient_density", "spectral_flatness"]].fillna(0)
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(features)

    # Determine coloring column
    if "sentiment" in df.columns and pd.api.types.is_numeric_dtype(df["sentiment"]):
        color_column = "sentiment"
        color_label = "Sentiment Polarity"
    else:
        color_column = "loudness_LUFS"
        color_label = "Loudness (LUFS)"

    fig = px.scatter(
        x=embedding[:, 0], y=embedding[:, 1], text=df["track_number"],
        color=df[color_column],
        size=df["duration_sec"] * 3,  # 👈 Scale size by duration
        size_max=50,                  # 👈 Max size scale
        color_continuous_scale="Viridis",
        labels={"x": "UMAP 1", "y": "UMAP 2", color_column: color_label},
        title="🌌 UMAP Projection of Track Features"
    )

    fig.update_traces(textposition="top center")
    fig.update_layout(coloraxis_colorbar=dict(title=color_label))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠️ UMAP skipped – either not installed or features missing.")



# === 🌌 Neon Galaxy: UMAP + KMeans (Clean Clusters) ===
st.sidebar.subheader("🌌 Neon Galaxy Settings")
if UMAP_AVAILABLE:
    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans

        # Sidebar sliders
        n_neighbors = st.sidebar.slider("UMAP Neighbors", min_value=3, max_value=50, value=10, step=1)
        n_clusters = st.sidebar.slider("Number of Mood Clusters", min_value=2, max_value=5, value=3, step=1)

        st.markdown("## 🌌 Neon Sonic Mood Galaxy (K-Means)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ["track_number", "popularity"]]

        # PCA to compress
        pca = PCA(n_components=min(len(feature_cols), 10)).fit_transform(df[feature_cols].fillna(0))

        # UMAP projection
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.3, random_state=42)
        embedding = reducer.fit_transform(pca)

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(embedding)

        # Assign to dataframe
        df['UMAP1'], df['UMAP2'] = embedding[:, 0], embedding[:, 1]
        df['cluster'] = clusters

        # Map cluster IDs to named moods
        mood_names = {
            0: "💜 Midnight Pop Anthems",
            1: "💙 Dreamy Vault Explorers",
            2: "💚 Golden Hour Ballads",
            3: "🧡 Vault Outliers",
            4: "💛 Bonus Cluster"
        }
        df['mood'] = df['cluster'].map(mood_names).fillna("🌟 Unknown Mood")

        # Plot Neon Galaxy
        fig = px.scatter(
            df,
            x='UMAP1',
            y='UMAP2',
            color='mood',
            size=[25] * len(df),  # 🔥 BIGGER DOTS
            text='track_number',  # 🔥 Track numbers inside
            hover_data={'track': True, 'mood': True},
            color_discrete_sequence=px.colors.qualitative.Plotly * 3,
            template='plotly_dark',
            title="🌌 Neon Sonic Mood Galaxy (*1989 Taylor’s Version*)"
        )
        fig.update_traces(marker=dict(line=dict(width=1, color='white')))
        fig.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            title_font_size=24
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cluster Summary Table
        st.markdown("### 📊 Mood Cluster Feature Summary")
        summary = df.groupby('mood')[feature_cols].mean().round(2)
        st.dataframe(summary)

    except Exception as e:
        st.error(f"❌ Failed to generate Neon Galaxy: {e}")
else:
    st.warning("⚠️ Neon Galaxy skipped – UMAP or KMeans not installed.")




## === Audio Visualizers ===
#st.markdown(f"### 🎵 Audio Visualizers for Track #{selected_track_num}: {selected_track_name}")

#try:
#    y, sr = librosa.load(str(track_path), sr=22050)
#    duration = librosa.get_duration(y=y, sr=sr)

#    # Waveform
#    st.subheader("📈 Waveform")
#    fig, ax = plt.subplots(figsize=(10, 3))
#    librosa.display.waveshow(y, sr=sr, ax=ax, color="steelblue")
#    ax.set_xlabel("Time (s)")
#    ax.set_ylabel("Amplitude")
#    ax.set_title(f"Waveform – {selected_track_name}")
#    st.pyplot(fig)

#    # 2D Mel-Spectrogram
#    st.subheader("🎨 2D Mel-Spectrogram")
#    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#    S_db = librosa.power_to_db(S, ref=np.max)
#    fig, ax = plt.subplots(figsize=(10, 4))
#    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
#    ax.set_title(f"Mel-Spectrogram – {selected_track_name}")
#    fig.colorbar(img, ax=ax, format="%+2.0f dB")
#    st.pyplot(fig)

#except Exception as e:
#    st.error(f"⚠️ Failed to load audio: {e}")


# === Additional Plots from Second CSV ===
SECOND_CSV = ROOT / "1989_album_features.csv"
if SECOND_CSV.exists():
    df2 = pd.read_csv(SECOND_CSV)
    df2["track_number"] = range(1, len(df2) + 1)

    st.header("🎨 Extra Insights from the data:")

    # --- VGG UMAP ---
    vgg_cols = [col for col in df2.columns if col.startswith("vgg")]
    if len(vgg_cols) >= 50 and UMAP_AVAILABLE:
        st.subheader("🌌 Sonic Landscape from VGG Embeddings")
        reducer = umap.UMAP(random_state=42, metric="cosine")
        embedding = reducer.fit_transform(df2[vgg_cols].fillna(0))

        # Set dot size: fixed size or scaled by duration
        dot_size = np.repeat(20, len(df2))  # 👈 All dots size 20
        if "duration_sec" in df2.columns:
            dot_size = df2["duration_sec"] * 4

        # Decide color column and label
        if "danceability" in df2.columns and pd.api.types.is_numeric_dtype(df2["danceability"]):
            color_column = "danceability"
            color_label = "Danceability"
        else:
            color_column = None
            color_label = ""

        fig = px.scatter(
            x=embedding[:, 0], y=embedding[:, 1],
            color=df2[color_column] if color_column else None,
            size=dot_size,
            size_max=50,
            text=df2["track_number"],
            hover_data={"track": df2["track"]},
            labels={"x": "UMAP 1", "y": "UMAP 2", color_column: color_label},
            color_continuous_scale="Turbo",
            title="VGG Embeddings UMAP Projection"
        )

    fig.update_traces(
        marker=dict(line=dict(width=1, color="white")),
        textposition="middle center"
    )
    if color_column:  # Add colorbar title only if color is used
        fig.update_layout(coloraxis_colorbar=dict(title=color_label))

    st.plotly_chart(fig, use_container_width=True)


    # --- Pitch vs Inharmonicity ---
    if {"pitch_jitter", "inharmonicity"}.issubset(df2.columns):
        st.subheader("🎤 Pitch Stability vs Harmonicity")
        
        # Fixed size or scale by duration_sec if available
        dot_size = np.repeat(20, len(df2))  # 👈 All dots size 20
        if "duration_sec" in df2.columns:
            dot_size = df2["duration_sec"] * 3  # 👈 Scaled

        fig = px.scatter(
            df2, x="pitch_jitter", y="inharmonicity",
            color="danceability" if "danceability" in df2.columns else None,
            size=dot_size,
            size_max=40,  # 👈 Allow bigger bubbles
            text="track_number", hover_name="track",
            labels={
                "pitch_jitter": "Pitch Jitter",
                "inharmonicity": "Inharmonicity",
                "danceability": "Danceability"  # 👈 Capitalized label
            },
            color_continuous_scale="Viridis",
            title="Pitch Jitter vs Inharmonicity"
        )
        fig.update_traces(
            textposition="middle center",  # 👈 Numbers inside
            marker=dict(line=dict(width=1.5, color="white"), opacity=0.85)  # 👈 Neon outline & glow
        )
        st.plotly_chart(fig, use_container_width=True)



    # --- Danceability Bar Chart ---
    if "danceability" in df2.columns:
        st.subheader("🕺 Danceability Ranking")
        fig = px.bar(
            df2.sort_values("danceability", ascending=False),
            x="track_number", y="danceability", text="danceability",
            color="danceability", color_continuous_scale="Plasma",
            labels={"danceability": "Danceability", "track_number": "Track #"},
            title="Danceability by Track"
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    # --- Key Strength Bar Chart ---
    if "key_strength" in df2.columns:
        st.subheader("🎹 Tonal Strength (Key Confidence)")
        fig = px.bar(
            df2, x="track_number", y="key_strength", color="key_strength",
            color_continuous_scale="Magma",
            labels={"key_strength": "Key Strength", "track_number": "Track #"},
            title="Key Detection Confidence per Track"
        )
        st.plotly_chart(fig, use_container_width=True)



else:
    st.warning("⚠️ Second CSV (1989_album_features.csv) not found.")


#--- Second Dataset: 1989_album_features.csv ---
