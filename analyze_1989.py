# 1989_analysis_hybrid_neon.py – Streamlit Ultra Neon Galaxy Suite
# 📦 Requirements: pip install pandas numpy plotly scikit-learn umap-learn hdbscan streamlit seaborn

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
import umap
import hdbscan

# --------------------------
# 🎯 Load Data
# --------------------------
DATA_CSV = "1989_album_features.csv"
df = pd.read_csv(DATA_CSV)
df['track_clean'] = df['track'].str.replace(r"\s*\(Taylor's Version\).*", "", regex=True)
st.set_page_config(page_title="Neon Sonic Mood Galaxy", layout="wide", page_icon="🌌")

# --------------------------
# ⚡ Sidebar Settings
# --------------------------
st.sidebar.title("⚙️ Settings")
n_neighbors = st.sidebar.slider("UMAP Neighbors", min_value=3, max_value=50, value=10)
min_cluster_size = st.sidebar.slider("HDBSCAN Min Cluster Size", min_value=2, max_value=10, value=3)
popularity_scale = st.sidebar.checkbox("Scale Point Size by Popularity", value=True)

# --------------------------
# 🔥 Feature Selection
# --------------------------
features = df.select_dtypes(include=[np.number]).columns.tolist()
features = [col for col in features if col != 'popularity']
X = df[features]
y = df['popularity']

# --------------------------
# 🎯 Popularity Prediction
# --------------------------
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X, y)
df['pred_rf'] = rf.predict(X)

xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
xgb.fit(X, y)
df['pred_xgb'] = xgb.predict(X)

# --------------------------
# 🌌 UMAP + HDBSCAN
# --------------------------
pca = PCA(n_components=10).fit_transform(X)
umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.3, random_state=42)
embedding = umap_model.fit_transform(pca)

clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
clusters = clusterer.fit_predict(embedding)

df['UMAP1'], df['UMAP2'] = embedding[:, 0], embedding[:, 1]
df['cluster'] = clusters

# --------------------------
# ⚡ Reassign Noise
# --------------------------
centroids = df[df['cluster'] != -1].groupby('cluster')[['UMAP1', 'UMAP2']].mean()
for idx, row in df[df['cluster'] == -1].iterrows():
    dists = centroids.apply(lambda c: np.linalg.norm([row['UMAP1'] - c['UMAP1'], row['UMAP2'] - c['UMAP2']]), axis=1)
    nearest_cluster = dists.idxmin()
    df.at[idx, 'cluster'] = nearest_cluster
    df.at[idx, 'reassigned'] = True
df['reassigned'].fillna(False, inplace=True)

# --------------------------
# 🎨 Mood Names + Colors
# --------------------------
mood_names = {0: "💜 Midnight Pop Anthems", 1: "💙 Dreamy Vault Explorers", 2: "💚 Golden Hour Ballads", 3: "🧡 Vault Outliers"}
colors = px.colors.qualitative.Plotly * 3  # Extend color palette
df['mood'] = df['cluster'].map(mood_names)

# --------------------------
# 🌌 Neon Mood Galaxy Plot
# --------------------------
size_col = df['popularity'] * 5 if popularity_scale else 10
fig = px.scatter(
    df,
    x='UMAP1',
    y='UMAP2',
    color='mood',
    size=size_col,
    hover_data=['track_clean', 'popularity', 'pred_rf'],
    color_discrete_sequence=colors,
    template='plotly_dark',
    title="🌌 Neon Sonic Mood Galaxy (*1989 Taylor’s Version*)"
)
fig.update_traces(marker=dict(line=dict(width=0.5, color='white')))
fig.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font_size=24
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------
# 📊 Cluster Summaries
# --------------------------
st.sidebar.markdown("### 📊 Cluster Feature Averages")
summary = df.groupby('mood')[features + ['popularity']].mean().round(2)
st.sidebar.dataframe(summary)

# --------------------------
# 💾 Download Data
# --------------------------
st.sidebar.markdown("### 💾 Download")
csv = summary.to_csv().encode('utf-8')
st.sidebar.download_button("Download Cluster Summary CSV", csv, "cluster_summary.csv", "text/csv")

st.sidebar.success("🎉 Ready to explore the galaxy!")

# --------------------------
# 🎶 Track Table
# --------------------------
st.markdown("### 🎶 Track Details")
st.dataframe(df[['track_clean', 'mood', 'popularity', 'pred_rf', 'pred_xgb']].sort_values('popularity', ascending=False))

