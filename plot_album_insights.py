
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from math import pi

# —— Config —— #
INPUT_CSV = "album_features_deep_1.csv"
OUTPUT_DIR = "plots"
sns.set_style("darkgrid")
plt.style.use("dark_background")
sns.set_palette("bright")
sns.set_context("talk")

# create output folder
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# —— Load data —— #
df = pd.read_csv(INPUT_CSV)
df = df.loc[:, df.notna().any()]  # drop fully empty cols

# —— 1) Radar fingerprint per track —— #
categories = ['danceability', 'key_strength', 'inharmonicity', 'pitch_std', 'spec_flat']
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

for idx, row in df.iterrows():
    values = row[categories].fillna(0).tolist()
    values += values[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, color='cyan')
    ax.fill(angles, values, alpha=0.3, color='magenta')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', size=10)
    ax.set_yticklabels([])
    ax.set_title(f"{row['track']}", color='lime', size=12)
    fig.savefig(f"{OUTPUT_DIR}/radar_{idx+1:02d}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# —— 2) PCA scatter (colored by tempo) —— #
features = ['danceability','key_strength','inharmonicity','tempo_bpm',
            'beat_fscore','pitch_std','drum_rms','spec_rough',
            'spec_flat','rms_halfspeed','pitch_jitter']
X = df[features].fillna(0).values
pca = PCA(n_components=2)
coords = pca.fit_transform(X)

fig, ax = plt.subplots(figsize=(8,6))
sc = ax.scatter(coords[:,0], coords[:,1],
                c=df['tempo_bpm'], cmap='rainbow', s=100, edgecolor='white')
ax.set_title("PCA of Tracks (Tempo BPM)", color='lime')
ax.set_xlabel("PC1", color='white'); ax.set_ylabel("PC2", color='white')
plt.colorbar(sc, ax=ax, label='Tempo BPM')
fig.savefig(f"{OUTPUT_DIR}/pca_tempo.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# —— 3) t-SNE scatter (colored by danceability) —— #
tsne = TSNE(n_components=2, random_state=0, perplexity=5, n_iter=1000)
tsne_coords = tsne.fit_transform(X)

fig, ax = plt.subplots(figsize=(8,6))
sc = ax.scatter(tsne_coords[:,0], tsne_coords[:,1],
                c=df['danceability'], cmap='Spectral', s=100, edgecolor='white')
ax.set_title("t-SNE of Tracks (Danceability)", color='lime')
ax.set_xlabel("Dim 1", color='white'); ax.set_ylabel("Dim 2", color='white')
plt.colorbar(sc, ax=ax, label='Danceability')
fig.savefig(f"{OUTPUT_DIR}/tsne_dance.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# —— 4) Danceability vs Tempo —— #
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x='tempo_bpm', y='danceability',
                hue='pitch_jitter', size='inharmonicity',
                sizes=(50,300), data=df, ax=ax, palette='bright', edgecolor='white')
ax.set_title("Danceability vs Tempo", color='lime')
ax.set_xlabel("Tempo (BPM)", color='white'); ax.set_ylabel("Danceability", color='white')
ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
fig.savefig(f"{OUTPUT_DIR}/dance_vs_tempo.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# —— 5) Dynamic range boxplot —— #
fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(data=df[['rms_mean','rms_std']].fillna(0), palette='cool', ax=ax)
ax.set_title("Dynamic Range (RMS Mean & STD)", color='lime')
ax.set_ylabel("RMS", color='white')
fig.savefig(f"{OUTPUT_DIR}/dynamic_range.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# —— 6) Harmonic complexity line plot —— #
df_ord = df.sort_values('order')
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(df_ord['order'], df_ord['key_strength'], 'o-', label='Key Strength', color='cyan')
ax.plot(df_ord['order'], df_ord['inharmonicity'], 's-', label='Inharmonicity', color='magenta')
ax.set_title("Harmonic Complexity by Track Order", color='lime')
ax.set_xlabel("Track Order", color='white'); ax.set_ylabel("Value", color='white')
ax.legend()
fig.savefig(f"{OUTPUT_DIR}/harmonic_complexity.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# —— 7) Spectral features heatmap —— #
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df.set_index('track')[features], cmap='magma', ax=ax)
ax.set_title("Spectral Features Heatmap", color='lime')
fig.savefig(f"{OUTPUT_DIR}/spectral_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("All plots saved in", OUTPUT_DIR)

import librosa, librosa.display

# Create a folder for spectrograms
os.makedirs(f"{OUTPUT_DIR}/spectrograms", exist_ok=True)

for idx, fn in enumerate(df['track']):
    path = os.path.join(AUDIO_DIR, fn)
    y, sr = librosa.load(path, sr=22050, mono=True)
    
    # Compute log-power Mel spectrogram (or STFT)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Plot
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel Spectrogram: {fn}", color='white')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/spectrograms/spec_{idx+1:02d}.png", dpi=200)
    plt.close()

import seaborn as sns

# Compute correlation matrix on your cleaned features
corr = df[features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, cbar_kws={"shrink": .8})
plt.title("Feature Correlation Matrix", color='lime')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_correlation_heatmap.png", dpi=200)
plt.close()

from pandas.plotting import scatter_matrix

# Pick a subset of 5–6 key features for clarity
subset = ['tempo_bpm', 'danceability', 'key_strength', 'centroid_mean', 'spectral_flatness', 'rms_mean']

fig = plt.figure(figsize=(12, 12))
axes = scatter_matrix(df[subset].fillna(0), alpha=0.6, figsize=(12,12), diagonal='kde', color='cyan')
plt.suptitle("Pairwise Feature Relationships", color='lime', size=16)
for ax in axes.flatten():
    ax.tick_params(labelcolor='white')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('white')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pairwise_scatter_matrix.png", dpi=200)
plt.close()