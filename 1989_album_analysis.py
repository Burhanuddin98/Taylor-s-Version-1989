# deep_album_insights.py – Taylor’s Version of album analytics
import os, sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Optional dependencies
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import pyloudnorm as pyln
    LOUDNESS_AVAILABLE = True
except ImportError:
    LOUDNESS_AVAILABLE = False

try:
    from lyricsgenius import Genius
    import nltk, textstat
    from transformers import pipeline
    LYRICS_AVAILABLE = True
except ImportError:
    LYRICS_AVAILABLE = False

# === PATHS ===
ROOT = Path(__file__).parent
AUDIO_DIR = ROOT / "downsampled"
FEATURES_DIR = ROOT / "features"
PLOTS_DIR = ROOT / "plots"
CSV_FILE = FEATURES_DIR / "1989_album_features.csv"

# Ensure output folders exist
FEATURES_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# === HELPER FUNCTIONS ===
def save_plot(fig, name):
    out_path = PLOTS_DIR / name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved: {out_path}")

def get_integrated_lufs(y, sr):
    if LOUDNESS_AVAILABLE:
        meter = pyln.Meter(sr)
        return meter.integrated_loudness(y)
    else:
        rms = np.sqrt(np.mean(y ** 2))
        return 20 * np.log10(rms + 1e-9) - 0.691  # fallback approximation

# === AUDIO FEATURE EXTRACTION ===
print("🎧 Extracting audio features...")
import librosa, librosa.display
tracks = sorted([f for f in AUDIO_DIR.glob("*.wav")])
if not tracks:
    sys.exit(f"❌ No WAV files found in {AUDIO_DIR}")

data = []
for track in tqdm(tracks, desc="Audio Analysis"):
    try:
        y, sr = librosa.load(track, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # Features
        lufs = get_integrated_lufs(y, sr)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        trans_density = len(onsets) / max(duration, 1)
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))

        data.append([track.name, duration, lufs, centroid, trans_density, flatness])
    except Exception as e:
        print(f"⚠️ Skipping {track.name} due to error: {e}")
        data.append([track.name, np.nan, np.nan, np.nan, np.nan, np.nan])

df = pd.DataFrame(data, columns=[
    "track", "duration_sec", "loudness_LUFS",
    "spectral_centroid_Hz", "transient_density", "spectral_flatness"
])

# === LYRICS & NLP ===
if LYRICS_AVAILABLE:
    print("📝 Fetching lyrics & computing NLP metrics...")
    genius = Genius("3UqOf0KWZ4EyGeyDffLZgQPHjux6UNu-heuKokMocwfcoOVqwwwlc_FHAxzq-ZnY", 
                skip_non_songs=True, 
                excluded_terms=["(Remix)", "(Live)"])
    sentiment_model = pipeline("sentiment-analysis")
    nltk.download("punkt")

    lex_div, readability, sentiment = [], [], []
    for track in tqdm(df["track"], desc="Lyrics Analysis"):
        title = track.replace("(Taylor's Version)", "").replace(".wav", "").strip()
        try:
            song = genius.search_song(title, "Taylor Swift")
            if song and song.lyrics:
                words = nltk.word_tokenize(song.lyrics)
                lex_div.append(len(set(words)) / len(words))
                readability.append(textstat.flesch_reading_ease(song.lyrics))
                sent = sentiment_model(song.lyrics[:512])[0]
                sentiment.append(sent["score"] if sent["label"] == "POSITIVE" else -sent["score"])
            else:
                raise ValueError("Lyrics not found")
        except Exception as e:
            print(f"⚠️ Lyrics failed for {title}: {e}")
            lex_div.append(np.nan)
            readability.append(np.nan)
            sentiment.append(np.nan)

    df["lexical_diversity"] = lex_div
    df["readability"] = readability
    df["sentiment"] = sentiment
else:
    print("⚠️ Skipping lyrics analysis – dependencies missing")

# Save features
df.to_csv(CSV_FILE, index=False)
print(f"✅ Features saved to {CSV_FILE}")

# === PLOTS ===
sns.set_theme(style="darkgrid")

# 1. Loudness Plot
fig = plt.figure(figsize=(12, 6))
sns.barplot(x="track", y="loudness_LUFS", data=df, color="skyblue")
plt.xticks(rotation=90, ha="right")
plt.title("Track Loudness (Integrated LUFS)")
plt.axhline(-14, color="red", linestyle="--", label="Spotify Target LUFS")
plt.legend()
save_plot(fig, "loudness_lufs.png")

# 2. Transient Density Bar Race (Optional animation)
fig = plt.figure(figsize=(12, 6))
sns.barplot(x="track", y="transient_density", data=df, color="orange")
plt.xticks(rotation=90, ha="right")
plt.title("Transient Density (Onsets per Second)")
save_plot(fig, "transient_density.png")

# 3. Brightness vs Lexical Diversity Bubble Chart
if LYRICS_AVAILABLE:
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(df["spectral_centroid_Hz"], df["lexical_diversity"],
                s=df["duration_sec"]*5, c=df["sentiment"], cmap="coolwarm", alpha=0.7, edgecolor="k")
    plt.xlabel("Spectral Centroid (Hz)")
    plt.ylabel("Lexical Diversity")
    plt.colorbar(label="Sentiment")
    plt.title("Brightness vs Lexical Diversity (Bubble size = Duration)")
    save_plot(fig, "brightness_vs_lexdiv.png")

# 4. Diverging Sentiment Bars
if LYRICS_AVAILABLE:
    fig = plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values("sentiment")
    colors = df_sorted["sentiment"].apply(lambda x: "green" if x > 0 else "red")
    plt.bar(df_sorted["track"], df_sorted["sentiment"], color=colors)
    plt.xticks(rotation=90, ha="right")
    plt.title("Sentiment Divergence Across Tracks")
    plt.axhline(0, color="black", linewidth=0.8)
    save_plot(fig, "sentiment_divergence.png")

# 5. Lexical Readability Boxplot
if LYRICS_AVAILABLE:
    fig = plt.figure(figsize=(8, 6))
    sns.boxplot(y="readability", data=df)
    plt.title("Lexical Readability (Flesch Score)")
    save_plot(fig, "lexical_readability.png")

# 6. UMAP Sonic Map
if UMAP_AVAILABLE:
    print("🔮 Running UMAP dimensionality reduction...")
    features = df[["loudness_LUFS", "spectral_centroid_Hz", "transient_density", "spectral_flatness"]].fillna(0)
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(features)

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=df["sentiment"] if LYRICS_AVAILABLE else "b", cmap="coolwarm", s=80)
    for i, txt in enumerate(df["track"]):
        plt.annotate(txt, (embedding[i, 0], embedding[i, 1]), fontsize=7)
    plt.title("UMAP Sonic Landscape")
    save_plot(fig, "umap_sonic_map.png")

print("\n🎯 Analysis complete. Check 'features/' and 'plots/' folders.")
