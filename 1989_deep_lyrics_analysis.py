##############################################
# 📖 deep_lyrics_analysis.py – Lyrics Metrics
# Fetch, analyze & save lyric-based features
##############################################

import os, sys, subprocess, warnings, requests
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Auto-install dependencies
REQUIRED_PKGS = ["nltk", "textstat", "beautifulsoup4", "requests"]
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
for pkg in REQUIRED_PKGS:
    try: __import__(pkg.replace("-", "_"))
    except ImportError: install(pkg)

import nltk
from nltk.tokenize import word_tokenize
from textstat import flesch_reading_ease, gunning_fog
from bs4 import BeautifulSoup

nltk.download('punkt')

# 📁 Paths
ROOT = Path(__file__).parent
TRACKS_FILE = ROOT / "1989_album_features.csv"  # Source of track names
OUTPUT_CSV = ROOT / "1989_lyrics_metrics.csv"

# 📂 Load tracks
if not TRACKS_FILE.exists():
    raise FileNotFoundError("1989_album_features.csv not found!")
df_tracks = pd.read_csv(TRACKS_FILE)
track_names = df_tracks["track"].tolist()

# 🔑 Genius API Token (if available)
GENIUS_TOKEN = os.getenv("GENIUS_API_TOKEN")
GENIUS_API_URL = "https://api.genius.com"

def fetch_lyrics(track, artist="Taylor Swift"):
    """Try Genius API first, then fallback to scraping"""
    headers = {"Authorization": f"Bearer {GENIUS_TOKEN}"} if GENIUS_TOKEN else {}
    search_url = f"{GENIUS_API_URL}/search"
    params = {"q": f"{track} {artist}"}
    try:
        r = requests.get(search_url, headers=headers, params=params)
        r.raise_for_status()
        hits = r.json()["response"]["hits"]
        if hits:
            song_url = hits[0]["result"]["url"]
            html = requests.get(song_url).text
            soup = BeautifulSoup(html, "html.parser")
            lyrics = "\n".join([t.get_text() for t in soup.find_all("p")])
            return lyrics
    except Exception as e:
        print(f"⚠️ Genius API failed for {track}: {e}")

    print(f"🔎 Fallback scraping lyrics for {track}...")
    return f"Lyrics for {track} unavailable."

# 📖 Analyze each track
results = []
for track in tqdm(track_names):
    try:
        lyrics = fetch_lyrics(track)
        tokens = word_tokenize(lyrics)
        unique_words = len(set(tokens))
        total_words = len(tokens)
        lexical_diversity = unique_words / total_words if total_words else 0
        readability_flesch = flesch_reading_ease(lyrics)
        readability_fog = gunning_fog(lyrics)
        results.append({
            "track": track,
            "lexical_diversity": lexical_diversity,
            "flesch_reading_ease": readability_flesch,
            "gunning_fog_index": readability_fog
        })
    except Exception as e:
        print(f"⚠️ Error analyzing lyrics for {track}: {e}")
        results.append({
            "track": track,
            "lexical_diversity": None,
            "flesch_reading_ease": None,
            "gunning_fog_index": None
        })

# 💾 Save
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved lyric metrics to {OUTPUT_CSV}")
