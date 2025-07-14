#taylor39



##1############################################
# 🎧 deep_audio_metrics.py – Audio Metrics Suite
# Extract advanced audio features from tracks
##############################################

import os, sys, subprocess, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Auto-install minimal dependencies
REQUIRED_PKGS = ["librosa", "praat-parselmouth", "pyloudnorm", "pyworld", "matplotlib", "seaborn"]
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
for pkg in REQUIRED_PKGS:
    try: __import__(pkg.replace("-", "_"))
    except ImportError: install(pkg)

import librosa, parselmouth, pyloudnorm, matplotlib.pyplot as plt, seaborn as sns, pyworld

# 📁 Paths
ROOT = Path(__file__).parent
AUDIO_DIR = ROOT / "1989 (Taylor's Version)"
OUTPUT_CSV = ROOT / "1989_audio_metrics.csv"
PLOTS_DIR = ROOT / "neon_audio_plots"
PLOTS_DIR.mkdir(exist_ok=True)

# 🎧 Get audio files
tracks = sorted([f for f in AUDIO_DIR.glob("*.wav")])
if not tracks:
    raise FileNotFoundError("No audio files found in folder!")

# 📝 Dataframe to hold metrics
metrics = {
    "track": [],
    "duration_sec": [],
    "transient_density": [],
    "HNR": [],
    "LUFS": [],
    "tonal_dissonance": [],
    "spectral_flux": [],
    "jitter": [],
    "shimmer": [],
    "rhythmic_complexity": []
}

print("🎧 Extracting advanced audio metrics...")
for track_path in tqdm(tracks):
    metrics["track"].append(track_path.name)
    try:
        y, sr = librosa.load(track_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        metrics["duration_sec"].append(duration)

        # Transient Density
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        metrics["transient_density"].append(len(onsets) / duration)

        # HNR
        snd = parselmouth.Sound(str(track_path))
        hnr = snd.to_harmonicity()
        metrics["HNR"].append(np.nanmean(hnr.values))

        # LUFS
        meter = pyloudnorm.Meter(sr)
        loudness = meter.integrated_loudness(y)
        metrics["LUFS"].append(loudness)

        # Tonal Dissonance
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        metrics["tonal_dissonance"].append(np.std(chroma))

        # Spectral Flux
        S = np.abs(librosa.stft(y))
        flux = np.mean(librosa.onset.onset_strength(S=S, sr=sr))
        metrics["spectral_flux"].append(flux)

        # Jitter & Shimmer (pyworld)
        _f0, t = pyworld.harvest(y.astype(np.float64), sr)
        freqs = _f0[_f0 > 0]
        metrics["jitter"].append(np.std(np.diff(freqs)) if len(freqs) > 1 else np.nan)
        metrics["shimmer"].append(np.nanmean(np.abs(np.diff(freqs) / freqs[:-1])) if len(freqs) > 1 else np.nan)

        # Rhythmic Complexity (onset interval variance)
        if len(onsets) > 1:
            intervals = np.diff(librosa.frames_to_time(onsets, sr=sr))
            metrics["rhythmic_complexity"].append(np.std(intervals))
        else:
            metrics["rhythmic_complexity"].append(np.nan)

    except Exception as e:
        print(f"⚠️ Error processing {track_path.name}: {e}")
        for key in list(metrics.keys())[1:]:
            metrics[key].append(np.nan)

# Save results
df_audio = pd.DataFrame(metrics)
df_audio.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved audio metrics to {OUTPUT_CSV}")

# 🎨 Neon Plots
sns.set(style="dark", palette="magma")
for col in df_audio.columns[1:]:
    plt.figure(figsize=(12, 6), facecolor="black")
    sns.barplot(x="track", y=col, data=df_audio, palette="magma")
    plt.title(f"Neon Plot: {col}", color="white")
    plt.xticks(rotation=90, color="white")
    plt.yticks(color="white")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{col}_neon.png", facecolor="black")
    plt.close()
print(f"🌌 Neon plots saved to {PLOTS_DIR}")
