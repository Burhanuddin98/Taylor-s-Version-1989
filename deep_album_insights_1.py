
# deep_album_insights.py  –  mega-suite album analyser with resumable checkpoints
# ----------------------------------------------------
# NOTE: run inside the same folder that already has /audio and album_features.csv
# ----------------------------------------------------
import os
# Silence Essentia warnings
os.environ.setdefault("ESSENTIA_LOG_LEVEL", "ERROR")

import sys, warnings, re
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths
ROOT     = Path(__file__).parent
AUDIO_DIR = ROOT / "audio"
BASE_CSV = ROOT / "album_features.csv"
DEEP_CSV = ROOT / "album_features_deep_1.csv"

# Load base or resume from deep CSV
if DEEP_CSV.exists():
    df = pd.read_csv(DEEP_CSV)
    print(f"▶ Resuming from {DEEP_CSV.name}")
else:
    if not BASE_CSV.exists():
        sys.exit("Run the earlier feature script first to create album_features.csv")
    df = pd.read_csv(BASE_CSV)
    print(f"▶ Starting from {BASE_CSV.name}")

# Helper to save state after each block
def save_state(step_name):
    df.to_csv(DEEP_CSV, index=False)
    print(f"✅ Completed {step_name}, saved to {DEEP_CSV.name}\n")

# 1. Essentia  (danceability, key strength, inharmonicity)
if "danceability" not in df.columns:
    try:
        import essentia.standard as ess
        print("Running Essentia block…")

        # Initialize Essentia algorithms
        dance_fn    = ess.Danceability()
        key_fn      = ess.KeyExtractor()
        spectrum_fn = ess.Spectrum()
        peaks_fn    = ess.SpectralPeaks(magnitudeThreshold=0.0001, minFrequency=10, maxFrequency=22050)
        inh_fn      = ess.Inharmonicity()

        dance, ks_list, inh = [], [], []

        for fn in tqdm(df["track"], desc="Essentia features"):
            path = str(AUDIO_DIR / fn)
            try:
                # Load mono audio
                y = ess.MonoLoader(filename=path)()
                if len(y) < 44100:  # Less than 1 second
                    raise ValueError("Audio too short (<1s)")

                # Danceability
                try:
                    dance_value = dance_fn(y)[0]
                except Exception:
                    print(f"⚠️ Danceability failed for {fn}")
                    dance_value = np.nan

                # Key strength
                try:
                    _, _, ks_value = key_fn(y)
                except Exception:
                    print(f"⚠️ KeyExtractor failed for {fn}")
                    ks_value = np.nan

                # Inharmonicity
                inh_value = np.nan  # Default in case all fail
                for segment in [y[:65536], y]:  # First 64k samples then full
                    try:
                        spec = spectrum_fn(segment)
                        freqs, mags = peaks_fn(spec)
                        if len(freqs) > 0 and len(mags) > 0:
                            inh_value = inh_fn(freqs, mags)
                            break  # Success, stop retrying
                        else:
                            print(f"⚠️ No spectral peaks for {fn} segment")
                    except Exception:
                        continue

                if np.isnan(inh_value):
                    print(f"⚠️ Inharmonicity failed for {fn}")

            except Exception as e:
                print(f"❌ Skipping {fn} due to error: {e}")
                dance_value, ks_value, inh_value = np.nan, np.nan, np.nan

            dance.append(dance_value)
            ks_list.append(ks_value)
            inh.append(inh_value)

        # Assign to dataframe
        df["danceability"]  = dance
        df["key_strength"]  = ks_list
        df["inharmonicity"] = inh
        save_state("Essentia")

    except ImportError:
        warnings.warn("Essentia not installed – skipping")
else:
    print("Skipping Essentia (already computed)\n")



# 2. Tempo & Downbeat estimation (Librosa fallback)
if "tempo_rnn" not in df.columns:
    try:
        import librosa
        print("Running Librosa tempo block…")

        tempos, down_conf = [], []

        for fn in tqdm(df["track"], desc="Librosa tempo features"):
            path = str(AUDIO_DIR / fn)
            try:
                y, sr = librosa.load(path, sr=None, mono=True)
                # Use librosa's beat tracker
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
                tempos.append(float(tempo))

                # Approximate downbeat confidence as % of beats with even spacing
                if len(beats) > 2:
                    beat_intervals = np.diff(beats)
                    deviation = np.std(beat_intervals) / np.mean(beat_intervals)
                    downbeat_conf = max(0.0, 1.0 - deviation)  # 1.0 = perfect spacing
                else:
                    downbeat_conf = np.nan

                down_conf.append(downbeat_conf)
            except Exception as e:
                print(f"⚠️  Librosa tempo error on {fn}: {e}")
                tempos.append(np.nan)
                down_conf.append(np.nan)

        df["tempo_rnn"]     = tempos
        df["downbeat_conf"] = down_conf
        save_state("Librosa tempo")
    except ImportError:
        warnings.warn("Librosa not installed – skipping tempo & downbeat")
else:
    print("Skipping tempo & downbeat (already computed)\n")



# 3. CREPE  (pitch-trace variability)
if "pitch_std" not in df.columns:
    try:
        import librosa, crepe
        print("Running CREPE block…")
        pitch_var = []
        for fn in tqdm(df["track"]):
            path = str(AUDIO_DIR / fn)
            y, sr = librosa.load(path, sr=16000, mono=True)
            mid = len(y)//2
            clip = y[mid-240000:mid+240000]
            _, freq, conf, _ = crepe.predict(clip, sr, step_size=50, viterbi=True)
            freq = freq[conf>0.3]
            pitch_var.append(np.std(freq) if len(freq)>0 else np.nan)

        df["pitch_std"] = pitch_var
        save_state("CREPE")
    except ImportError:
        warnings.warn("CREPE not installed – skipping")
else:
    print("Skipping CREPE (already computed)\n")

# 4. Percussive RMS via Librosa HPSS (fast alternative to Spleeter)
if "drum_rms" not in df.columns:
    try:
        import librosa

        print("Running fast-percussive HPSS block…")
        drum_rms = []
        for fn in tqdm(df["track"]):
            path = str(AUDIO_DIR / fn)
            # load at a moderate SR
            y, sr = librosa.load(path, sr=22050, mono=True)
            # split into harmonic/percussive
            h, p = librosa.effects.hpss(y, margin=4.0)
            # compute RMS on the *percussive* component
            frame_rms = librosa.feature.rms(y=p, frame_length=2048, hop_length=512)[0]
            drum_rms.append(float(np.mean(frame_rms)))

        df["drum_rms"] = drum_rms
        save_state("HPSS (drum RMS)")
    except ImportError:
        warnings.warn("Librosa not installed – skipping HPSS drum RMS")
else:
    print("Skipping HPSS (already computed)\n")


# 5. mir_eval  (beat F-score)
if "beat_fscore" not in df.columns:
    try:
        import mir_eval, librosa
        # ensure beat_proc and bt_proc are defined (re-init if pipeline resumed)
        from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
        beat_proc = RNNBeatProcessor()
        bt_proc   = BeatTrackingProcessor(fps=100)

        print("Running mir_eval block…")
        scores = []
        for fn in tqdm(df["track"]):
            path = str(AUDIO_DIR / fn)
            y, sr = librosa.load(path, mono=True)
            _, est_beats = librosa.beat.beat_track(y=y, sr=sr)
            est_times = librosa.frames_to_time(est_beats, sr=sr)

            # get madmom beats
            act = beat_proc(path)
            mm_beats = bt_proc(act)
            if len(est_times) and len(mm_beats):
                f = mir_eval.beat.f_measure(est_times, mm_beats)
            else:
                f = np.nan
            scores.append(float(f))

        df["beat_fscore"] = scores
        save_state("mir_eval")
    except ImportError:
        warnings.warn("mir_eval not installed – skipping")
else:
    print("Skipping mir_eval (already computed)")

# 6. Librosa replacement for pyACA (spectral roughness, flatness)
if "spec_rough" not in df.columns:
    try:
        import librosa
        print("Running Librosa (roughness & flatness) block…")
        rough, flat = [], []

        for fn in tqdm(df["track"], desc="Librosa features"):
            path = str(AUDIO_DIR / fn)
            try:
                y, sr = librosa.load(path, sr=None, mono=True)

                # Spectral flatness
                flatness = librosa.feature.spectral_flatness(y=y)
                flat.append(float(np.mean(flatness)))

                # Roughness approximation: variance of spectral centroid differences
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                roughness = np.var(np.diff(centroid))
                rough.append(float(roughness))

            except Exception as e:
                print(f"⚠️  Librosa error on {fn}: {e}")
                rough.append(np.nan)
                flat.append(np.nan)

        df["spec_rough"] = rough
        df["spec_flat"]  = flat
        save_state("Librosa roughness/flatness")
    except ImportError:
        warnings.warn("Librosa not installed – skipping")
else:
    print("Skipping Librosa roughness/flatness (already computed)")


# 7. torchvggish  (128-D embeddings)
if "vgg0" not in df.columns:
    try:
        from torchvggish import vggish, vggish_input
        import torch
        print("Running torchvggish block…")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # instantiate and move model to device
        model = vggish()
        model.to(device)

        embs = []
        for fn in tqdm(df["track"]):
            path = str(AUDIO_DIR / fn)
            x = vggish_input.wavfile_to_examples(path)
            x = torch.tensor(x).to(device)
            with torch.no_grad():
                y = model(x)
            embs.append(y.mean(0).cpu().numpy())

        df_emb = pd.DataFrame(np.vstack(embs), columns=[f"vgg{i}" for i in range(128)])
        df = pd.concat([df, df_emb], axis=1)
        save_state("torchvggish")
    except ImportError:
        warnings.warn("torchvggish not installed – skipping")
else:
    print("Skipping torchvggish (already computed)")

# 8. Half-speed RMS via Librosa time-stretch (fast fallback)
if "rms_halfspeed" not in df.columns:
    try:
        import librosa
        print("Running librosa time-stretch block…")
        half_rms = []
        for fn in tqdm(df["track"]):
            path = str(AUDIO_DIR / fn)
            y, sr = librosa.load(path, sr=None, mono=True)
            # stretch to 50% speed
            slow = librosa.effects.time_stretch(y, rate=0.5)
            half_rms.append(float(np.sqrt(np.mean(slow**2))))

        df["rms_halfspeed"] = half_rms
        save_state("librosa time-stretch")
    except ImportError:
        warnings.warn("Librosa not installed – skipping half-speed RMS")
else:
    print("Skipping half-speed RMS (already computed)")


# 9. pyworld  (jitter)
if "pitch_jitter" not in df.columns:
    try:
        import pyworld, librosa
        print("Running pyworld block…")
        jit = []
        for fn in tqdm(df["track"]):
            path = str(AUDIO_DIR / fn)
            y, sr = librosa.load(path, sr=16000)
            f0, t = pyworld.dio(y.astype(np.float64), sr)
            f0 = pyworld.stonemask(y.astype(np.float64), f0, t, sr)
            f0 = f0[f0>0]
            jit.append(float(np.std(np.diff(f0))) if len(f0)>2 else np.nan)

        df["pitch_jitter"] = jit
        save_state("pyworld")
    except ImportError:
        warnings.warn("pyworld not installed – skipping")
else:
    print("Skipping pyworld (already computed)\n")

print("All feature blocks complete.")




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

