# analysis_progress.py  –  talkative album analyser  (Python ≥3.9)

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Environment tweaks (before heavy imports)
# ──────────────────────────────────────────────────────────────────────────────
import os, sys, time, re
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # no 32-thread storms
os.environ.setdefault("MKL_NUM_THREADS",    "1")

import numpy as np, pandas as pd, librosa, matplotlib
matplotlib.use("Agg")                        # never pops a window
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta
from matplotlib import colors

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Config knobs
# ──────────────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
AUDIO_DIR  = ROOT / "audio"
OUT_CSV    = ROOT / "album_features.csv"

SR         = 11_025      # down-sample to speed up ← set 22_050 if you want HD
CLIP_SEC   = 30          # analyse only the middle 30 s  (set 0 for full track)
N_FFT      = 1024
HOP        = 256

PRINT_EVERY = 1          # seconds between status prints

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Helper
# ──────────────────────────────────────────────────────────────────────────────
def stats(arr: np.ndarray, tag: str):
    return {f"{tag}_{m}": float(v) for m, v in
            dict(mean=arr.mean(), std=arr.std(ddof=0), max=arr.max()).items()}

def kw(fun, **kwarg):            # cope with librosa 1.0 kwargs-only API
    return fun(**kwarg)

def nice_time(s):
    return str(timedelta(seconds=int(s)))

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Gather files
# ──────────────────────────────────────────────────────────────────────────────
files = sorted(list(AUDIO_DIR.glob("*.wav")) + list(AUDIO_DIR.glob("*.mp3")))
if not files:
    sys.exit(f"❌  No .wav or .mp3 in {AUDIO_DIR}")

print(f"🔍  Found {len(files)} audio files\n")

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Crunch
# ──────────────────────────────────────────────────────────────────────────────
rows, t_start = [], time.perf_counter()

for i, f in enumerate(files, 1):
    lap0 = time.perf_counter()
    y, _ = librosa.load(f, sr=SR, mono=True)

    if CLIP_SEC:
        mid   = len(y)//2
        half  = CLIP_SEC * SR // 2
        y     = y[mid-half: mid+half]

    mfcc      = kw(librosa.feature.mfcc, y=y, sr=SR, n_mfcc=13,
                   n_fft=N_FFT, hop_length=HOP)
    centroid  = kw(librosa.feature.spectral_centroid, y=y, sr=SR,
                   n_fft=N_FFT, hop_length=HOP)
    bandwidth = kw(librosa.feature.spectral_bandwidth, y=y, sr=SR,
                   n_fft=N_FFT, hop_length=HOP)
    rms       = kw(librosa.feature.rms, y=y, frame_length=N_FFT, hop_length=HOP)
    tempo, beats = kw(librosa.beat.beat_track, y=y, sr=SR, hop_length=HOP)
    onset_env    = kw(librosa.onset.onset_strength, y=y, sr=SR,
                      hop_length=HOP)

    order = int(re.match(r"^(\d+)", f.stem).group(1)) if re.match(r"^\d+", f.stem) else i

    row = dict(track=f.name, order=order, tempo_bpm=float(tempo),
               beats=len(beats), onset_env_mean=float(onset_env.mean()))
    row |= stats(mfcc,      "mfcc")
    row |= stats(centroid,  "centroid")
    row |= stats(bandwidth, "bandwidth")
    row |= stats(rms,       "rms")
    rows.append(row)

    # ── live progress line
    lap = time.perf_counter() - lap0
    total = time.perf_counter() - t_start
    print(f"✓ [{i:>2}/{len(files)}] {f.name:<30}  "
          f"track {lap:4.1f}s  |  elapsed {nice_time(total)}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Save table
# ──────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows).sort_values("order").reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print(f"\n✅  Features saved → {OUT_CSV}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 6.  Plots (fast)
# ──────────────────────────────────────────────────────────────────────────────
print("📊  Rendering plots …")

# 6-A energy + tempo arc
plt.figure(figsize=(8,3))
plt.plot(df["order"], df["rms_mean"], marker="o", label="RMS")
ax2 = plt.gca().twinx()
ax2.bar(df["order"], df["tempo_bpm"], alpha=.3, label="BPM")
plt.title("Energy & tempo arc")
plt.xlabel("Track #")
plt.tight_layout()
plt.savefig("album_arc.png", dpi=300)

# 6-B radar fingerprint
axes = ["centroid_mean","bandwidth_mean","rms_mean",
        "tempo_bpm","onset_env_mean","mfcc_std"]
album_avg = df[axes].mean()
chart_pop = pd.Series([2200,1200,0.05,118,0.3,150], index=axes)

def radar(vals, lab, col):
    ang = np.linspace(0, 2*np.pi, len(axes), endpoint=False)
    plt.polar(np.append(ang, ang[0]),
              np.append(vals, vals[0]),
              label=lab, color=col, linewidth=2, alpha=.7)

plt.figure(figsize=(6,6))
radar(album_avg.values, "Album", "tab:blue")
radar(chart_pop.values, "Chart pop", "tab:orange")
plt.xticks(np.linspace(0,2*np.pi,len(axes),endpoint=False), axes, fontsize=8)
plt.title("Psychoacoustic fingerprint")
plt.legend(loc="upper right"); plt.tight_layout()
plt.savefig("album_fingerprint.png", dpi=300)

# 6-C heat map
Z = (df[["centroid_mean","bandwidth_mean","rms_mean",
         "tempo_bpm","mfcc_std"]] - df[["centroid_mean","bandwidth_mean",
         "rms_mean","tempo_bpm","mfcc_std"]].mean())/df[["centroid_mean",
         "bandwidth_mean","rms_mean","tempo_bpm","mfcc_std"]].std(ddof=0)

plt.figure(figsize=(6, 0.6*len(df)))
norm = getattr(colors, "CenteredNorm", colors.TwoSlopeNorm)(vcenter=0)
plt.imshow(Z, aspect="auto", cmap="coolwarm", norm=norm)
plt.yticks(range(len(df)), df["track"], fontsize=7)
plt.xticks(range(Z.shape[1]), Z.columns, rotation=45, ha="right")
plt.colorbar(label="Z-score")
plt.title("Track traits (z-score)")
plt.tight_layout()
plt.savefig("feature_heatmap.png", dpi=300)

print("🎉  Plots written: album_arc.png, album_fingerprint.png, feature_heatmap.png")
