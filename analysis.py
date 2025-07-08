# analysis.py  –  robust album analyser  (Python ≥3.9)

# ------------------------------------------------------------------
# 1. Imports & constants
# ------------------------------------------------------------------
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa                        # 0.10.x  OR  1.0.x both OK
import librosa.display

SR        = 22_050
ROOT      = Path(__file__).parent
AUDIO_DIR = ROOT / "audio"
OUT_CSV   = ROOT / "album_features.csv"

# ------------------------------------------------------------------
# 2. Helper functions
# ------------------------------------------------------------------
def summary_stats(arr: np.ndarray, prefix: str) -> dict:
    """Return mean / std / max of a 1-D *or* 2-D array."""
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std" : float(arr.std(ddof=0)),
        f"{prefix}_max" : float(arr.max()),
    }


# We need to cope with librosa 1.0’s keyword-only API
def kw_ok(fun, **kwargs):
    """Call librosa function with keywords regardless of version."""
    return fun(**kwargs)              # works in both pre-1.0 and 1.0


# ------------------------------------------------------------------
# 3. Feature extraction loop
# ------------------------------------------------------------------
rows = []

audio_files = sorted(AUDIO_DIR.glob("*.wav")) + sorted(AUDIO_DIR.glob("*.mp3"))
if not audio_files:
    raise SystemExit(f"No audio found in {AUDIO_DIR!s}")

for idx, wav in enumerate(audio_files, start=1):
    y, sr = librosa.load(wav, sr=SR, mono=True)

    # Track order: use leading digits if present, else fallback to idx
    m = re.match(r"^(\d+)", wav.stem)
    order = int(m.group(1)) if m else idx

    # MFCC & other features — always pass keywords
    mfcc      = kw_ok(librosa.feature.mfcc, y=y, sr=sr, n_mfcc=13)
    centroid  = kw_ok(librosa.feature.spectral_centroid, y=y, sr=sr)
    bandwidth = kw_ok(librosa.feature.spectral_bandwidth, y=y, sr=sr)
    rms       = kw_ok(librosa.feature.rms, y=y)
    chroma    = kw_ok(librosa.feature.chroma_cqt, y=y, sr=sr)
    tempo, beats = kw_ok(librosa.beat.beat_track, y=y, sr=sr)
    onset_env    = kw_ok(librosa.onset.onset_strength, y=y, sr=sr)

    feats = {
        "track"          : wav.name,
        "order"          : order,
        "tempo_bpm"      : float(tempo),
        "beats"          : len(beats),
        "onset_env_mean" : float(onset_env.mean()),
    }
    feats |= summary_stats(mfcc,      "mfcc")
    feats |= summary_stats(centroid,  "centroid")
    feats |= summary_stats(bandwidth, "bandwidth")
    feats |= summary_stats(rms,       "rms")
    feats |= summary_stats(chroma,    "chroma")
    rows.append(feats)

df = pd.DataFrame(rows).sort_values("order").reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print(f"✅  Wrote per-track features → {OUT_CSV}")

# ------------------------------------------------------------------
# 4. Visuals
# ------------------------------------------------------------------
# 4-A  Loudness & tempo arc
plt.figure(figsize=(8, 3))
plt.plot(df["order"], df["rms_mean"], marker="o", label="RMS loudness")
ax2 = plt.gca().twinx()
ax2.bar(df["order"], df["tempo_bpm"], alpha=.3, label="Tempo (BPM)")
plt.title("Energy & tempo arc across the album")
plt.xlabel("Track #")
plt.tight_layout()
plt.savefig("album_arc.png", dpi=300)

# 4-B  Radar fingerprint vs “chart-pop” median
axes = ["centroid_mean", "bandwidth_mean", "rms_mean",
        "tempo_bpm", "onset_env_mean", "mfcc_std"]
album_avg = df[axes].mean()
chart_pop = pd.Series([2200, 1200, 0.05, 118, 0.3, 150], index=axes)  # placeholder

def radar(values, label, color):
    angles = np.linspace(0, 2*np.pi, len(axes), endpoint=False)
    vals   = np.concatenate((values, [values[0]]))
    ang    = np.concatenate((angles, [angles[0]]))
    plt.polar(ang, vals, label=label, linewidth=2, alpha=0.7, color=color)

plt.figure(figsize=(6, 6))
radar(album_avg.values, "This album", "tab:blue")
radar(chart_pop.values, "Billboard median", "tab:orange")
plt.xticks(np.linspace(0, 2*np.pi, len(axes), endpoint=False), axes, fontsize=8)
plt.title("Psychoacoustic fingerprint")
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.savefig("album_fingerprint.png", dpi=300)

# 4-C  Z-score heat-map
from matplotlib import colors
feat_cols = ["centroid_mean", "bandwidth_mean", "rms_mean", "tempo_bpm", "mfcc_std"]
Z = (df[feat_cols] - df[feat_cols].mean()) / df[feat_cols].std(ddof=0)

plt.figure(figsize=(6, 0.6 * len(df)))
norm = getattr(colors, "CenteredNorm", colors.TwoSlopeNorm)(vcenter=0)
plt.imshow(Z, aspect="auto", cmap="coolwarm", norm=norm)
plt.yticks(range(len(df)), df["track"], fontsize=7)
plt.xticks(range(len(feat_cols)), feat_cols, rotation=45, ha="right")
plt.colorbar(label="Z-score (0 = album mean)")
plt.title("Which song brings which trait?")
plt.tight_layout()
plt.savefig("feature_heatmap.png", dpi=300)

print("✅  Plots saved: album_arc.png  album_fingerprint.png  feature_heatmap.png")
