#!/usr/bin/env python3
# --------------------------------------------
# 1989_ml_embeddings_cpu.py – zero-drama VGGish
# --------------------------------------------
import os, warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""              # 🔒 tell TF “pretend no GPU”
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"             # quiet logs
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"  # disable XLA JIT

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
import librosa                                         # only for resample

ROOT        = Path(__file__).parent
AUDIO_DIR   = ROOT / "1989 (Taylor's Version)"
OUTPUT_CSV  = ROOT / "1989_ml_embeddings.csv"

tracks = sorted(AUDIO_DIR.glob("*.wav"))
if not tracks:
    raise SystemExit("❌  No .wav files found!")

print("🔌  Loading VGGish from TF-Hub (CPU)…")
vggish = hub.load("https://tfhub.dev/google/vggish/1")   # one-time download → ~/.cache

def embed(file_path: Path) -> np.ndarray | None:
    try:
        wav, sr = sf.read(file_path)
        if wav.ndim > 1:                     # stereo → mono
            wav = wav.mean(axis=1)
        if sr != 16_000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16_000)
        wav = tf.convert_to_tensor(wav, dtype=tf.float32)[tf.newaxis, :]  # [1, N]
        with tf.device("/CPU:0"):
            emb = vggish(wav)                # [1, 128]
        return tf.reduce_mean(emb, axis=0).numpy()
    except Exception as e:
        warnings.warn(f"⚠️ {file_path.name}: {e}")
        return None

rows = []
print("🎧  Extracting embeddings (CPU)…")
for p in tqdm(tracks):
    vec = embed(p)
    if vec is not None:
        rows.append({"track": p.name, **{f"vggish_{i}": v for i, v in enumerate(vec)}})

pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
print(f"✅  Saved {len(rows)} tracks → {OUTPUT_CSV}")
