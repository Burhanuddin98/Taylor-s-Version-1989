
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
DEEP_CSV = ROOT / "album_features_deep.csv"

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
        dance_fn    = ess.Danceability()
        key_fn      = ess.KeyExtractor()
        spectrum_fn = ess.Spectrum()
        peaks_fn    = ess.SpectralPeaks()
        inh_fn      = ess.Inharmonicity()

        dance, ks_list, inh = [], [], []
        for fn in tqdm(df["track"]):
            path = str(AUDIO_DIR / fn)
            y = ess.MonoLoader(filename=path)()
            dance.append(dance_fn(y)[0])
            _, _, ks = key_fn(y)
            ks_list.append(ks)
            spec = spectrum_fn(y[:65536])
            freqs, mags = peaks_fn(spec)
            try:
                inh.append(inh_fn(freqs, mags))
            except RuntimeError:
                inh.append(np.nan)

        df["danceability"]  = dance
        df["key_strength"]  = ks_list
        df["inharmonicity"] = inh
        save_state("Essentia")
    except ImportError:
        warnings.warn("Essentia not installed – skipping")
else:
    print("Skipping Essentia (already computed)\n")

# 2. madmom  (RNN tempo, downbeat)
if "tempo_rnn" not in df.columns:
    try:
        from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
        from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
        beat_proc = RNNBeatProcessor()
        bt_proc   = BeatTrackingProcessor(fps=100)
        down_proc = RNNDownBeatProcessor()
        dbn_proc  = DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], fps=100)

        print("Running madmom block…")
        tempos, down_conf = [], []
        for fn in tqdm(df["track"]):
            path = str(AUDIO_DIR / fn)
            act = beat_proc(path)
            beats = bt_proc(act)
            tempos.append(60.0/np.median(np.diff(beats)) if len(beats)>1 else np.nan)
            try:
                _ = down_proc(path)
                _ = dbn_proc(_)
                down_conf.append(np.nan)
            except Exception:
                down_conf.append(np.nan)

        df["tempo_rnn"]     = tempos
        df["downbeat_conf"] = down_conf
        save_state("madmom")
    except ImportError:
        warnings.warn("madmom not installed – skipping")
else:
    print("Skipping madmom (already computed)\n")

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

# 6. pyACA  (roughness, flatness)
if "spec_rough" not in df.columns:
    try:
        import pyACA
        import numpy as np
        print("Running pyACA block…")
        # Check API availability
        if not hasattr(pyACA, 'featureSpectralRoughness'):
            warnings.warn("pyACA featureSpectral functions not available – filling NaN")
            df['spec_rough'] = np.nan
            df['spec_flat']  = np.nan
        else:
            rough, flat = [], []
            for fn in tqdm(df["track"]):
                path = str(AUDIO_DIR / fn)
                y, sr = pyACA.ToolReadAudio(path)
                try:
                    r = pyACA.featureSpectralRoughness(y, sr)
                    f = pyACA.featureSpectralFlatness(y, sr)
                    rough.append(float(np.mean(r)))
                    flat.append(float(np.mean(f)))
                except Exception:
                    rough.append(np.nan)
                    flat.append(np.nan)
            df['spec_rough'] = rough
            df['spec_flat']  = flat
        save_state("pyACA")
    except ImportError:
        warnings.warn("pyACA not installed – skipping")
else:
    print("Skipping pyACA (already computed)")

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
