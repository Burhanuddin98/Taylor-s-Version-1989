from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
import numpy as np

# Load the pretrained RNN beat detector
beat_proc = RNNBeatProcessor()
bt_proc   = BeatTrackingProcessor(fps=100)

# Run on one file
act   = beat_proc('audio/your_track.wav')
beats = bt_proc(act)

# Compute BPM from median inter-beat interval
if len(beats) > 1:
    bpm = 60.0 / np.median(np.diff(beats))
else:
    bpm = float('nan')

print(f"Detected Beats: {len(beats)} → BPM ≈ {bpm:.2f}")
print("First 5 beat times (s):", beats[:5])
