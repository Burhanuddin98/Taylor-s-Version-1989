# v7_neon_explorer_rewrite.py – Neon Reputation Audio Explorer (Fixed)
# --------------------------------------------------------------------
# 🖤 PySide6 GUI + Matplotlib + Sounddevice (no PyQtGraph, fixed Album Terrain)

import sys
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import sounddevice as sd
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QListWidget, QLabel)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Paths
DATA_CSV = "1989_album_features.csv"
AUDIO_DIR = "1989 (Taylor's Version)"

# Load data
df = pd.read_csv(DATA_CSV)
df['track_clean'] = df['track'].str.replace(r"\s*\(Taylor's Version\).*", "", regex=True)
audio_cache = {}

# Preload audio
for _, row in df.iterrows():
    track = row['track']
    path = os.path.join(AUDIO_DIR, track)
    if os.path.isfile(path):
        y, sr = librosa.load(path, sr=None, mono=True)
        audio_cache[track] = (y, sr)
    else:
        print(f"⚠️ Missing file: {path}")

class NeonExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎤 Neon Reputation Audio Explorer")
        self.setGeometry(100, 100, 1200, 800)
        self.current_index = 0
        self.playing = False

        # Neon Reputation palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(0, 0, 0))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        self.setPalette(palette)

        # Widgets
        self.track_list = QListWidget()
        for _, row in df.iterrows():
            self.track_list.addItem(row['track_clean'])
        self.track_list.currentRowChanged.connect(self.load_track)

        self.plot_area = FigureCanvas(plt.figure(facecolor='black'))

        self.info_label = QLabel("Select a track to begin")
        self.info_label.setFont(QFont('Arial', 14))
        self.info_label.setAlignment(Qt.AlignCenter)

        btn_spectrogram = QPushButton("🌈 Spectrogram")
        btn_energy = QPushButton("📈 Energy")
        btn_terrain = QPushButton("🌌 Album Terrain")
        btn_play = QPushButton("▶️ Play/Pause")

        btn_spectrogram.clicked.connect(self.show_spectrogram)
        btn_energy.clicked.connect(self.show_energy)
        btn_terrain.clicked.connect(self.show_album_terrain)
        btn_play.clicked.connect(self.toggle_playback)

        for btn in [btn_spectrogram, btn_energy, btn_terrain, btn_play]:
            btn.setStyleSheet("background-color: #ff007f; color: white; font-weight: bold;")

        button_layout = QHBoxLayout()
        for btn in [btn_spectrogram, btn_energy, btn_terrain, btn_play]:
            button_layout.addWidget(btn)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.info_label)
        main_layout.addWidget(self.plot_area)
        main_layout.addLayout(button_layout)

        central_widget = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(self.track_list, 1)
        layout.addLayout(main_layout, 4)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def load_track(self, index):
        if index < 0 or index >= len(df): return
        self.current_index = index
        track_name = df.iloc[index]['track_clean']
        self.info_label.setText(f"🎵 {track_name}")

    def show_spectrogram(self):
        track = df.iloc[self.current_index]['track']
        y, sr = audio_cache.get(track, (None, None))
        if y is None:
            self.info_label.setText("⚠️ Audio not found")
            return

        self.plot_area.figure.clf()
        ax = self.plot_area.figure.add_subplot(111)
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)) + 1e-6, ref=np.max)
        img = librosa.display.specshow(S, sr=sr, hop_length=512, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
        ax.set_title("Spectrogram", color='white')
        ax.set_xlabel("Time (s)", color='white')
        ax.set_ylabel("Frequency (Hz)", color='white')
        ax.tick_params(colors='white')
        self.plot_area.figure.colorbar(img, ax=ax, format='%+2.0f dB')
        self.plot_area.draw()

    def show_energy(self):
        track = df.iloc[self.current_index]['track']
        y, sr = audio_cache.get(track, (None, None))
        if y is None:
            self.info_label.setText("⚠️ Audio not found")
            return

        self.plot_area.figure.clf()
        ax = self.plot_area.figure.add_subplot(111)
        rms = librosa.feature.rms(y=y)[0]
        ax.plot(rms, color='#00ffcc')
        ax.set_title("RMS Energy", color='white')
        ax.set_xlabel("Frame", color='white')
        ax.set_ylabel("RMS", color='white')
        ax.tick_params(colors='white')
        self.plot_area.draw()

    def show_album_terrain(self):
        self.plot_area.figure.clf()
        ax = self.plot_area.figure.add_subplot(111)

        # Build 2D terrain matrix
        max_len = max([audio_cache[track][0].shape[0] for track in df['track'] if track in audio_cache])
        terrain = []
        for _, row in df.iterrows():
            track = row['track']
            y, sr = audio_cache.get(track, (None, None))
            if y is None:
                terrain.append(np.zeros((128,)))
                continue
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
            S_dB = librosa.power_to_db(S, ref=np.max)
            mean_power = np.mean(S_dB, axis=1)
            terrain.append(mean_power)

        terrain_matrix = np.stack(terrain)
        img = ax.imshow(terrain_matrix, aspect='auto', cmap='magma', origin='lower')
        ax.set_title("Album Terrain - Mean Mel Power", color='white')
        ax.set_xlabel("Mel Frequency Bands", color='white')
        ax.set_ylabel("Track Index", color='white')
        ax.tick_params(colors='white')
        self.plot_area.figure.colorbar(img, ax=ax, format='%+2.0f dB')
        self.plot_area.draw()

    def toggle_playback(self):
        track = df.iloc[self.current_index]['track']
        y, sr = audio_cache.get(track, (None, None))
        if y is None:
            self.info_label.setText("⚠️ Audio not found")
            return
        if not self.playing:
            sd.play(y, sr)
            self.playing = True
            self.info_label.setText("▶️ Playing")
        else:
            sd.stop()
            self.playing = False
            self.info_label.setText("⏸️ Paused")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NeonExplorer()
    window.showMaximized()
    sys.exit(app.exec())
