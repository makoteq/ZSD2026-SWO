import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# --- KONFIGURACJA ---
file_path = "data/radar_log.csv"
playback_speed = 4.0  # 1.0 = czas rzeczywisty, 2.0 = 2x szybciej, 0.5 = slow motion
smooth_window = 10     # Wielkość filtra (ile klatek uśredniać). 1 = brak filtra.
# --------------------

if not os.path.exists(file_path):
    print(f"❌ BŁĄD: Brak pliku {file_path}")
else:
    df = pd.read_csv(file_path)

    # 1. Obliczenia podstawowe
    df['azimuth_rad'] = np.radians(df['azimuth_deg'])
    df['x_raw'] = df['depth_m'] * np.sin(df['azimuth_rad'])
    df['y_raw'] = df['depth_m'] * np.cos(df['azimuth_rad'])

    # 2. FILTR ANALIZY (Średnia krocząca dla wygładzenia trasy)
    # Pomaga, gdy samochód "skacze" o kilka centymetrów na boki
    df['x'] = df['x_raw'].rolling(window=smooth_window, min_periods=1, center=True).mean()
    df['y'] = df['y_raw'].rolling(window=smooth_window, min_periods=1, center=True).mean()

    # Parametry czasu
    df['delta_t'] = df['timestamp_s'].diff().fillna(0)
    df['freq_hz'] = df['delta_t'].apply(lambda x: 1/x if x > 0 else 0)

    # 3. Konfiguracja wykresu
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(right=0.7) 

    # Niebieski punkt to aktualna pozycja, szary to ślad
    scatter = ax.scatter([], [], s=100, zorder=3)
    trail, = ax.plot([], [], 'gray', alpha=0.3, linewidth=1, label='Ślad trasy')

    ax.set_xlim(-60, 60)
    ax.set_ylim(0, 250)
    ax.set_xlabel("Szerokość (X) [m]")
    ax.set_ylabel("Dystans (Y) [m]")
    ax.grid(True, linestyle='--', alpha=0.4)

    # Panel boczny
    info_text = ax.text(1.05, 0.5, '', transform=ax.transAxes, 
                        fontsize=10, family='monospace', verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Logika klatek
    all_frames = np.arange(df['frame'].min(), df['frame'].max() + 1)
    x_hist, y_hist = [], []
    last_valid_data = None

    def update(frame_id):
        global last_valid_data
        frame_data = df[df['frame'] == frame_id]
        
        # Sprawdzamy czy obiekt jest wykryty w tej klatce
        is_detected = not frame_data.empty
        
        if is_detected:
            data = frame_data.iloc[0]
            last_valid_data = data
            color = 'blue'
            status_msg = "✅ TRACKING"
            
            x_hist.append(data['x'])
            y_hist.append(data['y'])
            current_x, current_y = data['x'], data['y']
        else:
            # Jeśli zgubiliśmy obiekt, używamy ostatniej znanej pozycji
            color = 'red'
            status_msg = "⚠️ LOST (No Detection)"
            if last_valid_data is not None:
                current_x, current_y = last_valid_data['x'], last_valid_data['y']
            else:
                current_x, current_y = 0, 0

        # Aktualizacja grafiki
        scatter.set_offsets(np.c_[current_x, current_y])
        scatter.set_color(color)
        trail.set_data(x_hist, y_hist)
        
        # Dane do raportu
        ts = data['timestamp_s'] if is_detected else (last_valid_data['timestamp_s'] if last_valid_data is not None else 0)
        speed = data['speed_kmh'] if is_detected else 0
        
        report = [
            f"--- {status_msg} ---",
            f"Prędkość wyświetlania: {playback_speed}x",
            f"Filtr (okno): {smooth_window} klatek",
            f"",
            f"Frame ID:  {int(frame_id)}",
            f"Timestamp: {ts:.4f} s",
            f"",
            f"--- KINEMATYKA ---",
            f"Speed:     {speed:>6.2f} km/h",
            f"X:         {current_x:>6.2f} m",
            f"Y:         {current_y:>6.2f} m",
            f"",
            f"--- DIAGNOSTYKA ---",
            f"Status:    {'Wykryto' if is_detected else 'Brak sygnału'}",
            f"Sampling:  {data['freq_hz'] if is_detected else 0:>5.1f} Hz"
        ]
        
        info_text.set_text("\n".join(report))
        return scatter, trail, info_text

    # Obliczanie interwału na podstawie prędkości (bazowo 100ms / speed)
    anim_interval = int(100 / playback_speed)

    ani = FuncAnimation(fig, update, frames=all_frames, 
                        interval=anim_interval, repeat=False, blit=False)

    ax.legend(loc='upper left')
    plt.title(f"Analiza Radarowa: {os.path.basename(file_path)}")
    plt.show()