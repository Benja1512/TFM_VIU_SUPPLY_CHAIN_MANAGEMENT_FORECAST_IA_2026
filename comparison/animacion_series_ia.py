import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import platform
import subprocess

def animar_series_ia():
    # Leer resultados comparativos
    df = pd.read_csv("results/metricas/comparacion_series_ia.csv")

    # Configuración de la figura
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_xlim(df['semana'].min(), df['semana'].max())
    ax.set_ylim(df[['real','prophet','random_forest','ann']].min().min(),
                df[['real','prophet','random_forest','ann']].max().max())

    line_real, = ax.plot([], [], 'ko-', label="Real")
    line_prophet, = ax.plot([], [], 'bx--', label="Prophet")
    line_rf, = ax.plot([], [], 'gs--', label="Random Forest")
    line_ann, = ax.plot([], [], 'rd--', label="ANN")

    ax.set_title("Evolución de Predicciones IA vs Real")
    ax.set_xlabel("Semana")
    ax.set_ylabel("Demanda")
    ax.legend(loc="upper left", frameon=True)  # ✅ Leyenda fija arriba izquierda
    ax.grid(True)

    def init():
        line_real.set_data([], [])
        line_prophet.set_data([], [])
        line_rf.set_data([], [])
        line_ann.set_data([], [])
        return line_real, line_prophet, line_rf, line_ann

    def animate(i):
        x = df['semana'][:i]
        line_real.set_data(x, df['real'][:i])
        line_prophet.set_data(x, df['prophet'][:i])
        line_rf.set_data(x, df['random_forest'][:i])
        line_ann.set_data(x, df['ann'][:i])
        return line_real, line_prophet, line_rf, line_ann

    # interval = 1200 ms (animación lenta y clara)
    ani = animation.FuncAnimation(
        fig, animate, frames=len(df), init_func=init, blit=True, interval=1200
    )

    os.makedirs("results/graficos", exist_ok=True)
    gif_path = os.path.abspath("results/graficos/animacion_series_ia.gif")
    mp4_path = os.path.abspath("results/graficos/animacion_series_ia.mp4")

    # Guardar en GIF
    ani.save(gif_path, writer="pillow")
    # Guardar en MP4 (necesita ffmpeg instalado)
    try:
        ani.save(mp4_path, writer="ffmpeg")
        print("\n✅ Animaciones guardadas en:")
        print(" - GIF:", gif_path)
        print(" - MP4:", mp4_path)
    except Exception as e:
        print("⚠️ No se pudo exportar a MP4 (instala ffmpeg si lo necesitas):", e)

    plt.close()

    # Abrir automáticamente el GIF
    try:
        if platform.system() == "Windows":
            subprocess.run(["explorer", gif_path])  # Compatible con Git Bash
        elif platform.system() == "Darwin":  # macOS
            subprocess.call(["open", gif_path])
        else:  # Linux
            subprocess.call(["xdg-open", gif_path])
    except Exception as e:
        print("⚠️ No se pudo abrir automáticamente el GIF:", e)

if __name__ == "__main__":
    animar_series_ia()
