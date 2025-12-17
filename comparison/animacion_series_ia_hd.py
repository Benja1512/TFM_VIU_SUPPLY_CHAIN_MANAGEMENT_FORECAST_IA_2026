import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import platform
import subprocess

def animar_series_ia_hd():
    # Leer resultados comparativos
    df = pd.read_csv("results/metricas/comparacion_series_ia.csv")

    # Configuración de la figura (resolución alta para 1080p)
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)  # 1920x1080 px

    ax.set_xlim(df['semana'].min(), df['semana'].max())
    ax.set_ylim(df[['real','prophet','random_forest','ann']].min().min(),
                df[['real','prophet','random_forest','ann']].max().max())

    line_real, = ax.plot([], [], 'ko-', label="Real")
    line_prophet, = ax.plot([], [], 'bx--', label="Prophet")
    line_rf, = ax.plot([], [], 'gs--', label="Random Forest")
    line_ann, = ax.plot([], [], 'rd--', label="ANN")

    ax.set_title("Evolución de Predicciones IA vs Real (HD)", fontsize=18)
    ax.set_xlabel("Semana", fontsize=14)
    ax.set_ylabel("Demanda", fontsize=14)
    ax.legend(loc="upper left", frameon=True, fontsize=12)
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

    # Animación lenta y clara (1 frame/segundo)
    ani = animation.FuncAnimation(
        fig, animate, frames=len(df), init_func=init, blit=True, interval=1200
    )

    os.makedirs("results/graficos", exist_ok=True)
    mp4_path = os.path.abspath("results/graficos/animacion_series_ia_hd.mp4")

    # Guardar en MP4 en alta resolución
    ani.save(mp4_path, writer="ffmpeg", fps=1, dpi=200)

    plt.close()

    print("\n🎬 MP4 HD guardado en:", mp4_path)

    # Abrir automáticamente el MP4
    try:
        if platform.system() == "Windows":
            subprocess.run(["explorer", mp4_path])
        elif platform.system() == "Darwin":  # macOS
            subprocess.call(["open", mp4_path])
        else:  # Linux
            subprocess.call(["xdg-open", mp4_path])
    except Exception as e:
        print("⚠️ No se pudo abrir automáticamente el MP4:", e)


if __name__ == "__main__":
    animar_series_ia_hd()
