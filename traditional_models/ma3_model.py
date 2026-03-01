import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os


def ejecutar_modelo_ma3(archivo_csv='data/demanda_semanal.csv'):
    # ==============================
    # 📁 CREAR CARPETA MA3
    # ==============================
    base_path = 'results/ma3'
    os.makedirs(base_path, exist_ok=True)

    # Leer datos
    df = pd.read_csv(archivo_csv, sep=';')
    df.columns = df.columns.str.strip()

    # ==============================
    # 🔹 MEDIA MÓVIL 3
    # ==============================
    df['ma3'] = df['demanda'].rolling(window=3).mean()

    # ==============================
    # 🔹 TEST 41–52
    # ==============================
    df_test = df[df['semana'] >= 41].copy()

    # Métricas
    mae = mean_absolute_error(df_test['demanda'], df_test['ma3'])
    rmse = np.sqrt(mean_squared_error(df_test['demanda'], df_test['ma3']))

    # 🔹 MAPE manual (más control académico)
    mape = np.mean(
        np.abs((df_test['demanda'] - df_test['ma3']) / df_test['demanda'])
    ) * 100

    print("Evaluación periodo 41–52")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # ==============================
    # 💾 GUARDAR MÉTRICAS
    # ==============================
    metricas = pd.DataFrame({
        'modelo': ['MA-3'],
        'ventana': [3],
        'periodo_evaluacion': ['41-52'],
        'MAE': [mae],
        'RMSE': [rmse],
        'MAPE (%)': [mape]
    })

    metricas.to_csv(os.path.join(base_path, 'ma3_metricas.csv'), index=False)

    # ==============================
    # 💾 SERIE COMPLETA
    # ==============================
    df[['semana', 'demanda', 'ma3']].rename(
        columns={'ma3': 'pronostico_ma3'}
    ).to_csv(os.path.join(base_path, 'ma3_forecast_series.csv'), index=False)

    # ==============================
    # 📊 GRÁFICO
    # ==============================
    plt.figure(figsize=(10, 5))
    plt.plot(df['semana'], df['demanda'], label='Demanda real', marker='o')
    plt.plot(df['semana'], df['ma3'], label='Pronóstico MA-3', linestyle='--', marker='x')
    plt.axvline(x=40, color='red', linestyle=':', label='Inicio evaluación')
    plt.title('Pronóstico MA-3 (Train 1–40 / Test 41–52)')
    plt.xlabel('Semana')
    plt.ylabel('Demanda')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(base_path, 'ma3_forecast.png'))
    plt.show()


if __name__ == "__main__":
    ejecutar_modelo_ma3()