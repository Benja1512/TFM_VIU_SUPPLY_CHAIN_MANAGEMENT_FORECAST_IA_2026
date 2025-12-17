import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os

def ejecutar_modelo_ma3(archivo_csv='data/demanda_semanal.csv'):
    # Crear carpetas si no existen
    os.makedirs('results/graficos', exist_ok=True)
    os.makedirs('results/metricas', exist_ok=True)

    # Leer datos
    df = pd.read_csv(archivo_csv, sep=';')
    df.columns = df.columns.str.strip()

    # Calcular Media Móvil de 3 periodos
    df['ma3'] = df['demanda'].rolling(window=3).mean()

    # Calcular métricas
    mae = mean_absolute_error(df['demanda'][2:], df['ma3'][2:])  # desde índice 2 porque hay NaN al inicio
    rmse = np.sqrt(mean_squared_error(df['demanda'][2:], df['ma3'][2:]))

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Guardar métricas en CSV
    metricas = pd.DataFrame({
        'modelo': ['MA-3'],
        'ventana': [3],
        'MAE': [mae],
        'RMSE': [rmse]
    })
    metricas.to_csv('results/metricas/ma3_metricas.csv', index=False)

    # Guardar serie pronosticada para comparación
    df[['semana', 'demanda', 'ma3']].rename(columns={'ma3': 'pronostico_ma3'}).to_csv(
        'results/metricas/ma3_forecast_series.csv', index=False
    )

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(df['semana'], df['demanda'], label='Demanda real', marker='o')
    plt.plot(df['semana'], df['ma3'], label='Pronóstico MA-3', linestyle='--', marker='x')
    plt.title('Pronóstico con Media Móvil de 3 periodos (MA-3)')
    plt.xlabel('Semana')
    plt.ylabel('Demanda')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/graficos/ma3_forecast.png')
    plt.show()

if __name__ == "__main__":
    ejecutar_modelo_ma3()
