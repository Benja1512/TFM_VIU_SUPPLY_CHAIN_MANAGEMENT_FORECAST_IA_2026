import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import numpy as np

def ejecutar_modelo_ses(archivo_csv='data/demanda_semanal.csv', smoothing_alpha=0.3):
    # Crear carpetas si no existen
    os.makedirs('results/graficos', exist_ok=True)
    os.makedirs('results/metricas', exist_ok=True)

    # Leer datos
    df = pd.read_csv(archivo_csv, sep=';')
    df.columns = df.columns.str.strip()

    # Aplicar SES
    modelo = SimpleExpSmoothing(df['demanda'])
    modelo_ajustado = modelo.fit(smoothing_level=smoothing_alpha, optimized=False)
    df['pronostico_ses'] = modelo_ajustado.fittedvalues

    # Calcular métricas
    mae = mean_absolute_error(df['demanda'][1:], df['pronostico_ses'][1:])
    rmse = np.sqrt(mean_squared_error(df['demanda'][1:], df['pronostico_ses'][1:]))

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Guardar métricas en CSV
    metricas = pd.DataFrame({
        'modelo': ['SES'],
        'alpha': [smoothing_alpha],
        'MAE': [mae],
        'RMSE': [rmse]
    })
    metricas.to_csv('results/metricas/ses_metricas.csv', index=False)

    # Guardar la serie completa para análisis comparativos
    df[['semana', 'demanda', 'pronostico_ses']].to_csv('results/metricas/ses_forecast_series.csv', index=False)

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(df['semana'], df['demanda'], label='Demanda real', marker='o')
    plt.plot(df['semana'], df['pronostico_ses'], label='Pronóstico SES', linestyle='--', marker='x')
    plt.title('Pronóstico con Suavizamiento Exponencial Simple (SES)')
    plt.xlabel('Semana')
    plt.ylabel('Demanda')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/graficos/ses_forecast.png')
    plt.show()

if __name__ == "__main__":
    ejecutar_modelo_ses()
