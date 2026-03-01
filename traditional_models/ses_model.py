import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os

def ejecutar_modelo_ses_excel(archivo_csv='data/demanda_semanal.csv', alpha=0.3):

    # ==============================
    # 📁 CREAR SOLO CARPETA SES
    # ==============================
    base_path = 'results/ses'
    os.makedirs(base_path, exist_ok=True)

    df = pd.read_csv(archivo_csv, sep=';')
    df.columns = df.columns.str.strip()

    # ==============================
    # 🔹 SES MANUAL (IGUAL QUE EXCEL)
    # ==============================

    df['pronostico_ses'] = np.nan

    # Inicialización
    df.loc[0, 'pronostico_ses'] = df.loc[0, 'demanda']

    # Cálculo recursivo
    for t in range(1, len(df)):
        df.loc[t, 'pronostico_ses'] = (
            alpha * df.loc[t-1, 'demanda']
            + (1 - alpha) * df.loc[t-1, 'pronostico_ses']
        )

    # ==============================
    # 🔹 EVALUACIÓN 41–52
    # ==============================
    df_eval = df[df['semana'] >= 41].copy()

    mae = mean_absolute_error(df_eval['demanda'], df_eval['pronostico_ses'])
    rmse = np.sqrt(mean_squared_error(df_eval['demanda'], df_eval['pronostico_ses']))
    mape = np.mean(
        np.abs((df_eval['demanda'] - df_eval['pronostico_ses']) / df_eval['demanda'])
    ) * 100

    print("Evaluación periodo 41–52 (SES estilo Excel)")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # ==============================
    # 💾 GUARDAR MÉTRICAS
    # ==============================
    metricas = pd.DataFrame({
        'modelo': ['SES_manual_excel'],
        'alpha': [alpha],
        'periodo_evaluacion': ['41-52'],
        'MAE': [mae],
        'RMSE': [rmse],
        'MAPE (%)': [mape]
    })

    metricas.to_csv(os.path.join(base_path, 'ses_metricas.csv'), index=False)

    # ==============================
    # 💾 GUARDAR SERIE
    # ==============================
    df[['semana', 'demanda', 'pronostico_ses']].to_csv(
        os.path.join(base_path, 'ses_forecast_series.csv'),
        index=False
    )

    # ==============================
    # 📊 GRÁFICO
    # ==============================
    plt.figure(figsize=(10, 5))
    plt.plot(df['semana'], df['demanda'], label='Demanda real', marker='o')
    plt.plot(df['semana'], df['pronostico_ses'], label='SES (Excel)', linestyle='--', marker='x')
    plt.axvline(x=40, color='red', linestyle=':', label='Inicio evaluación')
    plt.title('SES Manual (Train 1–40 / Test 41–52)')
    plt.xlabel('Semana')
    plt.ylabel('Demanda')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(base_path, 'ses_forecast.png'))
    plt.show()


if __name__ == "__main__":
    ejecutar_modelo_ses_excel()