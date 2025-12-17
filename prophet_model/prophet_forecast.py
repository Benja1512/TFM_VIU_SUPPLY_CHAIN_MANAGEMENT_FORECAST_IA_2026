# prophet_model/prophet_forecast.py

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Crear carpetas necesarias
os.makedirs('results/graficos', exist_ok=True)
os.makedirs('results/metricas', exist_ok=True)

# Cargar datos
df = pd.read_csv('data/demanda_semanal.csv', sep=';')
df.columns = df.columns.str.strip()
df['semana'] = pd.to_numeric(df['semana'], errors='coerce')
df['demanda'] = pd.to_numeric(df['demanda'], errors='coerce')
df = df.dropna()

# Preparar datos para Prophet
df_prophet = df.rename(columns={'semana': 'ds', 'demanda': 'y'})
df_prophet['ds'] = pd.date_range(start='2023-01-01', periods=len(df), freq='W')

# Entrenar modelo Prophet
model = Prophet()
model.fit(df_prophet)
future = model.make_future_dataframe(periods=0, freq='W')
forecast = model.predict(future)

# Guardar predicciones
forecast[['ds', 'yhat']].to_csv('results/metricas/prophet_forecast_series.csv', index=False)

# Calcular métricas
mae = mean_absolute_error(df_prophet['y'], forecast['yhat'])
rmse = np.sqrt(mean_squared_error(df_prophet['y'], forecast['yhat']))
mape = np.mean(np.abs((df_prophet['y'] - forecast['yhat']) / df_prophet['y'])) * 100

# Guardar métricas
df_metricas = pd.DataFrame({
    'modelo': ['Prophet'],
    'alpha': ['-'],
    'MAE': [mae],
    'RMSE': [rmse],
    'MAPE': [mape]
})
df_metricas.to_csv('results/metricas/prophet_metricas.csv', index=False)

# Mostrar gráfico
plt.figure(figsize=(12, 6))
plt.plot(df_prophet['ds'], df_prophet['y'], label='Demanda real', marker='o')
plt.plot(forecast['ds'], forecast['yhat'], label='Pronóstico Prophet', linestyle='--')
plt.title('Pronóstico con Prophet')
plt.xlabel('Fecha (Semana)')
plt.ylabel('Demanda')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/graficos/prophet_forecast.png')
plt.show()

# =====================================================
# NUEVO BLOQUE: Guardar archivo de escenarios para políticas
# =====================================================
os.makedirs('results/escenarios', exist_ok=True)
df_escenarios = forecast[['yhat']].rename(columns={'yhat': 'base_prophet'})
df_escenarios.to_csv('results/escenarios/escenarios_prophet.csv', index=False)
print("✅ Escenarios base guardados en: results/escenarios/escenarios_prophet.csv")

# Confirmación final
print("✅ Prophet ejecutado correctamente")
print("📊 MAE:", round(mae, 2), "| RMSE:", round(rmse, 2), "| MAPE:", round(mape, 2), "%")
